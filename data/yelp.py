import gzip
import json
import numpy as np
import os
import os.path as osp
import pandas as pd
import polars as pl
import torch

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import download_google_url
from torch_geometric.data import extract_zip
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import fs
from typing import Callable, List, Optional, Dict


def parse(path):
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


class YelpReviews(InMemoryDataset, PreprocessingMixin):
    gdrive_id = "1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G"  # Reuse or change if needed
    gdrive_filename = "P5_data.zip"  # Reuse or change if needed

    def __init__(
        self,
        root: str,
        split: str,  # e.g., 'yelp'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        category="categories",
    ) -> None:
        self.split = split
        self.brand_mapping = {}
        self.category = category
        super(YelpReviews, self).__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.split]

    @property
    def processed_file_names(self) -> str:
        return f"data_{self.split}.pt"

    def download(self) -> None:
        path = download_google_url(self.gdrive_id, self.root, self.gdrive_filename)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, "data")
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def _remap_ids(self, x):
        return x - 1

    def get_brand_name(self, brand_id: int) -> str:
        return self.brand_mapping.get(brand_id, "Unknown")

    def get_brand_mapping(self) -> Dict[int, str]:
        return self.brand_mapping

    def train_test_split(self, max_seq_len=20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}
        user_ids = []
        with open(os.path.join(self.raw_dir, self.split, "sequential_data.txt"), "r") as f:
            for line in f:
                parsed_line = list(map(int, line.strip().split()))
                user_ids.append(parsed_line[0])
                items = [self._remap_ids(id) for id in parsed_line[1:]]

                train_items = items[:-3]
                sequences["train"]["itemId"].append(train_items)
                sequences["train"]["itemId_fut"].append(items[-3])

                eval_items = items[-(max_seq_len + 2): -2]
                sequences["eval"]["itemId"].append(
                    eval_items + [-1] * (max_seq_len - len(eval_items))
                )
                sequences["eval"]["itemId_fut"].append(items[-2])

                test_items = items[-(max_seq_len + 1): -1]
                sequences["test"]["itemId"].append(
                    test_items + [-1] * (max_seq_len - len(test_items))
                )
                sequences["test"]["itemId_fut"].append(items[-1])

        for sp in splits:
            sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])
        return sequences
    

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        with open(os.path.join(self.raw_dir, self.split, "datamaps.json"), "r") as f:
            data_maps = json.load(f)

        sequences = self.train_test_split(max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"]) for k, v in sequences.items()
        }

        # Yelp-specific: Use business_id as key
        biz2id = pd.DataFrame(
            [{"business_id": k, "id": self._remap_ids(int(v))} for k, v in data_maps["item2id"].items()]
        )

        # ✅ Load meta_data.pkl
        #item_data = pd.read_pickle(os.path.join(self.raw_dir, self.split, "meta_data.pkl"))
        item_data = pd.DataFrame(pd.read_pickle(os.path.join(self.raw_dir, self.split, "meta_data.pkl")))
        print(item_data[["name", "categories"]].head())

        item_data = (
            item_data
            .merge(biz2id, on="business_id")
            .sort_values(by="id")
        )
        item_data["name"] = item_data["name"].fillna("Unknown")

        item_data["categories"] = item_data["categories"].apply(
            lambda x: x.split(", ") if isinstance(x, str) else x
        )

        print("Aantal rijen na merge:", len(item_data)) 

        # Fix 'categories' (list type)
        item_data["categories"] = item_data["categories"].apply(
            lambda x: x if isinstance(x, list) and len(x) > 0 else ["Unknown"]
        )

        # Use first category as pseudo-brand
        item_data["brand"] = item_data["categories"].apply(
            lambda cats: cats[0] if isinstance(cats, list) and len(cats) > 0 else "Unknown"
        )

        print("Unieke brands vóór mapping:", item_data["brand"].unique())

        unique_brands = item_data["brand"].unique()
        self.brand_mapping = {i: brand for i, brand in enumerate(unique_brands)}
        brand_to_id = {brand: i for i, brand in self.brand_mapping.items()}
        item_data["brand_id"] = item_data["brand"].map(lambda x: brand_to_id.get(x, -1))

        # Build sentence representation
        sentences = item_data.apply(
            lambda row: "Name: "
            + str(row["name"])
            + "; "
            + "Category: "
            + str(row["brand"])
            + "; "
            + "Location: "
            + str(row.get("city", ""))
            + ", "
            + str(row.get("state", ""))
            + "; "
            + "Rating: "
            + str(row.get("stars", ""))
            + "; ",
            axis=1,
        )

        #brand_ids = item_data["brand_id"]
        brand_ids = item_data.apply(lambda row: row["brand_id"], axis=1)
        item_emb = self._encode_text_feature(sentences)

        data["item"].x = item_emb
        data["item"].text = np.array(sentences)
        data["item"].brand_id = np.array(brand_ids)
        data["brand_mapping"] = self.brand_mapping

        gen = torch.Generator()
        gen.manual_seed(42)
        #data["item"].is_train = torch.rand(item_emb.shape[0], generator=gen) > 0.05

        n_total = item_emb.shape[0]
        permutation = torch.randperm(n_total, generator=gen)

        n_train = int(0.8 * n_total)
        n_eval = int(0.1 * n_total)
        splits = [n_train, n_train + n_eval]

        train_items = permutation[:splits[0]]
        eval_items = permutation[splits[0]:splits[1]]
        test_items = permutation[splits[1]:]
    
        keys = ["is_train", "is_eval", "is_test"]
        for key in keys:
            mask = torch.zeros(n_total, dtype=torch.bool)
            if key == "is_train":
                mask[train_items] = True
            elif key == "is_eval":
                mask[eval_items] = True
            elif key == "is_test":
                mask[test_items] = True
            data["item"][key] = mask

        self.save([data], self.processed_paths[0])

        brand_mapping_path = os.path.join(self.processed_dir, f"brand_mapping_{self.split}.json")
        with open(brand_mapping_path, "w") as f:
            json.dump(self.brand_mapping, f)

