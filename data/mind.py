import urllib.request
import zipfile

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
from typing import Callable
from typing import List
from typing import Optional, Dict, Union

class RawMindDataset(InMemoryDataset, PreprocessingMixin):
    def __init__(
        self,
        root: str,
        split: str = "train",  # or 'dev', 'test'
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        max_seq_len: int = 20,
        category: str = "category",
    ):
        self.split = split
        self.max_seq_len = max_seq_len
        self.category = category
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self):
        return ["behaviors.tsv", "news.tsv"]

    @property
    def processed_file_names(self):
        return f"mind_{self.split}.pt"
    
    def train_test_split(self, behaviors, max_seq_len=20):
        splits = ["train", "eval", "test"]
        sequences = {sp: defaultdict(list) for sp in splits}

        behaviors['history'] = behaviors['history'].fillna("")
        behaviors['history_len'] = behaviors['history'].str.split().apply(len)

        behaviors['capped_len'] = behaviors['history_len'].clip(upper=20)

        behaviors_sorted = behaviors.sort_values(['userId', 'capped_len'], ascending=[True, False])

        unique_behaviors = behaviors_sorted.drop_duplicates(subset=['userId'], keep='first').reset_index(drop=True)

        for idx, row in unique_behaviors.iterrows():
            user_id = row["userId"]
            
            history = list(map(int, row["history"].split()))
            
            if len(history) < 5:
                continue
            
            # Train: 
            train_items = history[:-3]
            sequences["train"]["itemId"].append(train_items)
            sequences["train"]["itemId_fut"].append(history[-3])
            sequences["train"]["userId"].append(user_id)
            
            # Eval: 
            eval_items = history[-(max_seq_len + 2):-2]
            padded_eval = eval_items + [-1] * (max_seq_len - len(eval_items))
            sequences["eval"]["itemId"].append(padded_eval)
            sequences["eval"]["itemId_fut"].append(history[-2])
            sequences["eval"]["userId"].append(user_id)
            
            # Test: 
            test_items = history[-(max_seq_len + 1):-1]
            padded_test = test_items + [-1] * (max_seq_len - len(test_items))
            sequences["test"]["itemId"].append(padded_test)
            sequences["test"]["itemId_fut"].append(history[-1])
            sequences["test"]["userId"].append(user_id)

        for sp in splits:
            #sequences[sp]["userId"] = user_ids
            sequences[sp] = pl.from_dict(sequences[sp])

        return sequences
    
    def _create_article_id_mapping(self, behaviors: pd.DataFrame, news: pd.DataFrame) -> Dict[str, int]:
       
        history_ids = behaviors["history"].dropna().str.split().explode()
        impression_ids = behaviors["impressions"].dropna().str.replace("-", " ").str.split().explode()
        all_ids = pd.concat([news["id"], history_ids, impression_ids]).dropna().unique()
        
        return {article_id: idx for idx, article_id in enumerate(all_ids)}
    
    def _create_user_id_mapping(self, behaviors: pd.DataFrame) -> Dict[str, int]:
        unique_users = behaviors["userId"].unique()
        return {user: idx for idx, user in enumerate(unique_users)}


    def process(self, max_seq_len=20):
        data = HeteroData()

        behaviors_path = os.path.join(self.raw_dir, "MINDsmall_train", "behaviors.tsv")
        news_path = os.path.join(self.raw_dir, "MINDsmall_train", "news.tsv")

        # Load raw files
        behaviors = pd.read_csv(
            behaviors_path,
            sep="\t",
            names=["userId", "timestamp", "history", "impressions"],
            usecols=[1, 2, 3, 4],
        )

        news = pd.read_csv(
            news_path,
            sep="\t",
            names=["id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"],
            usecols=[0, 1, 2, 3, 4],
        )

        user2id = self._create_user_id_mapping(behaviors)
        behaviors["userId"] = behaviors["userId"].map(user2id)  

        article2id = self._create_article_id_mapping(behaviors, news)

        news["itemId"] = news["id"].map(article2id)

        behaviors["history"] = behaviors["history"].fillna("").apply(
            lambda s: " ".join(str(article2id[i]) for i in s.split() if i in article2id)
        )
        behaviors["impressions"] = behaviors["impressions"].fillna("").apply(
            lambda s: " ".join(str(article2id[i.strip("-")]) for i in s.split() if i.strip("-") in article2id)
        )

        sequences = self.train_test_split(behaviors, max_seq_len=max_seq_len)
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"]) for k, v in sequences.items()
        }

        sentences = news.apply(
            lambda row: f"Title: {row['title']}; Abstract: {row['abstract']}; Category: {row['category']}; Subcategory: {row['subcategory']};",
            axis=1,
        )
        item_emb = self._encode_text_feature(sentences)

        news["first_subcategory"] = news["subcategory"].fillna("").apply(lambda x: x.split(";")[0] if x else "Unknown")

        unique_subcats = news["first_subcategory"].unique()
        self.brand_mapping = {i: subcat for i, subcat in enumerate(unique_subcats)}

        subcat_to_id = {subcat: i for i, subcat in self.brand_mapping.items()}

        news["brand_id"] = news["first_subcategory"].map(lambda x: subcat_to_id.get(x, -1))

        brand_ids = news.apply(lambda row: row["brand_id"], axis=1)


        data["item"].x = item_emb
        data["item"].text = np.array(sentences)
        data["item"].is_train = torch.rand(item_emb.shape[0], generator=torch.Generator().manual_seed(42)) > 0.05
        data["item"].brand_id = np.array(
            brand_ids
        )
        data["brand_mapping"] = self.brand_mapping

        self.save([data], self.processed_paths[0])