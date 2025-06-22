from collections import defaultdict
from torch import Tensor
import torch
import math
from einops import rearrange
import numpy as np

#######################################################################
import pandas as pd
from rectools.metrics.diversity import IntraListDiversity
from rectools.metrics.distances import PairwiseHammingDistanceCalculator
#######################################################################

def compute_dcg(relevance: list) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))


def compute_ndcg_for_semantic_ids(pred: Tensor, actual: Tensor, k: int) -> float:
    """
    Compute NDCG@k for one example of semantic ID tuples.
    pred: [K, D] tensor — top-k predicted semantic IDs
    actual: [D] tensor — ground truth semantic ID
    """
    actual_tuple = tuple(actual.tolist())  # Convert to hashable tuple
    relevance = [1 if tuple(row.tolist()) == actual_tuple else 0 for row in pred[:k]]
    dcg = compute_dcg(relevance)
    idcg = compute_dcg(sorted(relevance, reverse=True))
    return dcg / idcg if idcg > 0 else 0.0


class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        return self.gini_coefficient(list(freqs.values()))


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()
        self.zero_semantic_count = defaultdict(int)
        self.unknown_category_count = 0
        self.total_category_lookups = 0

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    def accumulate(self, actual: Tensor, top_k: Tensor, tokenizer=None) -> None:
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        for i in range(D):
            match_found, rank = pos_match[..., : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(
                    matched_rank[matched_rank < k]
                )

            match_found, rank = pos_match[..., i : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])

        B = actual.size(0)
        for b in range(B):
            gold_docs = actual[b]
            pred_docs = top_k[b]
            for k in self.ks:
                topk_pred = pred_docs[:k]
                # hits = sum(1 for doc in topk_pred if doc in gold_docs)
                hits = torch.any(torch.all(topk_pred == gold_docs, dim=1)).item()
                self.metrics[f"h@{k}"] += float(hits > 0)
                self.metrics[f"ndcg@{k}"] += compute_ndcg_for_semantic_ids(
                    pred_docs, gold_docs, k
                )
                # if the tokinzer is given then for each prediction find the catergoy and add it to the list and then caclulate the gini coefficient
                if tokenizer is not None:
                    list_gini = []
                    for pred in topk_pred:
                        idx = str(pred.tolist()[:-1])
                        category = tokenizer.map_to_category.get(idx, "UNKNOWN")
                        list_gini.append({"id": idx, "category": category})
                        self.zero_semantic_count[idx] += 1  # ← LOG HERE
                        self.total_category_lookups += 1  # ← total lookups
                        if category == "UNKNOWN":
                            self.unknown_category_count += 1  # ← count UNKNOWNs
                    self.metrics[f"gini@{k}"] += GiniCoefficient().calculate_list_gini(
                        list_gini, key="category"
                    )
                #######################################################################
                if k > 1:
                    # Calculate Intra-List Diversity (ILD)
                    user_ids = [b]*k
                    item_ids = list(range(k))
                    ranking = list(range(1, k + 1))
                    reco = pd.DataFrame({
                        "user_id": user_ids,
                        "rank": ranking,
                        "item_id": item_ids,
                        
                    })  
                    # print(f"ILD DF for user {b} with k {k} : {reco.head()}") 
                    features_df = pd.DataFrame(topk_pred.cpu().numpy(), columns=["feat_1", "feat_2", "feat_3", "feat_4"])     
                    # print(f"Features DF for user {b} with k {k} : {features_df.head()}")     
                    calc = PairwiseHammingDistanceCalculator(features_df)
                    self.metrics[f"ild@{k}"] = IntraListDiversity(k=k, distance_calculator=calc).calc(reco)
                #######################################################################
        self.total += B

    def reduce(self) -> dict:
        print(f"\nNumber of semantic IDs mapped to UNKNOWN category: {self.unknown_category_count}")
        print(f"Total category lookups: {self.total_category_lookups}")

        if self.total_category_lookups > 0:
            unknown_percentage = 100 * self.unknown_category_count / self.total_category_lookups
            print(f"Percentage of UNKNOWNs: {unknown_percentage:.2f}%")
        else:
            print("No category lookups performed.")

        print("\nTop 10 most frequent predicted semantic IDs:")
        for semantic_id, count in sorted(self.zero_semantic_count.items(), key=lambda x: -x[1])[:10]:
            print(f"{semantic_id}: {count}")
        return {k: v / self.total for k, v in self.metrics.items()}
