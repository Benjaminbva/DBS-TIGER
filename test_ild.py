
import pandas as pd
import torch
import random
from rectools.metrics.diversity import IntraListDiversity
from rectools.metrics.distances import PairwiseHammingDistanceCalculator

k = 5 # Number of top-k items to consider
b = 0 # User ID 

# Create recommendation DataFrame
user_ids = [b]*k
item_ids = list(range(k))
ranking = list(range(1, k + 1))
reco = pd.DataFrame({
    "user_id": user_ids,
    "rank": ranking,
    "item_id": item_ids, 
})  

# Example top-k predictions with high and low diversity
unique_values = random.sample(range(1, 256), 20)  
topk_pred_high_div = torch.tensor(unique_values).reshape(5, 4)
topk_pred_low_div = torch.tensor([
    [200, 200,  50,  50],
    [200, 200,  50,  51],
    [200, 200,  51,  50],
    [200, 201,  50,  50],
    [201, 200,  50,  50],
])

# Calculate Intra-List Diversity (ILD) for high diversity example
features_df_high = pd.DataFrame(topk_pred_high_div.numpy(), columns=["feat_1", "feat_2", "feat_3", "feat_4"])
calc_high = PairwiseHammingDistanceCalculator(features_df_high)
ild_high = IntraListDiversity(k=k, distance_calculator=calc_high).calc(reco)
print(f"High Diversity ILD: {ild_high:.4f}")

# Calculate Intra-List Diversity (ILD) for low diversity example
features_df_low = pd.DataFrame(topk_pred_low_div.numpy(), columns=["feat_1", "feat_2", "feat_3", "feat_4"])
calc_low = PairwiseHammingDistanceCalculator(features_df_low)
ild_low = IntraListDiversity(k=k, distance_calculator=calc_low).calc(reco)
print(f"Low Diversity ILD: {ild_low:.4f}")
