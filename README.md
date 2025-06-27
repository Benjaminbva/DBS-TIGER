# 📘 Project Title


## 🧑‍💻 Team Members
- Benjamin van Altena – benjamin.van.altena@student.uva.nl  
- Elyanne Oey – elyanne.oey@student.uva.nl  
- Sharanda Suttorp – sharanda.suttorp@student.uva.nl 
- Lisanne Wallaard - lisanne.wallaard@student.uva.nl

## 👥 Supervising TAs
- Yubao Tang (Main Supervisor)
- Owen de Jong (Co-supervisor)


---

## 🧾 Project Abstract
Recommender systems are employed for filtering and prioritizing content, with recent advances shifting towards generative approaches for sequential recommendation. Rajput et al. \cite{rajput2023recommendersystemsgenerativeretrieval} present TIGER, a generative transformer-based model designed for sequential recommendation, claiming it outperforms other SOTA models. It generates semantically meaningful item IDs, and then predicts next-item Semantic IDs with a sequence-to-sequence model. While it includes a tunable parameter to promote diversity, post-processing methods for encouraging diversity remain limited. This study investigates the reproducibility of their research by attempting to replicate the results and comparing with two other baseline methods. The main contribution of this research focuses on improving diversity output by implementing Diverse Beam Search (DBS), implementing the Intra-List Diversity metric to asses diversity. Additionally, we examined the generalizability of TIGER by testing its performance on the Yelp dataset. Our results indicate that we were able to partially validate the original findings. TIGER demonstrates generalizability to a different dataset. Furthermore, the application of DBS enhances output diversity. 

---

## 📊 Summary of Results


### Reproducability 

- Exact replication of TIGER results was not possible due to limiations of the codebase.
- The replicated values of the baselines (SASRec and S$^3$-Rec) approximate reported ones.
- Across datasets, the trends partially align, where "Sports and Outdoors" is perceived as the most difficult.
- S$^3$-Rec outperforms the SASRec, while TIGER considerably underperforms both.

### Extensions

- New Dataset Extension (Yelp): TIGER generalizes to Yelp similarly compared to other difficult datasets, such as "Sports and Outdoors"
- Diversity Evaluation (ILD@k): ILD@$k$ diversity metric is integreated into our evaluation pipeline using the Rectools library and its results showed that TIGER already contains a moderate level of diversity across the datasets.
- Methodology Extension (Diverse Beam Search): Compared to standard beam search with both normal temperature and increased temperature, we observe that Diverse Beams Search generally leads to the highest diversity scores in terms of ILD@$k$ without substantially sacrificing its relevance score measured by NDCG@$k$.

---

## 🛠️ Task Definition

TIGER focuses on sequential recommendation and introduces a new paradigm: generative retrieval for recommendation. Instead of relying on matching item embeddings to user embeddings, TIGER directly generates the next semantic ID token-by-token based on the user's interaction history. The input consists of a user ID and the sequence of semantic IDs of previously interacted items. The output is the predicted semantic ID of the next item.

---

## 📂 Datasets

- [x] [Amazon Reviews Datasets (Beauty)](https://github.com/jeykigung/P5)
  - [x] Pre-processing: removed users with fewer than 5 interactions. Applied leave-one-out protocol for evaluation. The final item in the sequence serves as the test instance, the second-to-last item is reserved for validation, and the preceding items are used for training. The number of items in a sequence is limited to 20 during training.
  - [x] Subsets considered: The full datasets were used without partitioning into specific subsets.
  - [x] Dataset size: # users: 22.363, # items: 12.101, sparsity: 0.0734%, sequence length (mean): 8.87, sequence length (median) 6.
  - [x] Attributes for user, item and/or group fairness: No fairness attributes were used; the study focused on diversity using Diverse Beam Search and ILD metric.

- [x] [Amazon Reviews Datasets (Sports and Outdoors)](https://github.com/jeykigung/P5)
  - [x] Pre-processing: removed users with fewer than 5 interactions. Applied leave-one-out protocol for evaluation. The final item in the sequence serves as the test instance, the second-to-last item is reserved for validation, and the preceding items are used for training. The number of items in a sequence is limited to 20 during training.
  - [x] Subsets considered: The full datasets were used without partitioning into specific subsets.
  - [x] Dataset size: # users: 35.598, # items: 18.357, sparsity: 0.0453%, sequence length (mean): 8.32, sequence length (median) 6.
  - [x] Attributes for user, item and/or group fairness: No fairness attributes were used; the study focused on diversity using Diverse Beam Search and ILD metric.

- [x] [Amazon Reviews Datasets (Toys and Games)](https://github.com/jeykigung/P5)
  - [x] Pre-processing: removed users with fewer than 5 interactions. Applied leave-one-out protocol for evaluation. The final item in the sequence serves as the test instance, the second-to-last item is reserved for validation, and the preceding items are used for training. The number of items in a sequence is limited to 20 during training.
  - [x] Subsets considered: The full datasets were used without partitioning into specific subsets.
  - [x] Dataset size: # users: 19.412, # items: 11.924, sparsity: 0.0724%, sequence length (mean): 8.63, sequence length (median) 6.
  - [x] Attributes for user, item and/or group fairness: No fairness attributes were used; the study focused on diversity using Diverse Beam Search and ILD metric.

- [x] [Amazon Reviews Datasets (Yelp)](https://github.com/jeykigung/P5)
  - [x] Pre-processing: removed users with fewer than 5 interactions. Applied leave-one-out protocol for evaluation. The final item in the sequence serves as the test instance, the second-to-last item is reserved for validation, and the preceding items are used for training. The number of items in a sequence is limited to 20 during training.
  - [x] Subsets considered: The full datasets were used without partitioning into specific subsets.
  - [x] Dataset size: # users: 30.431, # items: 20.033, sparsity: 0.0519%, sequence length (mean): 11.40, sequence length (median) 8.
  - [x] Attributes for user, item and/or group fairness: No fairness attributes were used; the study focused on diversity using Diverse Beam Search and ILD metric.

---

## 📏 Metrics

- [x] Recall@$k$
  - [x] Description: calculates how many of the relevant items are in the top k. Hits are calculated in the codebase, which is is equivalent to computing Recall since there is only one actual next item.
- [x] NDCG@$k$
  - [x] Description: measures how well the ranked items align with the ideal ranking. 
- [x] ILD@$k$
  - [x] Description: measures the mean pairwise distance among the top-$k$ items recommended to the user. It is implemnted using the `IntraListDivesrity` class from the 
the [Rectools library](https://rectools.readthedocs.io/en/latest/api/rectools.metrics.diversity.IntraListDiversity.html). As the distance calculator, we use the _Pairwise Hamming Distance_, which leadings in our implementation to a distance of at least 1 and at most 4. 

---

## 🔬 Baselines & Methods

- [x] [SASRec](https://ieeexplore.ieee.org/abstract/document/8594844?casa_token=lOdnxe7VGB0AAAAA:r7vyi2i1y-wxW9hI9SHxkkPX7ztWs6sw1yiO2fOkYxzdRPPZRrXoNtt_Kz4htA5R2aJqknAaGg) (Self-Attentive Sequential Recommendation) employs a Transformer with a causal mask to capture the sequential behavior of users. Causal masking restricts the model's attention to the current and previous positions, which is crucial for sequential tasks. Built upon self-attention, SASRec uses multi-head attention to model long-term relationships based on limited interactions, enabling it to generate the next recommended item in a sequence.
- [x] [S$^3$-Rec](https://arxiv.org/abs/2008.07873) (Self-Supervised Learning for Sequential Recommendation) is a self-attentive model that enhances the sequential recommendation performance by pre-training a bi-directional Transformer using self-supervised learning. To achieve this, it exploits the intrinsic data correlations based on the Mutual Information Maximization principle (MIM), which offers a framework for capturing correlations among diverse data representations. 

### 🧠 High-Level Description of Method

TIGER consists of 2 stages:

- Stage 1 (semantic ID generation): Item metadata → Sentence-T5 encoder → Dense item embeddings → RQ-VAE with 3 codebooks → Discrete semantic ID
- Stage 2 (generative retrieval): User + item history sequence → Transformer encoder-decoder → Token-by-token semantic ID prediction → Next item prediction

---

## 🌱 Proposed Extensions

- New Dataset Extension: Validating TIGER’s generalization on a dataset outside the Amazon domain (Yelp).
- Diversity Evaluation: Integrating a new diversity metric, Intra-List Diversity (ILD@$k$).
- Methodology Extension: Implementing Diverse Beam Search in the decoding phase and comparing it against standard beam search with normal temperature and increased temperature. 

## References

1. Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al.  
   **Accurate structure prediction of biomolecular interactions with AlphaFold 3.**  
   *Nature, 630(8016):493–500, 2024.*

---

