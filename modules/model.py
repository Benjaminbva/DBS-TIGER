import gin
import torch

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from typing import Dict, Tuple, Union, List # NEW BEN


# Needed to make torch.compile succeed
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim,
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)

        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(
            num_embeddings=sem_id_dim, embedding_dim=embedding_dim
        )

        self.transformer = (
            TransformerEncoderDecoder(
                d_in=attn_dim,
                d_out=attn_dim,
                dropout=dropout,
                num_heads=num_heads,
                encoder_layers=n_layers // 2,
                decoder_layers=n_layers // 2,
            )
            if self.jagged_mode
            else nn.Transformer(
                d_model=attn_dim,
                nhead=num_heads,
                num_encoder_layers=n_layers // 2,
                num_decoder_layers=n_layers // 2,
                dim_feedforward=1024,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)

    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)

        B, N, D = sem_ids_emb.shape

        pos_max = N // self.sem_id_dim
        # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)

        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)

        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)
            input_embedding_fut = torch.cat(
                [input_embedding_fut, sem_ids_emb_fut + tte_fut], axis=1
            )

        if self.jagged_mode:
            input_embedding = padded_to_jagged_tensor(
                input_embedding,
                lengths=seq_lengths + 1,
                max_len=input_embedding.shape[1],
            )

            seq_lengths_fut = torch.tensor(
                input_embedding_fut.shape[1],
                device=input_embedding_fut.device,
                dtype=torch.int64,
            ).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(
                input_embedding_fut,
                lengths=seq_lengths_fut,
                max_len=input_embedding_fut.shape[1],
            )
        else:
            mem_mask = torch.cat(
                [
                    torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
                    batch.seq_mask,
                ],
                axis=1,
            )
            f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
            f_mask[~mem_mask] = float("-inf")

        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))

        if self.jagged_mode:
            transformer_output = self.transformer(
                x=transformer_input,
                context=transformer_context,
                padding_mask=batch.seq_mask,
                jagged=self.jagged_mode,
            )
        else:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                transformer_input.shape[1]
            )
            transformer_output = self.transformer(
                src=transformer_context,
                tgt=transformer_input,
                tgt_is_causal=True,
                tgt_mask=causal_mask,
                src_key_padding_mask=f_mask,
                memory_key_padding_mask=f_mask,
            )

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id(
        self, batch: TokenizedSeqBatch, temperature: int = 1, top_k: bool = True
    ) -> GenerationOutput:

        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 32 if top_k else 1
        n_top_k_candidates = 200 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None,
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            samples_batched = torch.multinomial(
                probas_batched, num_samples=n_top_k_candidates
            )

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(
                    samples_batched.unsqueeze(-1)
                )
            else:
                prefix = torch.cat(
                    [
                        generated.flatten(0, 1)
                        .unsqueeze(1)
                        .repeat_interleave(n_top_k_candidates, axis=1),
                        samples_batched.unsqueeze(-1),
                    ],
                    axis=-1,
                )
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)

            sampled_log_probas = torch.log(
                torch.gather(probas_batched, 1, samples_batched)
            ).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            # Get top-K:
            sorted_log_probas, sorted_indices = (
                -10000 * (~is_valid_prefix)
                + sampled_log_probas
                + maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = (
                sorted_log_probas[:, :k],
                sorted_indices[:, :k],
            )
            top_k_samples = torch.gather(samples, 1, top_k_indices)

            if generated is not None:
                parent_id = torch.gather(
                    generated,
                    1,
                    (top_k_indices // n_top_k_candidates)
                    .unsqueeze(2)
                    .expand(-1, -1, i),
                )
                top_k_samples = torch.cat(
                    [parent_id, top_k_samples.unsqueeze(-1)], axis=-1
                )

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(
                        next_sem_ids.shape[1], device=next_sem_ids.device
                    ).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids,
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
                print("final sem ids:", generated[0])
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)

                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    cache = torch.zeros(
                        input_batch.sem_ids.shape[0],
                        input_batch.sem_ids.shape[1] + 1,
                        self.attn_dim,
                        device=input_batch.sem_ids.device,
                    )
                    cache_mask = torch.cat(
                        [
                            torch.ones(
                                input_batch.sem_ids.shape[0],
                                1,
                                dtype=bool,
                                device=input_batch.seq_mask.device,
                            ),
                            input_batch.seq_mask,
                        ],
                        axis=1,
                    )
                    cache[cache_mask] = self.transformer.cached_enc_output.values()
                    lengths = (
                        self.transformer.cached_enc_output.offsets()
                        .diff()
                        .repeat_interleave(k)
                    )
                    cache = cache.repeat_interleave(k, dim=0)
                    self.transformer.cached_enc_output = padded_to_jagged_tensor(
                        cache, lengths, max_len=cache.shape[1]
                    )

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(
                        k, dim=0
                    ),
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())

                print("[DEBUG] DBS penalty applied", flush=True)

        return GenerationOutput(
            sem_ids=generated.squeeze(), log_probas=log_probas.squeeze()
        )

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id_dbs(
        self, batch: TokenizedSeqBatch, 
        temperature: int = 1, top_k: bool = True,
        num_groups: int = 1,
        diversity_lambda: Union[float, List[float]] = 0.8,
        diversity_func: str = "hamming",
        ngram_size: int = 2 # bigram
    ) -> GenerationOutput:

        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        k = 32 if top_k else 1
        n_top_k_candidates = 200 if top_k else 1

        # ── Step 1: split beam into G groups of size B_per_group ───────────
        G = num_groups
        assert k % G == 0, "k must be divisible by num_groups"
        B_per_group = k // G

        # build per-group λ list: lambda for first group is 0, for the rest it is diversity_lambda
        if isinstance(diversity_lambda, (int, float)):
            lambdas = [0.0] + [float(diversity_lambda)] * (G - 1)
        else:
            assert len(diversity_lambda) == G, "need one λ per group"
            lambdas = [0.0] + [float(x) for x in diversity_lambda[1:]]

        # ── Step 2: initialize one input & one beam-state per group ────────
        base_input       = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None,
        )
        input_batches = [base_input] * G
        generated_groups = [None] * G
        logprobs_groups = [None] * G

        # ── Step 3: for each position i, expand each sub-beam in turn ─────
        for i in range(self.sem_id_dim):
            print("i:", i)
            #print("sem id dim:", self.sem_id_dim)
            new_inputs = []
            new_gens = []
            new_logprobs  = []

            for g in range(G):
                inp = input_batches[g]
                λg = lambdas[g]

                # — forward & sampling —
                logits = self.forward(inp).logits
                probas = F.softmax(logits / temperature, dim=-1)
                samples = torch.multinomial(probas, num_samples=n_top_k_candidates)
                B_, C_ = samples.shape

                if generated_groups[g] is None:
                    valid = self.inference_verifier_fn(samples.unsqueeze(-1))
                else:
                    pref = torch.cat([
                        generated_groups[g]
                            .flatten(0,1)
                            .unsqueeze(1)
                            .repeat_interleave(n_top_k_candidates, axis=1),
                        samples.unsqueeze(-1)
                    ], axis=-1)
                    valid = self.inference_verifier_fn(pref).reshape(B, -1)

                sampled_lp = torch.log(torch.gather(probas, 1, samples)).reshape(B, -1)
                flat_samps = samples.reshape(B, -1)

                # — pick previous logprobs or zero if none —
                prev_logprobs = (
                    logprobs_groups[g]
                    if logprobs_groups[g] is not None
                    else 0
                )

                raw_scores = (
                    -1e4 * (~valid)
                    + sampled_lp
                    + maybe_repeat_interleave(
                        prev_logprobs,
                        n_top_k_candidates,
                        dim=1,
                    )
                )


                if diversity_func == "hamming" and i != 3:
                    # Hamming‐diversity penalty for g > 0
                    if g > 0 and λg != 0.0:
                        counts = torch.zeros_like(raw_scores)
                        for h in range(g):
                            prev_last = new_gens[h][:,:, -1]    # (B, B_per_group)
                            eq = flat_samps.unsqueeze(2) == prev_last.unsqueeze(1)
                            counts = counts + eq.sum(dim=2).to(raw_scores.dtype)
                            #print("counts", counts)
                        raw_scores = raw_scores - λg * counts

                if diversity_func == "ngram":
                    # n-gram diversity penalty for groups > 0
                    if g > 0 and λg != 0.0:
                        # collect n-1 tokens suffixes from earlier groups
                        counts = torch.zeros_like(raw_scores)
                        for h in range(g):
                            prev_seqs = generated_groups[h]  # shape: (B, B_per_group_h, t) or (B, t) if first
                            # reshape to (B*beams, t)
                            flat_prev = prev_seqs.reshape(-1, prev_seqs.shape[-1])
                            # for each candidate, compare last n-1 tokens
                            if flat_prev.shape[-1] + 1 >= ngram_size:
                                suffix = flat_prev[:, - (ngram_size - 1):]
                                # expand suffix for all candidates
                                suffix_exp = suffix.view(B, -1, 1, ngram_size - 1)
                                flat_cand = samples.reshape(B, -1, 1)
                                # build ngram tensors: (B, prev_beams, V, n)
                                cand_exp = flat_cand.unsqueeze(1).expand(-1, suffix_exp.shape[1], -1, -1)
                                suffix_tile = suffix_exp.expand(-1, -1, cand_exp.shape[2], -1)
                                ngrams = torch.cat([suffix_tile, cand_exp], dim=-1)
                                # compare against previous group ngrams
                                prev_ng = ngrams.reshape(B, -1, ngram_size)
                                # for each candidate, count matches across prev sequences
                                eq = (prev_ng.unsqueeze(2) == ngrams.view(B, -1, 1, ngram_size)).all(dim=-1)
                                # sum matches per candidate token
                                counts = counts + eq.sum(dim=1)
                        raw_scores = raw_scores - λg * counts

                # — prune to B_per_group —
                sorted_scores, sorted_inds = raw_scores.sort(-1, descending=True)
                topk_scores = sorted_scores[:, :B_per_group]
                topk_inds = sorted_inds[:, :B_per_group]
                topk_samps = torch.gather(flat_samps, 1, topk_inds)

                # — rebuild `generated` & next‐batch for this group —
                if generated_groups[g] is not None:
                    parent_id = torch.gather(
                        generated_groups[g],
                        1,
                        (topk_inds // n_top_k_candidates)
                            .unsqueeze(2)
                            .expand(-1, -1, i),
                    )
                    topk_samps = torch.cat([parent_id, topk_samps.unsqueeze(-1)], axis=-1)
                    next_ids = topk_samps.flatten(end_dim=1)

                    new_inp = TokenizedSeqBatch(
                        user_ids=inp.user_ids,
                        sem_ids=inp.sem_ids,
                        sem_ids_fut=next_ids,
                        token_type_ids_fut=torch.arange(
                            next_ids.shape[1], device=next_ids.device
                        ).repeat(next_ids.shape[0], 1),
                        seq_mask=inp.seq_mask,
                        token_type_ids=inp.token_type_ids,
                    )
                    new_gen = topk_samps.clone().detach()
                    new_logprob = topk_scores.clone().detach()

                else:
                    next_ids = topk_samps.reshape(-1, 1)

                    if self.jagged_mode:
                        # Calculate sequence lengths for the current batch
                        seq_lengths = inp.seq_mask.sum(axis=1)
                        
                        # Create a new sequence mask that includes the extra token
                        new_seq_mask = torch.cat([
                            torch.ones(inp.sem_ids.shape[0], 1, dtype=bool, device=inp.seq_mask.device),
                            inp.seq_mask
                        ], axis=1)
                        
                        # Calculate sequence lengths including the extra token
                        new_seq_lengths = new_seq_mask.sum(axis=1)
                        
                        # If we have cached output, we need to ensure dimensions match
                        if self.transformer.cached_enc_output is not None:
                            # Get the cached sequence lengths
                            cached_lengths = self.transformer.cached_enc_output.offsets().diff()
                            
                            # If lengths don't match, we need to recompute
                            if not torch.all(new_seq_lengths == cached_lengths):
                                self.transformer.cached_enc_output = None
                                # Run a forward pass to initialize the cache with correct dimensions
                                _ = self.forward(inp)
                        
                        # Create cache with consistent dimensions
                        cache = torch.zeros(
                            inp.sem_ids.shape[0],
                            inp.sem_ids.shape[1] + 1,
                            self.attn_dim,
                            device=inp.sem_ids.device,
                        )
                        
                        if self.transformer.cached_enc_output is not None:
                            cache[new_seq_mask] = self.transformer.cached_enc_output.values()
                            lengths = (
                                self.transformer.cached_enc_output.offsets()
                                    .diff()
                                    .repeat_interleave(B_per_group)
                            )
                        else:
                            # If no cached output, use new sequence lengths
                            lengths = new_seq_lengths.repeat_interleave(B_per_group)
                        
                        # Ensure cache dimensions match the input
                        cache = cache.repeat_interleave(B_per_group, dim=0)
                        self.transformer.cached_enc_output = padded_to_jagged_tensor(
                            cache, lengths, max_len=cache.shape[1]
                        )
                        
                        # Update the input batch's sequence mask to match the cache
                        new_seq_mask = torch.cat([
                            torch.ones(inp.sem_ids.shape[0], 1, dtype=bool, device=inp.seq_mask.device),
                            inp.seq_mask
                        ], axis=1)
                        new_seq_mask = new_seq_mask.repeat_interleave(B_per_group, dim=0)

                    new_inp = TokenizedSeqBatch(
                        user_ids=inp.user_ids.repeat_interleave(B_per_group, dim=0),
                        sem_ids=inp.sem_ids.repeat_interleave(B_per_group, dim=0),
                        sem_ids_fut=next_ids,
                        token_type_ids_fut=torch.zeros_like(next_ids),
                        seq_mask=inp.seq_mask.repeat_interleave(B_per_group, dim=0),
                        token_type_ids=inp.token_type_ids.repeat_interleave(B_per_group, dim=0),
                    )
                    new_gen = topk_samps.unsqueeze(-1)
                    new_logprob = topk_scores.clone().detach()

                new_inputs.append(new_inp)
                new_gens.append(new_gen)
                new_logprobs.append(new_logprob)

            # end for g
            input_batches = new_inputs
            generated_groups = new_gens
            logprobs_groups = new_logprobs

        # ── Step 4: stitch all groups back together ─────────────────────────
        final_sem_ids = torch.cat(generated_groups, dim=1)
        final_log_probas = torch.cat(logprobs_groups,  dim=1)

        print("final sem ids:", final_sem_ids[0])

        # ── Step 5: global sort across the full beam ────────────────────────
        #   so that we truly pick the top-K by (possibly penalized) score
        sorted_scores, sorted_inds = final_log_probas.sort(dim=1, descending=True)
        # we need to expand sorted_inds to index the seq_len dimension too
        expand_idx = sorted_inds.unsqueeze(-1).expand(-1, -1, final_sem_ids.size(-1))
        final_sem_ids    = final_sem_ids.gather(dim=1, index=expand_idx)
        final_log_probas = sorted_scores

        print("final sem ids na sort:", final_sem_ids[0])

        return GenerationOutput(
            sem_ids=final_sem_ids,
            log_probas=final_log_probas
        )

    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)

        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(
                    jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B
                )[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(
                    F.cross_entropy(logits, target, reduction="none", ignore_index=-1),
                    "(b n) -> b n",
                    b=B,
                )
                loss = unred_loss.sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = (
                    rearrange(
                        F.cross_entropy(out, target, reduction="none", ignore_index=-1),
                        "(b n) -> b n",
                        b=B,
                    )
                    # .sum(axis=1)
                    # .mean()
                )
                loss = unred_loss.sum(axis=1).mean()
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
            loss_d = unred_loss.mean(axis=0)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(
                jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B
            )[:, -1, :]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:, -1, :]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)
