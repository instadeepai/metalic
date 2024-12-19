# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import itertools
import logging
import os
import random
import time
from copy import deepcopy
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union

import esm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import transformers
from omegaconf import DictConfig
from scipy.stats import spearmanr
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from meta.npt.proteinnpt.utils.esm.modules import (  # ESM1bLayerNorm,
    AxialTransformerLayer,
)
from meta.dataclasses import Candidate, OraclePoints
from meta.logger import Logger
from meta.models.base_metasurrogate import BaseMetaSurrogate
from meta.models.losses import get_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_TPU = False
TRY_TPU = False
if TRY_TPU:
    # Try to use TPU:
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        USE_TPU = True
    except ImportError as error:
        print(error)
        print("Assuming TPU is not available")

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


def dummy_padding(op: OraclePoints, pad_len: int = 400) -> OraclePoints:
    """
    Pad or truncate sequences in OraclePoints to the same length.
    """
    new_candidate_points = []
    for cp in op.candidate_points:
        s_len = len(cp.sequence)
        if s_len > pad_len:
            new_candidate_points.append(Candidate(cp.sequence[:pad_len]))
        elif s_len < pad_len:
            new_candidate_points.append(
                Candidate(
                    cp.sequence
                    + "".join(
                        random.choice(cp.sequence) for _ in range(pad_len - s_len)
                    )
                )
            )
        else:
            new_candidate_points.append(cp)
        assert (
            len(new_candidate_points[-1]) == pad_len
        ), f"Expected length {pad_len}, found {len(cp.sequence)}"
    return OraclePoints(new_candidate_points, op.oracle_values)


class ESMEmbedding(nn.Module):
    def __init__(
        self,
        model_name: str,
        layer: Optional[int],
        freeze_embed: bool,
        embed_on_cpu: bool,
    ) -> None:
        super().__init__()
        # Load pretrained embedding model
        self.model: transformers.PreTrainedModel = getattr(
            esm.pretrained, model_name
        )()[0]
        self.model.to(device).eval()
        self.model_name = model_name
        self.freeze_embed = freeze_embed
        self.embed_on_cpu = embed_on_cpu

        max_layer = int(self.model_name.split("_")[1].split("t")[1])

        if layer is None:
            # Extract number of layers from model_name if not explicitly passed
            self.layer = max_layer
        else:
            self.layer = layer
            if self.layer > max_layer:
                raise ValueError(
                    f"Cannot choose a layer value, {self.layer} higher"
                    f" than the number of layers in model, {max_layer}"
                )

        # Using ESM tokenizer until further notice
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            f"facebook/{model_name}"
        )

        # Freeze parameters if necessary
        if self.freeze_embed:
            for param in self.parameters():
                param.requires_grad = False

    def move_to_device(
        self,
    ) -> Tuple[Union[str, torch.device], Union[str, torch.device]]:
        # Move to CPU if necessary
        output_device = embedding_device = device  # type: ignore
        if self.embed_on_cpu:
            embedding_device = "cpu"  # type: ignore
            self.model.to(embedding_device)
        return output_device, embedding_device

    def forward(self, candidates: List[Candidate]) -> torch.Tensor:
        output_device, embedding_device = self.move_to_device()
        tokens = self.tokenizer(
            [c.sequence for c in candidates], return_tensors="pt"
        ).input_ids
        output_dict = self.model(
            tokens.to(embedding_device), repr_layers=[self.layer], return_contacts=False
        )
        output = output_dict["representations"][self.layer]
        output = output.to(output_device)
        return output.detach() if self.freeze_embed else output

    def wt_marginal_predictions(self, wt_seq: str, seqs: List[str]) -> torch.Tensor:
        output_device, embedding_device = self.move_to_device()

        N, L = len(seqs), len(wt_seq)  # noqa: N806
        # Get wild type logits
        wt_tensor = self.tokenizer(
            wt_seq, return_tensors="pt", padding=False
        ).input_ids.to(embedding_device)
        wt_logits = (
            self.model(wt_tensor, return_contacts=False)["logits"].detach().squeeze()
        )
        L, _ = wt_logits.shape  # noqa: N806
        # Get inputs for seqs
        tokens = [
            self.tokenizer(s, return_tensors="pt", padding=False).input_ids
            for s in seqs
        ]
        tensor_tokens = torch.stack(tokens).to(embedding_device).squeeze(1)
        assert tensor_tokens.shape == (
            N,
            L,
        ), f"Expected shape {(N, L)}, found {tensor_tokens.shape}"
        # Evaluate logits at wt
        wt_logits_at_wt = wt_logits.gather(
            1, wt_tensor.squeeze().unsqueeze(-1)
        ).squeeze()
        # Expanding wt_logits to match the batch size of tensor_tokens
        wt_logits_expanded = wt_logits.unsqueeze(0).expand(
            N, -1, -1
        )  # Shape: [N, L, alphabet_sz]
        # Evaluate wt logits at seqs
        wt_logits_at_seqs = wt_logits_expanded.gather(
            2, tensor_tokens.unsqueeze(-1)
        ).squeeze(-1)
        assert wt_logits_at_seqs.shape == (
            N,
            L,
        ), f"Expected shape {(N, L)}, found {wt_logits_at_seqs.shape}"
        # Return the sum of the difference between wt logits and wt logits at seqs
        output = torch.sum(wt_logits_at_seqs - wt_logits_at_wt, dim=1).detach()
        return output.to(output_device)

    @property
    def embed_dim(self) -> int:
        embed_dim = 0
        try:
            embed_dim = self.model.embed_dim  # for ESM2
        except AttributeError:
            embed_dim = self.model.embed_tokens.embedding_dim  # for ESM1
        return embed_dim


class FitnessPredictionModel(nn.Module):
    def __init__(
        self,
        input_sz: int,
        f_mlp_layer_sizes: List[int],
        latent_index: Optional[int],
        dropout_prob: float,
    ) -> None:
        """Create a fitness prediction model (head) with an optional stochastic latent layer."""
        super().__init__()
        self.input_sz = input_sz
        self.f_mlp_layer_sizes = f_mlp_layer_sizes
        self.latent_index = latent_index
        self.dropout_prob = dropout_prob

        self.fc_mean, self.fc_logvar = None, None

        # Define parameters for all layers
        self.f_layers = nn.ModuleList()
        f_input_sz = input_sz
        for idx, h_sz in enumerate(self.f_mlp_layer_sizes):
            if self.latent_index is not None and idx == self.latent_index:
                self.fc_mean = nn.Linear(f_input_sz, h_sz)
                self.fc_logvar = nn.Linear(f_input_sz, h_sz)
                # Add placeholder layer that has no parameters
                self.f_layers.append(nn.Identity())
            else:
                # Add linear layer, dropout, and ReLU activation as one layer
                next_layer = nn.Sequential(
                    nn.Linear(f_input_sz, h_sz),
                    nn.Dropout(self.dropout_prob),
                    nn.ReLU(),
                )
                self.f_layers.append(next_layer)
            f_input_sz = h_sz

        # final layer with output size 1
        self.f_layers.append(nn.Linear(f_input_sz, 1))

        # Check that the model was created correctly
        if self.latent_index is None:
            assert self.fc_mean is None, "Mean layer created without latent index"
            assert self.fc_logvar is None, "Logvar layer created without latent index"
        else:
            assert self.fc_mean is not None, "Mean layer not created"
            assert self.fc_logvar is not None, "Logvar layer not created"

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        mean, logvar, z = None, None, None
        for idx, layer in enumerate(self.f_layers):
            if self.latent_index is not None and idx == self.latent_index:
                mean, logvar = self.fc_mean(x), self.fc_logvar(x)  # type: ignore
                z = self.reparameterize(mean, logvar)
                x = z
            else:
                x = layer(x)
        return x, z, mean, logvar


class ProteinNPTMetaSurrogate(BaseMetaSurrogate, nn.Module):
    def __init__(  # noqa: CCR001
        self,
        name: str,
        model_config: DictConfig,
        train_config: DictConfig,
    ) -> None:
        super().__init__(
            name,
            support_size=train_config.support_size,
            query_size=train_config.query_size,
            use_all_data=train_config.use_all_data,
            max_context_sz=train_config.max_context_sz,
            num_outputs=1,
        )
        nn.Module.__init__(self)
        self.train_config = train_config
        self.use_perceiver_layer = model_config.use_perceiver_layer
        self.embed_dim = model_config.embed_dim
        self.conv_kernel_size = model_config.conv_kernel_size
        self.dropout_prob = model_config.dropout_prob
        self.num_protein_npt_layers = model_config.num_protein_npt_layers
        self.f_conditions_on_pooled_seq = model_config.f_conditions_on_pooled_seq
        self.use_cnn_before_npt = model_config.use_cnn_before_npt
        self.use_cnn_after_npt = model_config.use_cnn_after_npt
        self.axial_embed_dim = model_config.axial_embed_dim
        self.attention_heads = model_config.attention_heads
        self.attention_dropout = model_config.attention_dropout
        self.activation_dropout = model_config.activation_dropout
        self.max_tokens_per_context = model_config.max_tokens_per_context
        self.deactivate_col_attention = model_config.deactivate_col_attention
        self.use_tranception_style_row_attention = (
            model_config.use_tranception_style_row_attention
        )
        self.f_mlp_layer_sizes = model_config.f_mlp_layer_sizes
        self.latent_index = model_config.latent_index
        self.context_normalization = model_config.context_normalization
        self.condition_on_wild_type = model_config.condition_on_wild_type
        self.aux_pred = model_config.aux_pred
        self.aux_pred_model = model_config.aux_pred_model
        self.make_wt_aux_pred_zero = model_config.make_wt_aux_pred_zero

        self.using_synthetic_data = (
            train_config.latent_distance_data_weight is not None
            or train_config.zero_shot_data_weight is not None
            or train_config.nkde_data_weight is not None
        )

        if train_config.reptile:
            assert (
                train_config.support_size != 0
            ), "support set must be given for fine-tuning if using Reptile"

        assert (
            self.embed_dim % self.attention_heads == 0
        ), "Embedding dimension must be divisible by the number of attention heads"

        # Embedding model
        self.embedding_model = ESMEmbedding(  # type: ignore
            train_config.ESM_embed_model,
            train_config.ESM_embed_layer,
            train_config.freeze_embed,
            train_config.embed_on_cpu,
        )

        # Define layers
        self.perceiver_layers = (
            None if self.use_perceiver_layer else None
        )  # TODO: Perceiver layers
        # Project embedding to correct size if necessary
        self.token_embedding_projection: Optional[nn.Linear] = None
        if self.embedding_model.embed_dim != self.embed_dim:
            self.token_embedding_projection = nn.Linear(
                self.embedding_model.embed_dim, self.embed_dim
            )
        # Optional CNN layer. Used before and/or after NPT layers
        self.cnn_layer: Optional[nn.Module] = None
        if self.use_cnn_before_npt or self.use_cnn_after_npt:
            self.cnn_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=self.embed_dim,
                    kernel_size=self.conv_kernel_size,
                    padding="same",
                ),
                nn.Dropout(self.dropout_prob),
                nn.ReLU(),
            )
        # Fitness embeddings layer
        self.f_embed_layer = nn.Linear(
            2,  # Need to add one as we append the mask flag to each input target
            self.embed_dim,
        )
        # Fitness prediction layer(s)
        f_input_sz = (
            2 * self.embed_dim if self.f_conditions_on_pooled_seq else self.embed_dim
        )
        self.f_mlp = FitnessPredictionModel(
            f_input_sz, self.f_mlp_layer_sizes, self.latent_index, self.dropout_prob
        )
        # Auxiliary prediction embeddings layer
        self.aux_embed_layer = nn.Linear(1, self.embed_dim)
        # NPT layers
        self.npt_layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    self.embed_dim,
                    self.axial_embed_dim,
                    self.attention_heads,
                    self.dropout_prob,
                    self.attention_dropout,
                    self.activation_dropout,
                    self.max_tokens_per_context,
                    self.deactivate_col_attention,
                    self.use_tranception_style_row_attention,
                    num_targets=1,
                )
                for _ in range(self.num_protein_npt_layers)
            ]
        )
        # Dropout and norm layers
        self.dropout_module = nn.Dropout(self.dropout_prob)
        # Note these layer norms were originally ESM1bLayerNorm layers
        # But, I believe they were being imported as untrained LayerNorm anyway?
        self.emb_layer_norm_before = nn.LayerNorm(self.embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(self.embed_dim)

        # Create optimizer
        # Note, fordbidden layer types in NPT is forbidden_layer_types=[nn.LayerNorm],
        # but I think they have untrained LayerNorm layers so I don't know how
        # that could've made sense?
        # param_names_other_than_norm = model_utils.get_parameter_names(
        #     self, forbidden_layer_types=[nn.LayerNorm]
        # )
        # bias_param_names = [
        #     name for name in param_names_other_than_norm if "bias" in name
        # ]
        # bias_params = [p for n, p in self.named_parameters() if n in bias_param_names]
        # non_bias_param_names = [
        #     name for name in param_names_other_than_norm if "bias" not in name
        # ]
        # non_bias_params = [
        #     p for n, p in self.named_parameters() if n in non_bias_param_names
        # ]
        bias_params = tuple(
            param for name, param in self.named_parameters() if "bias" in name
        )
        non_bias_params = tuple(
            param for name, param in self.named_parameters() if "bias" not in name
        )
        self.optimizer = AdamW(
            [
                {
                    "params": bias_params,
                    "weight_decay": 0.0,
                },
                {
                    "params": non_bias_params,
                    "weight_decay": float(self.train_config.weight_decay),
                },
            ],
            lr=float(self.train_config.learning_rate),
            eps=float(self.train_config.adam_eps),
            betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
            foreach=False,  # Fixed similar issue: https://github.com/pytorch/pytorch/issues/106121
        )
        # Make a copy of initial optimizer parameters
        self.initial_optimizer_params = deepcopy(self.optimizer.state_dict())
        # Define the cosine learning rate scheduler with warmup
        assert (
            type(self.train_config.use_lr_scheduler) is bool
        ), f"Expected bool, got {type(self.train_config.use_lr_scheduler)}"
        if self.train_config.use_lr_scheduler:

            def lr_lambda(current_step: int) -> float:
                total_steps = self.train_config.num_sched_steps
                warmup_steps = self.train_config.warmup_frac * total_steps
                if current_step < warmup_steps:  # Increase LR
                    return float(current_step) / float(max(1, warmup_steps))
                else:  # Decrease LR
                    # Progress can range from 0.0 to 1.0 after warmup
                    progress = float(current_step - warmup_steps) / float(
                        max(1, total_steps - warmup_steps)
                    )
                    return max(
                        self.train_config.min_learning_rate_fraction,
                        0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))),
                    )

            self.lr_frac = lr_lambda

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
            self.initial_scheduler_params = deepcopy(self.scheduler.state_dict())

        # Move to GPU
        self.to(device)

        # Set to eval
        self.eval()

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Model initialized with the following parameters: {num_parameters}")

    def perceiver(self, tensor_tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        # Use self.embedding_model (in parallel over list?)
        # and then self.perceiver_layers defined above
        return tensor_tokens  # TODO

    def embed(self, candidates: List[Candidate]) -> torch.Tensor:
        # Check if all sequences are of the same length if not embedding to a constant length
        if not self.use_perceiver_layer:
            assert all(
                len(c.sequence) == len(candidates[0].sequence) for c in candidates
            ), "All sequences must be of the same length"

        if self.use_perceiver_layer:
            embed_seqs = self.perceiver(
                [self.embedding_model([c])[0] for c in candidates]
            )
            output_embeddings = torch.stack(embed_seqs)
        else:
            output_embeddings = self.embedding_model(candidates)

        return output_embeddings

    def cleanup(self, delete_checkpoints: bool = True) -> None:
        gc.collect()

    def save_model(self, step: int) -> None:
        checkpoint = {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "scheduler": self.scheduler.state_dict(),
        }
        assert self.save_dir is not None, "Save directory must be set"
        outpath = os.path.join(self.save_dir, f"checkpoint_{step}.pt")  # type: ignore
        # make sure the directory exists
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        torch.save(checkpoint, outpath)

    def save_col_attention(
        self,
        step: int,
        attention_list: List[Optional[torch.Tensor]],
        avg_batch: bool = False,
    ) -> None:
        attention_list = [
            a for a in attention_list if a is not None
        ]  # Remove None values
        if len(attention_list) == 0:  # No attention to save.
            return
        if not avg_batch:
            attention_list = attention_list[0:1]  # Only save the first batch
        assert self.save_dir is not None, "Save directory must be set"
        # Average weights over the batch, layers, and heads, and rows
        reduced_attention_list: List[torch.Tensor] = []
        max_context_sz = max(a.shape[-1] for a in attention_list)  # type: ignore
        for attention in attention_list:
            assert isinstance(attention, torch.Tensor)
            attention_sz = attention.shape[-1]
            if attention_sz < max_context_sz:
                # Skip if this task if there is not enough data
                continue
            attention = attention.squeeze(3)
            # Average over length of protein sequence
            attention = torch.mean(attention, dim=2)
            expected_shape = (
                self.num_protein_npt_layers,
                self.attention_heads,
                max_context_sz,
                max_context_sz,
            )
            assert (
                attention.shape == expected_shape
            ), f"Expected shape {expected_shape}, found {attention.shape}"
            reduced_attention_list.append(attention)
        # Skip if no tasks had enough queries
        if len(reduced_attention_list) == 0:
            return
        # Stack
        new_attention_tensor = torch.stack(reduced_attention_list)
        new_attention = new_attention_tensor.detach().cpu().numpy()
        # Compute Averages
        # Average over batch, layers, heads
        # The dimensions remaining represent attention over the difference sequences
        assert isinstance(new_attention, np.ndarray)
        new_attention = np.mean(
            new_attention,
            axis=(
                0,
                1,
                2,
            ),
        )
        new_attention = new_attention.squeeze()
        assert new_attention.shape == (
            max_context_sz,
            max_context_sz,
        ), f"Expected shape {(max_context_sz,max_context_sz)}, found {new_attention.shape}"
        # Save
        outpath = os.path.join(self.save_dir, f"col_attention_{step}.pt")
        # make sure the directory exists
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        torch.save(new_attention, outpath)
        # Save Heatmap
        heatmap_path = os.path.join(self.save_dir, f"col_attention_heatmap_{step}.png")
        plt.figure(figsize=(10, 8))
        # Increase text size
        sns.set(font_scale=2)
        # Set scale
        sns.heatmap(
            new_attention,
            cmap="viridis",
            vmin=0.8 / max_context_sz,
            vmax=1.2 / max_context_sz,
        )
        plt.xlabel("Protein Index")
        plt.ylabel("Protein Index")
        plt.savefig(heatmap_path)
        plt.close()

    def load_model(self) -> Optional[int]:
        assert self.save_dir is not None, "Save directory must be set"
        # Find checkpoints
        if not os.path.exists(self.save_dir):  # type: ignore
            log.info("No checkpoints found to load")
            return None
        # Find the most recent checkpoint
        checkpoints = [
            f
            for f in os.listdir(self.save_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
        max_step = -1
        for f in checkpoints:
            step = int(f.split("_")[1].split(".")[0])
            if step > max_step:
                max_step = step
        if max_step == -1:
            log.info("No checkpoints found to load")
            return None
        # Load the most recent checkpoint
        inpath = os.path.join(self.save_dir, f"checkpoint_{max_step}.pt")  # type: ignore
        checkpoint = torch.load(inpath)
        self.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"]
        return step

    def should_finetune(self, support_set: OraclePoints) -> bool:
        return not (
            len(support_set) == 0
            or self.train_config.num_finetune_at_eval is None
            or self.train_config.num_finetune_at_eval == 0
        )

    def _predict(  # noqa: CCR001
        self,
        support_set: OraclePoints,
        query_set: List[Candidate],
        task_name: str,
        early_stop_set: Optional[
            OraclePoints
        ] = None,  # For early stopping while fine-tuning
        return_params: bool = False,
    ) -> Union[np.ndarray, List[torch.nn.Parameter]]:

        assert (
            self.training is False or self.train_config.reptile
        ), "Model must be in eval mode or using Reptile"

        eval_query_size = (
            self.train_config.default_query_sz
            if self.train_config.query_size is None
            else self.train_config.query_size
        )

        if not self.should_finetune(support_set):
            # Make the predictions without fine-tuning
            with torch.no_grad():
                if self.train_config.eval_chunk_strategy == "chunk":
                    # Chunk up the query set
                    preds_list = []
                    for i in range(0, len(query_set), eval_query_size):
                        query_chunk = query_set[i : i + eval_query_size]
                        chunk_preds, _, _, _, _ = self.embed_and_forward(
                            support_set, query_chunk, task_name, False
                        )
                        preds_list.append(chunk_preds)
                    preds = torch.cat(preds_list, dim=0)
                else:
                    # Assume that the query set can be arbitrarily large
                    assert (
                        self.train_config.eval_chunk_strategy == "full"
                    ), "Invalid eval_chunk_strategy"
                    preds, _, _, _, _ = self.embed_and_forward(
                        support_set, query_set, task_name, False
                    )
                return preds.detach().cpu().numpy()

        # otherwise, we need to fine-tune the model on the support set:
        assert (
            len(support_set) > 0
        ), "Support set must have at least one example to finetune"

        # Get a seed for fit.
        # Note that numpy and torch are seeded in run_metasupervised.py,
        # so we can just request a random integer here.
        seed = np.random.randint(0, 1000)

        # Save the parameters
        params = deepcopy(self.state_dict())
        # Save the optimizer state
        optimizer_state = deepcopy(self.optimizer.state_dict())
        # Save the support and query sizes
        original_support_size = self.support_size
        original_query_size = self.query_size
        # Save the LR scheduler state
        if self.train_config.use_lr_scheduler:
            scheduler_state = deepcopy(self.scheduler.state_dict())

        # Load initial optimizer state
        self.optimizer.load_state_dict(self.initial_optimizer_params)
        # Load initial LR scheduler state, but change step to after warmup
        if self.train_config.use_lr_scheduler:
            after_warmup_step = int(
                self.train_config.warmup_frac * self.train_config.num_sched_steps
            )
            new_state_dict = deepcopy(self.initial_scheduler_params)
            new_state_dict["_step_count"] = after_warmup_step
            new_state_dict["_last_lr"] = [
                self.train_config.learning_rate,
                self.train_config.learning_rate,
            ]
            # Skip warmup step if not using warmup LR for fine-tuning
            # Note that even if warmup is not used, the scheduler will still be used
            # And the very first step will be the full LR
            if not self.train_config.warmup_lr_for_finetune:
                new_state_dict["last_epoch"] = (
                    after_warmup_step  # Step used in LambdaLR
                )
            self.scheduler.load_state_dict(new_state_dict)

        # Divide the support and query sets in half to avoid overlap
        self.support_size = len(support_set) // 2
        self.query_size = len(query_set) // 2
        # Limit the query size to half the support size since we can't use more data than in support
        self.query_size = min(self.query_size, len(support_set) // 2)

        # Fit the model on the support set
        self.fit(
            {task_name: support_set},
            seed,
            None,
            None,
            (
                self.train_config.reptile_num_finetune_at_train
                if self.train_config.reptile and self.training
                else self.train_config.num_finetune_at_eval
            ),
            early_stop_set=early_stop_set,
            finetuning=True,
        )
        if return_params:
            new_params = deepcopy(list(self.parameters()))

        # Make the predictions
        if not return_params:
            self.eval()  # Make sure the model is in eval mode for predictions after training
            with torch.no_grad():
                if self.train_config.finetune_eval_chunk_strategy == "split":
                    # Split the support and query set to match the sizes used for fine-tuning
                    assert len(
                        query_set <= eval_query_size
                    ), "Query set too large; chunking not implemented for split support."
                    # Split the support and query in half and make predictions on each half
                    support1 = support_set[: self.support_size // 2]
                    query1 = query_set[: self.query_size // 2]
                    preds1, _, _, _, _ = self.embed_and_forward(
                        support1, query1, task_name, False
                    )
                    support2 = support_set[self.support_size // 2 :]
                    query2 = query_set[self.query_size // 2 :]
                    preds2, _, _, _, _ = self.embed_and_forward(
                        support2, query2, task_name, False
                    )
                    preds = torch.cat([preds1, preds2], dim=0)
                elif self.train_config.finetune_eval_chunk_strategy == "full":
                    # Assume that the support is not split and query can be arbitrarily large
                    preds, _, _, _, _ = self.embed_and_forward(
                        support_set, query_set, task_name, False
                    )
                else:
                    assert (
                        self.train_config.finetune_eval_chunk_strategy == "chunk"
                    ), "Invalid finetune_eval_chunk_strategy"
                    preds_list = []
                    for i in range(0, len(query_set), eval_query_size):
                        query_chunk = query_set[i : i + eval_query_size]
                        chunk_preds, _, _, _, _ = self.embed_and_forward(
                            support_set, query_chunk, task_name, False
                        )
                        preds_list.append(chunk_preds)
                    preds = torch.cat(preds_list, dim=0)

        # Undo the fine-tuning
        # Reset the parameters
        self.load_state_dict(params)
        # Reset the optimizer
        self.optimizer.load_state_dict(optimizer_state)
        # Reset the support and query sizes
        self.support_size = original_support_size
        self.query_size = original_query_size
        # Reset the LR scheduler state
        if self.train_config.use_lr_scheduler:
            self.scheduler.load_state_dict(scheduler_state)

        if return_params:
            return new_params

        return preds.detach().cpu().numpy()

    def embed_and_forward(  # noqa: CCR001
        self,
        support_set: OraclePoints,
        query_set: List[Candidate],
        task_name: str,
        return_all_y: bool,
        return_attention: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # Get support x and y and query x
        support_values = support_set.oracle_values
        support_candidates = support_set.candidate_points
        sy = torch.tensor(support_values).float().to(device)
        sx = self.embed(support_candidates) if len(support_candidates) > 0 else None
        qx = self.embed(query_set)
        # Get wild type embedding if conditioning on wild type
        wild_type = self.metadata["target_seq"][task_name]
        wx = self.embed([Candidate(wild_type)]) if self.condition_on_wild_type else None
        # Get auxiliary 0-shot predictions if using them
        aux_predictions = None
        if self.aux_pred:
            # Manually create the aux zero-shot predictions
            if self.aux_pred_model is None:
                assert (
                    self.train_config.freeze_embed
                ), "Expected frozen embeddings with aux_pred."
                assert (
                    type(self.embedding_model) == ESMEmbedding  # type: ignore
                ), "Expected ESMEmbedding"
                all_seqs = (
                    [c.sequence for c in support_candidates]
                    + [c.sequence for c in query_set]
                    + [wild_type]
                )
                aux_predictions = self.embedding_model.wt_marginal_predictions(
                    wild_type, all_seqs
                ).detach()
                assert (
                    aux_predictions[-1] == 0.0
                ), f"WT should be 0, found {aux_predictions[-1]}"
                aux_predictions = (
                    aux_predictions - torch.mean(aux_predictions)
                ) / torch.std(aux_predictions)
                if self.make_wt_aux_pred_zero:
                    aux_predictions = aux_predictions - aux_predictions[-1]
                    assert (
                        aux_predictions[-1] == 0.0
                    ), f"WT should be 0, found {aux_predictions[-1]}"
                # Remove wild type from aux_predictions 0-shot predictions if needed
                if not self.condition_on_wild_type:
                    aux_predictions = aux_predictions[:-1]
            # Load zero shot aux_predictions
            else:
                all_candidates = support_candidates + query_set
                aux_predictions_list = [
                    c.features["standardized_" + self.aux_pred_model]
                    for c in all_candidates
                ]
                aux_predictions = (
                    torch.tensor(aux_predictions_list).float().to(device).detach()
                )
                assert not self.condition_on_wild_type, "aux_model not yet supported"

        return self.forward(
            sx, sy, qx, wx, aux_predictions, return_all_y, return_attention
        )

    def forward(  # noqa: CCR001
        self,
        sx: Optional[torch.Tensor],
        sy: torch.Tensor,
        qx: torch.Tensor,
        wx: Optional[torch.Tensor],
        aux_y: Optional[torch.Tensor],
        return_all_y: bool,
        return_attention: bool,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Forward pass through the model. Takes in support set sequences and values,
        query set sequences, and the wild type sequence, which may be ignored.
        Optionally, also take in auxiliary 0-shot predictions.
        """
        # Check input shapes
        if sx is None:
            sN, L, D = 0, qx.shape[1], qx.shape[2]  # noqa: N806
        else:
            assert sx.shape[0] == sy.shape[0], "sx and sy must have the same batch size"
            assert len(sy.shape) == 1, "sy must be of shape (batch_size,)"
            sN, L, D = sx.shape  # noqa: N806
        qN, qL, qD = qx.shape  # noqa: N806
        N = sN + qN  # noqa: N806
        assert (
            len(qx.shape) == 3
        ), "qx must be of shape (query_batch_size, seq_len, embed_dim)"
        assert (L, D) == (qL, qD), "sx and qx must have the same seq_len and embed_dim"
        if self.condition_on_wild_type:
            N += 1
            assert wx.shape == (1, L, D), "wx must be of shape (1, L, D)"  # type: ignore
        else:
            assert wx is None, "wx must be None if not conditioning on wild type"
        if aux_y is None:
            assert (
                not self.aux_pred
            ), "aux_y must not be None if using auxiliary predictions"
        else:
            assert (
                self.aux_pred
            ), "aux_y must be None if not using auxiliary predictions"
            assert aux_y.shape == (
                N,
            ), f"aux_y must be of shape (N,), found {aux_y.shape}"

        # Normalize or standardize sy:
        sy_avg_norm = 1.0 if sx is None else torch.mean(torch.abs(sy))
        sy_mean = 0.0 if sx is None else torch.mean(sy)
        sy_std = 1.0 if sx is None or len(sy) <= 1 else torch.std(sy)
        # Avoid division by zero
        if sy_std == 0.0:
            sy_std = 1.0
        if self.context_normalization == "standardize":
            sy = (sy - sy_mean) / sy_std
        elif self.context_normalization == "normalize":
            # Note: L1 and L2 norms are the same for scalar values
            sy = sy / sy_avg_norm
        else:
            assert self.context_normalization is None, "Invalid normalization"
            assert (
                self.train_config.landscape_normalization is not None
            ), "Either landscape_normalization or context_normalization must be set"
            assert (
                not self.using_synthetic_data
            ), "synthetic data will not be normalized"

        # Project token embeddings if necessary
        if sx is not None:
            sx_proj = (
                sx
                if self.token_embedding_projection is None
                else self.token_embedding_projection(sx)
            )
        qx_proj = (
            qx
            if self.token_embedding_projection is None
            else self.token_embedding_projection(qx)
        )
        x = torch.cat([sx_proj, qx_proj], dim=0) if sx is not None else qx_proj
        if self.condition_on_wild_type:
            wx_proj = (
                wx
                if self.token_embedding_projection is None
                else self.token_embedding_projection(wx)
            )
            x = torch.cat([x, wx_proj], dim=0)
            assert (
                not self.using_synthetic_data
            ), "synthetic wild types not yet supported"

        # Apply CNN
        if self.use_cnn_before_npt and self.cnn_layer is not None:
            assert len(x.size()) == 3, "Size error input"
            # N, L, D = x.size()
            x = x.permute(0, 2, 1)  # N, D, L
            x = self.cnn_layer(x)
            x = x.permute(0, 2, 1)
        x = x.view(1, N, L, self.embed_dim)  # 1, N, L, D

        # Construct empty fitness values for qy and masks
        qy = torch.zeros(
            qx.size(0),
        ).to(device)
        qy_mask = torch.ones_like(qy)  # Ones for missing values
        qy_with_mask = torch.stack([qy, qy_mask], dim=1)
        sy_mask = torch.zeros_like(sy)  # Zeros for observed values
        sy_with_mask = torch.stack([sy, sy_mask], dim=1)
        y = torch.cat([sy_with_mask, qy_with_mask], dim=0)
        if self.condition_on_wild_type:
            wy = torch.zeros((1,)).to(device)
            wy_mask = torch.tensor([-1.0]).to(device)  # -1 for WT
            wy_with_mask = torch.stack([wy, wy_mask], dim=1)
            y = torch.cat([y, wy_with_mask], dim=0)

        # Embed fitness values
        y = self.f_embed_layer(y)
        y = y.unsqueeze(1).unsqueeze(0)  # 1, N, 1, D
        assert y.shape == (1, N, 1, self.embed_dim), f"Found shape {y.shape}"

        # Embed auxiliary predictions if necessary
        if aux_y is not None:
            aux_y = self.aux_embed_layer(aux_y.unsqueeze(-1))  # type: ignore
            aux_y = aux_y.unsqueeze(1).unsqueeze(0)  # type: ignore
            assert aux_y.shape == (
                1,
                N,
                1,
                self.embed_dim,
            ), f"Found shape {aux_y.shape}"

        # Concatenate x and y
        x = torch.cat((x, y), dim=-2)  # 1, N, (L+1), D
        if aux_y is not None:  # 1, N, (L+2), D
            x = torch.cat((x, aux_y), dim=-2)

        # Normalize
        x = self.emb_layer_norm_before(x)

        # Dropout
        x = self.dropout_module(x)

        # NPT Layers
        all_head_weights = []
        x = x.permute(1, 2, 0, 3)  # 1 x N x L x D -> N x L x 1 x D
        L_with_aux = L + 2 if self.aux_pred else L + 1  # noqa: N806
        assert x.shape == (
            N,
            L_with_aux,
            1,
            self.embed_dim,
        ), f"Expected shape {(N, L_with_aux, 1, self.embed_dim)}. Found shape {x.shape}"
        for layer in self.npt_layers:
            x = layer(
                x,
                self_attn_padding_mask=None,  # TODO: Do we need this for any reason?
                need_head_weights=return_attention,
            )
            if return_attention:
                x, column_attn, _ = x
                if column_attn is None:
                    assert self.deactivate_col_attention
                all_head_weights.append(column_attn)
        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # N x L x 1 x D -> 1 x N x L x D
        assert x.shape == (
            1,
            N,
            L_with_aux,
            self.embed_dim,
        ), "Error with axial transformer"
        x = x.squeeze(0)  # N x L x D
        x, y = x[:, :L, :], x[:, L:, :]  # split back into (N,L,D) and (N,1,D)
        if aux_y is not None:  # Remove aux_y
            y = y[:, :1, :]

        # Use CNN, pool sequences, and add as a feature to y, if necessary
        y_dim = self.embed_dim
        assert not (
            self.use_cnn_after_npt and not self.f_conditions_on_pooled_seq
        ), "Using CNN after NPT poiontless without pooled seq."
        if self.f_conditions_on_pooled_seq:
            if self.use_cnn_after_npt == "CNN" and self.cnn_layer is not None:
                assert len(x.size()) == 3, "Size error input"
                # N, L, D = x.size()
                x = x.permute(0, 2, 1)  # N, D, L
                x = self.layer_pre_head(x)
                x = x.permute(0, 2, 1)
            x = x.mean(dim=-2)  # N, D
            y = torch.cat((x, y.squeeze(1)), dim=-1)  # N, 2 * D
            y_dim = 2 * self.embed_dim

        # Fitness prediction
        y = y.view(N, y_dim)
        if not return_all_y:
            y = y[sN:]  # Only predict for query set
        if self.condition_on_wild_type:
            y = y[:-1]  # Remove wild type
        qy, z, mean, logvar = self.f_mlp(y)
        assert qy.shape == (N if return_all_y else qN, 1), f"Found shape {qy.shape}"
        qy = qy.squeeze(-1)
        assert (
            len(qy.shape) == 1
        ), f"Expected shape (query_batch_size,), found {qy.shape}"

        # Unnormalize qy
        if self.context_normalization == "standardize":
            qy = qy * sy_std + sy_mean
        elif self.context_normalization == "normalize":
            qy = qy * sy_avg_norm
        else:
            assert self.context_normalization is None, "Invalid normalization"

        if return_attention:
            return (
                qy,
                z,
                mean,
                logvar,
                (
                    None
                    if self.deactivate_col_attention
                    else torch.stack(all_head_weights, dim=0)
                ),
            )
        return qy, z, mean, logvar, None

    def get_training_summary_metrics(
        self,
    ) -> Mapping[str, Union[float, int]]:
        return {}

    def preference_loss(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the preference loss between true and predicted fitness values.
        """
        assert (
            y_true.shape == y_pred.shape
        ), f"Shapes, {y_true.shape} and {y_pred.shape}, must match"

        num_samples = (
            len(y_true) - 1
            if self.train_config.num_pref_samples is None
            else self.train_config.num_pref_samples
        )
        num_instances = len(y_true)

        assert (
            num_samples < num_instances
        ), "num_samples must be less than num_instances"

        # Generate a matrix of indices with shape (num_instances, num_samples)
        # Create random indices in [1,n-1]
        idx_matrix = torch.rand(
            num_instances, num_instances - 1, device=device
        ).argsort(dim=1)[:, :num_samples]
        # Add 1 to any values greater than or equal to the row index to avoid self-comparisons
        adjustment = (
            idx_matrix >= torch.arange(num_instances, device=device).unsqueeze(1)
        ).long()
        idx_matrix += adjustment

        # Gather the y_true and y_pred values using the index matrix
        y_true_gathered = y_true[idx_matrix]
        y_pred_gathered = y_pred[idx_matrix]
        assert y_true_gathered.shape == (
            num_instances,
            num_samples,
        ), f"Expected shape {(num_instances, num_samples)}, found {y_true_gathered.shape}"
        assert y_pred_gathered.shape == (
            num_instances,
            num_samples,
        ), f"Expected shape {(num_instances, num_samples)}, found {y_pred_gathered.shape}"

        # Expand y_true and y_pred for comparison
        y_true_expanded = y_true.unsqueeze(1).expand(-1, num_samples)
        y_pred_expanded = y_pred.unsqueeze(1).expand(-1, num_samples)

        # Compute the differences
        y_true_diff = y_true_expanded - y_true_gathered
        y_pred_diff = y_pred_expanded - y_pred_gathered

        # Calculate the preference loss
        loss_pos = -torch.log(torch.sigmoid(y_pred_diff)) * (y_true_diff > 0).float()
        loss_neg = -torch.log(torch.sigmoid(-y_pred_diff)) * (y_true_diff < 0).float()

        # Sum the losses and average
        loss = torch.sum(loss_pos + loss_neg) / (num_instances * num_samples)

        return loss

    def reptile_update(  # noqa: CCR001
        self, task_thetas: List[List[torch.nn.Parameter]], step: int
    ) -> None:
        """Update the model parameters using Reptile, given thetas fine-tuned from each task."""
        if self.train_config.reptile_uses_adam:
            # Use Adam to update the model parameters
            self.optimizer.zero_grad()
            # Interpret -(mean_theta - theta)/lr as the loss gradient for use with Adam
            for param_num, theta in enumerate(self.parameters()):
                # Compute mean theta
                mean_theta = torch.mean(
                    torch.stack([p[param_num] for p in task_thetas]), dim=0
                )
                # Note: The learning rate used in the inner-loop skips the warmup phase
                #       and only runs for 100 updates max, so it is apprximately
                #       self.train_config.learning_rate the whole time
                # Set the gradient
                theta.grad = (theta.data - mean_theta) / self.train_config.learning_rate
            # Clip gradients
            nn.utils.clip_grad_norm_(
                self.parameters(), self.train_config.gradient_clip_value
            )
            # Use the same Adam (and LR) as the inner loop
            if USE_TPU:
                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()
            if self.train_config.use_lr_scheduler:
                self.scheduler.step()
        else:
            # Use the Reptile update rule
            # Get lr
            lr = self.train_config.reptile_lr
            if self.train_config.use_lr_scheduler:
                frac = self.lr_frac(step)
                lr = lr * frac
                self.scheduler.step()  # For use in inner loop
            # Move toward the average of thetas
            for param_num, theta in enumerate(self.parameters()):
                mean_theta = torch.mean(
                    torch.stack([p[param_num] for p in task_thetas]), dim=0
                )
                theta.data = theta.data + lr * (mean_theta - theta.data)

    def fit(  # noqa: CCR001
        self,
        train_data: Dict[str, OraclePoints],
        seed: int,
        logger: Optional[Logger],
        eval_func: Optional[Callable],
        num_steps: Optional[int] = None,
        early_stop_set: Optional[
            OraclePoints
        ] = None,  # For early stopping while fine-tuning
        finetuning: bool = False,
    ) -> None:
        """
        Train metasurrogate model on multiple datasets of labelled candidates.
        Uses generate_support_query_splits() in BaseMetaSurrogate to generate
        data for training.
        """
        # TODO: The data_generator should probably be a dataloader, but I'm not sure if
        # we can use the PyTorch dataloader for this purpose, and this code is running fast
        assert (
            type(self.train_config.num_train_steps) is int
        ), f"Expected int, got {type(self.train_config.num_train_steps)}"
        assert (
            type(self.train_config.batch_sz) is int
        ), f"Expected int, got {type(self.train_config.batch_sz)}"

        random_st = np.random.RandomState(seed)
        data_generator = self.generate_support_query_splits(
            train_data, random_st, None, self.train_config.landscape_normalization
        )

        # Get next batch
        next_batch = list(itertools.islice(data_generator, self.train_config.batch_sz))
        step = 0
        t0 = time.time()
        best_eval = None
        best_params = None
        best_step = None
        attention_list: List[Optional[torch.Tensor]] = []  # For logging attention

        if not finetuning and self.train_config.resume_training:
            log.info("Loading Model...")
            last_step = self.load_model()
            if last_step is not None:
                step = last_step
                log.info(f"Resuming training from step {step}")

        if finetuning:
            log.info("Fine-tuning...")
        else:
            log.info("Training...")
        self.train()  # Set to train mode

        num_steps = (
            self.train_config.num_train_steps if num_steps is None else num_steps
        )
        while step < num_steps:
            batch_start_t = time.time()

            log.info("\n\n")
            log.info(
                f"Step {step + 1} of {num_steps}: "
                f"{100 * (step + 1) / num_steps:.2f}%"
            )
            log.info(
                "First query seq hash: "
                + hex(hash(str(next_batch[0][1].candidate_points[0].sequence)))[2:]
            )

            self.optimizer.zero_grad()

            batch_loss = 0.0
            task_thetas: List[List[torch.nn.Parameter]] = []  # For Reptile
            attention_list = []  # For logging attention
            for c_num, (support_set, query_set, task_name, _) in enumerate(next_batch):
                self.train()  # Ensure in train mode
                # If using Reptile, adapt thetas for each task
                if self.train_config.reptile and not finetuning:
                    task_theta = self._predict(
                        support_set,
                        query_set.candidate_points,
                        task_name,
                        return_params=True,
                    )
                    task_thetas.append(task_theta)  # type: ignore
                    self.train()  # Ensure back to train mode after fine-tuning in predict
                    continue
                # Back propagate loss for each context in the batch sequentially
                # in order to avoid issues with different contexts
                #   TODO: Parallelize over devices
                if self.train_config.use_dummy_padding:
                    support_set, query_set = dummy_padding(support_set), dummy_padding(
                        query_set
                    )
                # TODO: When not limiting protein sizes, this is printing occasionally:
                # Token indices sequence length is longer than the specified maximum sequence
                # length for this model (1156 > 1024). Running this sequence through the model
                # will result in indexing errors. This might be from the tokenizer in ESM, but
                # could be from NPT layers. (At least during evaluation.)
                # Log stats
                all_seqs = support_set.candidate_points + query_set.candidate_points
                log.info(f"- C{c_num} Num sequences: {len(all_seqs)}")
                max_seq_len = max(len(c.sequence) for c in all_seqs)
                log.info(f"- C{c_num} Max sequence length: {max_seq_len}")
                total_tokens_in_context = sum(len(c.sequence) for c in all_seqs)
                log.info(
                    f"- C{c_num} Total tokens in context: {total_tokens_in_context}"
                )
                # Predict fitness values
                (
                    predicted_qy,
                    _,
                    latent_mean,
                    latent_logvar,
                    attention,
                ) = self.embed_and_forward(
                    support_set,
                    query_set.candidate_points,
                    task_name,
                    self.train_config.apply_loss_to_support,
                    return_attention=True,
                )
                attention_list.append(attention)
                # Compute loss. First normalize to prevent large loss
                qy = torch.tensor(query_set.oracle_values).float().to(device)
                sy = torch.tensor(support_set.oracle_values).float().to(device)
                combined_y = torch.cat([sy, qy], dim=-1)
                if self.train_config.apply_loss_to_support:
                    qy = combined_y
                if self.context_normalization == "standardize":
                    context_mean = torch.mean(combined_y)
                    context_std = 1.0 if len(combined_y) <= 1 else torch.std(combined_y)
                    qy_normd = (qy - context_mean) / context_std
                    pred_qy_normd = (predicted_qy - context_mean) / context_std
                    sy_mean_normd = (sy.mean() - context_mean) / context_std
                elif self.context_normalization == "normalize":
                    # Note: L1 and L2 norms are the same for scalar values
                    y_avg_norm = torch.mean(torch.abs(combined_y))
                    qy_normd = qy / y_avg_norm
                    pred_qy_normd = predicted_qy / y_avg_norm
                    sy_mean_normd = sy.mean() / y_avg_norm
                else:
                    qy_normd, pred_qy_normd, sy_mean_normd = qy, predicted_qy, sy.mean()
                    assert self.context_normalization is None, "Invalid normalization"
                # Compute MSE loss
                assert (
                    qy_normd.shape == pred_qy_normd.shape
                ), f"Shape mismatch: {qy_normd.shape} and {pred_qy_normd.shape}"
                mse_loss = MSELoss(reduction="mean")(qy_normd, pred_qy_normd)
                # Set loss and copmute preference loss if needed
                pref_loss = None
                loss = None
                if self.train_config.loss_type == "mse":
                    loss = mse_loss
                elif self.train_config.loss_type == "ranking_sampled":
                    pref_loss = self.preference_loss(qy_normd, pred_qy_normd)
                    loss = pref_loss
                else:
                    pref_loss = get_loss(
                        self.train_config.loss_type,
                        label_smoothing=self.train_config.label_smoothing,
                        label_smoothing_beta=self.train_config.label_smoothing_beta,
                    )(pred_qy_normd, qy_normd)
                    loss = pref_loss
                # Add KL to N(0,1)
                if self.latent_index is not None:
                    assert (
                        latent_mean is not None and latent_logvar is not None
                    ), "Latent layers not created when expected"
                    assert latent_mean.size(0) == qy.size(
                        0
                    ), "Latent layer size mismatch"
                    kl_div = -0.5 * torch.sum(
                        1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp()
                    )
                    kl_div = kl_div / latent_mean.size(0)  # Normalize by context size
                    loss += self.train_config.kl_weight * kl_div
                loss = loss / len(next_batch)  # Avearge loss over batch
                # Check if NaN
                if torch.isnan(loss):
                    # If using preference loss, and all preferences are the same, then skip loss
                    using_pref_loss = self.train_config.loss_type != "mse"
                    all_pref_same = torch.all(qy == qy[0]).item()
                    if using_pref_loss and all_pref_same:
                        continue
                    # Otherwise, raise error
                    log.info(f"NaN loss at step {step}")
                    log.info(f"Predicted qy: {predicted_qy}")
                    log.info(f"True qy: {qy}")
                    log.info(f"Task name: {task_name}")
                    log.info(f"Qy normd: {qy_normd}")
                    log.info(f"Pred qy normd: {pred_qy_normd}")
                    log.info(f"Sy: {sy}")
                    raise ValueError("NaN loss")
                # backpropagate, but don't update weights yet
                loss.backward()
                batch_loss += loss.item()
                # Log info:
                mse_loss_baseline = MSELoss(reduction="mean")(qy_normd, sy_mean_normd)
                context_info = {
                    "Mean_predicted_fitness": predicted_qy.mean().item(),
                    "Mean_actual_fitness": qy.mean().item(),
                    "Variance_predicted_fitness": predicted_qy.var().item(),
                    "Variance_actual_fitness": qy.var().item(),
                    "Mean_normd_predicted_fitness": pred_qy_normd.mean().item(),
                    "Mean_normd_actual_fitness": qy_normd.mean().item(),
                    "Variance_normd_predicted_fitness": pred_qy_normd.var().item(),
                    "Variance_normd_actual_fitness": qy_normd.var().item(),
                    "MSE_Loss": mse_loss.item(),
                    "MSE_Loss_if_predicting_mean": mse_loss_baseline.item(),
                }
                if pref_loss is not None:
                    context_info["Pref_Loss"] = pref_loss.item()
                for k, v in context_info.items():
                    k_space = k.replace("_", " ")
                    log.info(f"- C{c_num} {k_space}: {v:.7f}")
                context_info["Loss"] = loss.item()
                context_info["data_loss"] = loss.item()
                if len(qy) > 1:
                    spearman = spearmanr(
                        qy.detach().cpu().numpy(), predicted_qy.detach().cpu().numpy()
                    )[0]
                    context_info["spearman"] = spearman
                if logger is not None:
                    logger.write(
                        context_info,
                        label="context_info",
                        timestep=step + (c_num / len(next_batch)),
                    )

            # If using Reptile, adapt thetas for each task
            if self.train_config.reptile and not finetuning:
                self.reptile_update(task_thetas, step)
            else:
                # Print current lr
                # print(f"Current LR: {self.optimizer.param_groups[0]['lr']}")
                log.info(f"Training Loss: {batch_loss:.5f}")
                nn.utils.clip_grad_norm_(
                    self.parameters(), self.train_config.gradient_clip_value
                )
                if USE_TPU:
                    xm.optimizer_step(self.optimizer)
                else:
                    self.optimizer.step()
                if self.train_config.use_lr_scheduler:
                    self.scheduler.step()
                if logger is not None:
                    logger.write(
                        {"training_loss": batch_loss},
                        label="training",
                        timestep=step,
                    )
            next_batch = list(
                itertools.islice(data_generator, self.train_config.batch_sz)
            )

            step += 1
            if not USE_TPU:
                allocated_memory_gb = torch.cuda.memory_allocated(device) / 1e9
                log.info(f"Memory allocated on device: {allocated_memory_gb:.2f} GB")
                reserved_memory_gb = torch.cuda.memory_reserved(device) / 1e9
                log.info(f"Memory reserved on device: {reserved_memory_gb:.2f} GB")
            log.info(f"Batch Time elapsed: {(time.time() - batch_start_t):.2f} seconds")
            log.info(f"Total Time elapsed: {int(time.time() - t0)} seconds")
            time_remaining = (time.time() - t0) / step * (num_steps - step)
            log.info(f"Average Batch Time: {((time.time() - t0) / step):.2f} seconds")
            log.info(f"Estimated time remaining: {int(time_remaining)} seconds")

            # gc.collect()
            # torch.cuda.empty_cache()

            # Early stopping
            if early_stop_set is not None and finetuning:
                with torch.no_grad():
                    self.eval()
                    assert (
                        len(train_data) == 1
                    ), "Early stopping only works with one task"
                    assert task_name in train_data, train_data
                    # Get predictions on early stop set
                    es_preds, _, _, _, _ = self.embed_and_forward(
                        train_data[task_name],
                        early_stop_set.candidate_points,
                        task_name,
                        False,
                    )
                    es_preds = es_preds.detach()
                    # Compute Spearman correlation
                    es_spearman = spearmanr(
                        torch.tensor(early_stop_set.oracle_values).cpu().numpy(),
                        es_preds.cpu().numpy(),
                    )[0]
                    current_eval = (
                        es_spearman.item() if not np.isnan(es_spearman) else 0.0
                    )
                    if best_eval is None or current_eval > best_eval:
                        best_eval = current_eval
                        best_params = deepcopy(self.state_dict())
                        best_step = step
                    self.train()

            # Evaluate metrics
            if (
                (
                    step % self.train_config.eval_every == 0
                    or step >= num_steps
                    or step == 1
                )
                and logger is not None
                and not finetuning
            ):
                self.eval()
                log.info("Evaluating...")
                try:
                    log.info("Saving model...")
                    self.save_model(step)  # Save Checkpoint
                    self.save_col_attention(step, attention_list)  # Save Col Attention
                except Exception as e:
                    log.info(f"Error saving model: {e}")
                assert eval_func is not None, "Evaluation function must be provided"
                metrics, _ = eval_func(self)  # type: ignore
                logger.write(metrics, label="eval_metrics", timestep=step)
                log.info("Evaluation:", metrics)
                self.train()
            elif step == 1 and logger is not None and not finetuning:
                # Just save col attention for first step
                self.save_col_attention(step, attention_list)

        # Load best parameters if early stopping is set
        if early_stop_set is not None and finetuning:
            log.info(f"Best fine-tunning Spearman for early-stopping: {best_eval}")
            log.info(f"Step at best Spearman: {best_step}")
            self.load_state_dict(best_params)  # type: ignore

        if finetuning:
            log.info("Done fine-tuning")
            # Note: Don't want this in eval mode after fine-tuning for Reptile inner loop
        else:
            log.info("Done training")
            self.eval()
            try:
                log.info("Saving model...")
                self.save_model(step)  # Save Checkpoint
                self.save_col_attention(step, attention_list)  # Save Col Attention
            except Exception as e:
                log.info(f"Error saving model: {e}")
