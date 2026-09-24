# -*- coding: utf-8 -*-
"""MGCA-CRRF v6: complementary residual routing fusion.

This file was copied from the independent v5 package and then rewritten in the
new v6 directory. The stable ESM2/Morgan feature pipeline, four expert
encoders, bidirectional MHA interaction and two-expert MoE are retained. The
v5 uncertainty router, branch Gaussian NLL and whole-branch dropout are not
used: v6 separates a conservative global main path from an interaction-only
residual correction.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = PROJECT_ROOT / "local" / "ESM_Morgan_Hybrid_Fusion.py"
SOURCE_V4_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "ESM_Morgan_Hybrid_Fusion_nonredundant.py"
SOURCE_V5_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "v5_uar" / "ESM_Morgan_Hybrid_Fusion_uar_v5.py"
MODEL_VARIANT = "mgca_crrf_v6_complementary_residual_routing_2expert"
ABLATIONS = {
    "no",
    "protein_main",
    "drug_main",
    "cross_residual",
    "adaptive_gate",
    "residual_target",
    "explicit_interaction",
    "moe_balance",
}


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("mgca_crrf_v6_legacy_training", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import MGCA training logic from {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_module()


def __getattr__(name):
    """Forward unchanged data and feature utilities to the stable legacy module."""
    return getattr(legacy, name)


class BiasFreeRMSNorm(nn.Module):
    """RMS normalization that preserves an exact all-zero ablation tensor."""

    def __init__(self, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


class MHAInteractionFusion(nn.Module):
    """v4/v5 bidirectional expert-token MHA multiplicative interaction."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.attn_p = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_d = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.prot_query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prot_context_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prot_query_norm = BiasFreeRMSNorm(hidden_dim)
        self.drug_query_norm = BiasFreeRMSNorm(hidden_dim)
        self.drug_context_norm = BiasFreeRMSNorm(hidden_dim)
        self.prot_context_norm = BiasFreeRMSNorm(hidden_dim)
        self.dropout_p = nn.Dropout(dropout)
        self.dropout_d = nn.Dropout(dropout)

    def forward(self, prot_experts, drug_experts):
        drug_context, w_p2d = self.attn_p(
            query=prot_experts, key=drug_experts, value=drug_experts, need_weights=True
        )
        prot_context, w_d2p = self.attn_d(
            query=drug_experts, key=prot_experts, value=prot_experts, need_weights=True
        )
        prot_query = self.prot_query_norm(self.prot_query_proj(prot_experts))
        drug_query = self.drug_query_norm(self.drug_query_proj(drug_experts))
        drug_context = self.drug_context_norm(self.drug_context_proj(drug_context))
        prot_context = self.prot_context_norm(self.prot_context_proj(prot_context))
        interaction_p = self.dropout_p(prot_query * drug_context)
        interaction_d = self.dropout_d(drug_query * prot_context)
        return interaction_p.mean(dim=1), interaction_d.mean(dim=1), w_p2d, w_d2p


class ConservativeResidualGate(nn.Module):
    """Scalar cross-residual gate initialized to a small, exact probability."""

    def __init__(self, hidden_dim: int, dropout: float, initial_probability: float):
        super().__init__()
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial_probability must be strictly between zero and one")
        self.initial_probability = float(initial_probability)
        self.initial_logit = float(math.log(initial_probability / (1.0 - initial_probability)))
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, self.initial_logit)

    def forward(self, protein, drug, cross, product, difference, adaptive=True):
        if adaptive:
            logits = self.network(torch.cat([protein, drug, cross, product, difference], dim=-1))
        else:
            logits = protein.new_full((protein.shape[0], 1), self.initial_logit)
        return torch.sigmoid(logits), logits


class FullRegressionTransformer(legacy.FullRegressionTransformer):
    """MGCA-CRRF v6 with parameter-matched forward/loss ablations."""

    def __init__(
        self,
        proj_dim1=2560,
        proj_dim2=2048,
        hidden_dim=512,
        dropout=0.1,
        nums_of_experts=4,
        num_heads=8,
        moe_num_experts=2,
        ablation="no",
        base_loss_weight=0.2,
        residual_loss_weight=0.1,
        moe_balance_weight=0.01,
        residual_gate_init=0.1,
    ):
        if ablation not in ABLATIONS:
            raise ValueError(f"Unsupported v6 ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
        for name, value in (
            ("base_loss_weight", base_loss_weight),
            ("residual_loss_weight", residual_loss_weight),
            ("moe_balance_weight", moe_balance_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        super().__init__(
            proj_dim1=proj_dim1,
            proj_dim2=proj_dim2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nums_of_experts=nums_of_experts,
            num_heads=num_heads,
            moe_num_experts=2,
            ablation="no",
        )
        self.ablation = ablation
        self.base_loss_weight = float(base_loss_weight)
        self.residual_loss_weight = float(residual_loss_weight)
        self.moe_balance_weight = float(moe_balance_weight)
        self.residual_gate_init = float(residual_gate_init)
        self.residual_enabled = True

        self.expert_cross_att = MHAInteractionFusion(hidden_dim, num_heads, dropout)
        self.protein_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.drug_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.cross_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.protein_main_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_main_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.main_fusion_norm = BiasFreeRMSNorm(hidden_dim)
        self.product_norm = BiasFreeRMSNorm(hidden_dim)
        self.difference_norm = BiasFreeRMSNorm(hidden_dim)

        # Replace the inherited final MoE in-place: it now models only the
        # complementary interaction residual, not three redundant full-label
        # branches concatenated into a second compensating router.
        self.moe = legacy.MoEBlock(
            in_dim=hidden_dim * 3,
            out_dim=hidden_dim,
            num_experts=2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.residual_gate = ConservativeResidualGate(hidden_dim, dropout, residual_gate_init)
        self.last_aux = None

    def set_residual_enabled(self, enabled: bool):
        self.residual_enabled = bool(enabled)

    def forward(self, input1, input2):
        prot_experts = self.protein_expert_encoder(input1)
        drug_experts = self.drug_expert_encoder(input2)
        protein, protein_weights = self.prot_gate(prot_experts)
        drug, drug_weights = self.drug_gate(drug_experts)
        cross_p, cross_d, w_p2d, w_d2p = self.expert_cross_att(prot_experts, drug_experts)
        cross = self.cross_proj(torch.cat([cross_p, cross_d], dim=-1))

        protein = self.protein_branch_norm(protein)
        drug = self.drug_branch_norm(drug)
        cross = self.cross_branch_norm(cross)

        protein_main = self.protein_main_proj(protein)
        drug_main = self.drug_main_proj(drug)
        if self.ablation == "protein_main":
            protein_main = torch.zeros_like(protein_main)
        if self.ablation == "drug_main":
            drug_main = torch.zeros_like(drug_main)
        base_feature = self.main_fusion_norm(protein_main + drug_main)
        base_prediction = self.regressor(base_feature)

        product = self.product_norm(protein * drug)
        difference = self.difference_norm(torch.abs(protein - drug))
        if self.ablation == "explicit_interaction":
            product = torch.zeros_like(product)
            difference = torch.zeros_like(difference)
        residual_feature, moe_weights = self.moe(torch.cat([cross, product, difference], dim=-1))
        residual_prediction = self.residual_head(residual_feature)
        residual_gate, residual_gate_logits = self.residual_gate(
            protein,
            drug,
            cross,
            product,
            difference,
            adaptive=self.ablation != "adaptive_gate",
        )
        residual_active = self.residual_enabled and self.ablation != "cross_residual"
        residual_correction = (
            residual_gate * residual_prediction
            if residual_active
            else torch.zeros_like(residual_prediction)
        )
        prediction = base_prediction + residual_correction
        aux = {
            "protein_expert_weights": protein_weights,
            "drug_expert_weights": drug_weights,
            "attention_p2d": w_p2d,
            "attention_d2p": w_d2p,
            "base_prediction": base_prediction,
            "residual_prediction": residual_prediction,
            "residual_gate": residual_gate,
            "residual_gate_logits": residual_gate_logits,
            "residual_correction": residual_correction,
            "moe_weights": moe_weights,
        }
        self.last_aux = aux
        return prediction, aux

    def loss_components(self, labels, aux=None):
        aux = self.last_aux if aux is None else aux
        if aux is None:
            raise RuntimeError("loss_components requires a preceding forward pass")
        target = labels.reshape(-1, 1)
        zero = target.new_zeros(())
        base_loss = torch.mean((aux["base_prediction"] - target).pow(2))
        residual_target = (target - aux["base_prediction"]).detach()
        residual_loss = torch.mean((aux["residual_correction"] - residual_target).pow(2))
        if not self.residual_enabled or self.ablation in {"cross_residual", "residual_target"}:
            residual_loss = zero

        mean_moe = aux["moe_weights"].mean(dim=0).clamp_min(1e-8)
        uniform = 1.0 / mean_moe.numel()
        moe_balance = torch.sum(mean_moe * torch.log(mean_moe / uniform))
        if (
            not self.residual_enabled
            or self.ablation in {"cross_residual", "moe_balance"}
            or self.moe_balance_weight <= 0
        ):
            moe_balance = zero

        weighted_base = self.base_loss_weight * base_loss
        weighted_residual = self.residual_loss_weight * residual_loss
        weighted_balance = self.moe_balance_weight * moe_balance
        return {
            "base": weighted_base,
            "residual": weighted_residual,
            "moe_balance": weighted_balance,
            "total": weighted_base + weighted_residual + weighted_balance,
            "unweighted_base": base_loss,
            "unweighted_residual": residual_loss,
            "unweighted_moe_balance": moe_balance,
        }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    legacy.FullRegressionTransformer = FullRegressionTransformer
    legacy.main()


if __name__ == "__main__":
    main()
