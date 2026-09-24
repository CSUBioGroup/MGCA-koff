# -*- coding: utf-8 -*-
"""MGCA-UAR v5 model copied from the v4 non-redundant MGCA entry point.

The v4 source is intentionally left untouched.  This module keeps its stable
feature extraction, expert encoders, bidirectional MHA interaction and compact
two-expert final MoE, while replacing the self-only branch gates with joint,
uncertainty-aware reliability routing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = PROJECT_ROOT / "local" / "ESM_Morgan_Hybrid_Fusion.py"
SOURCE_V4_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "ESM_Morgan_Hybrid_Fusion_nonredundant.py"
MODEL_VARIANT = "mgca_uar_v5_joint_uncertainty_router_fixed3h_2expert"
ABLATIONS = {"no", "drug", "target", "bicross", "router", "uncertainty", "branch_dropout"}


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("mgca_uar_v5_legacy_training", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import MGCA training logic from {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_module()


def __getattr__(name):
    """Forward unchanged data/feature utilities to the stable legacy module."""
    return getattr(legacy, name)


class BiasFreeRMSNorm(nn.Module):
    """RMS normalization preserving an exact zero vector."""

    def __init__(self, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


class MHAInteractionFusion(nn.Module):
    """The v4 bidirectional MHA query-context multiplicative interaction."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.attn_p = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_d = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
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
            query=prot_experts, key=drug_experts, value=drug_experts,
            need_weights=True,
        )
        prot_context, w_d2p = self.attn_d(
            query=drug_experts, key=prot_experts, value=prot_experts,
            need_weights=True,
        )
        prot_query = self.prot_query_norm(self.prot_query_proj(prot_experts))
        drug_query = self.drug_query_norm(self.drug_query_proj(drug_experts))
        drug_context = self.drug_context_norm(self.drug_context_proj(drug_context))
        prot_context = self.prot_context_norm(self.prot_context_proj(prot_context))
        interaction_p = self.dropout_p(prot_query * drug_context)
        interaction_d = self.dropout_d(drug_query * prot_context)
        return (
            interaction_p.mean(dim=1),
            interaction_d.mean(dim=1),
            w_p2d,
            w_d2p,
        )


class BranchDistributionHead(nn.Module):
    """Predict a branch-local mean and bounded log variance."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, 1)
        self.logvar = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, feature):
        return self.mean(feature), self.logvar(feature).clamp(-8.0, 8.0)


class JointUncertaintyReliabilityRouter(nn.Module):
    """Joint, non-competitive reliability routing for all three branches."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        joint_dim = hidden_dim * 5 + 3
        self.content_router = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )
        final = self.content_router[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        protein,
        drug,
        cross,
        branch_logvars,
        availability,
        use_router: bool = True,
        use_uncertainty: bool = True,
    ):
        if not use_router:
            return availability, torch.zeros_like(availability)
        visible_protein = protein * availability[:, 0:1]
        visible_drug = drug * availability[:, 1:2]
        visible_cross = cross * availability[:, 2:3]
        joint = torch.cat(
            [
                visible_protein,
                visible_drug,
                visible_cross,
                visible_protein * visible_drug,
                torch.abs(visible_protein - visible_drug),
                availability,
            ],
            dim=-1,
        )
        logits = self.content_router(joint)
        if use_uncertainty:
            logits = logits - 0.5 * branch_logvars
        scales = 2.0 * torch.sigmoid(logits) * availability
        return scales, logits


class FullRegressionTransformer(legacy.FullRegressionTransformer):
    """MGCA-UAR v5 with fixed-3H parameter-matched ablations."""

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
        branch_dropout_prob=0.15,
        auxiliary_nll_weight=0.1,
    ):
        if ablation not in ABLATIONS:
            raise ValueError(f"Unsupported v5 ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
        if not 0.0 <= branch_dropout_prob < 1.0:
            raise ValueError("branch_dropout_prob must be in [0, 1)")
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
        self.branch_dropout_prob = float(branch_dropout_prob)
        self.auxiliary_nll_weight = float(auxiliary_nll_weight)
        self.expert_cross_att = MHAInteractionFusion(hidden_dim, num_heads, dropout)
        self.prot_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.drug_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.cross_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.protein_distribution = BranchDistributionHead(hidden_dim)
        self.drug_distribution = BranchDistributionHead(hidden_dim)
        self.cross_distribution = BranchDistributionHead(hidden_dim)
        self.reliability_router = JointUncertaintyReliabilityRouter(hidden_dim, dropout)
        self.moe = legacy.MoEBlock(
            in_dim=hidden_dim * 3,
            out_dim=hidden_dim,
            num_experts=2,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.last_aux = None

    def _availability(self, batch_size: int, device, dtype):
        availability = torch.ones(batch_size, 3, device=device, dtype=dtype)
        if self.training and self.ablation != "branch_dropout" and self.branch_dropout_prob > 0:
            drop_any = torch.rand(batch_size, device=device) < self.branch_dropout_prob
            branch_index = torch.randint(0, 3, (batch_size,), device=device)
            availability[drop_any, branch_index[drop_any]] = 0.0
        permanent_index = {"target": 0, "drug": 1, "bicross": 2}.get(self.ablation)
        if permanent_index is not None:
            availability[:, permanent_index] = 0.0
        return availability

    def forward(self, input1, input2):
        prot_experts = self.protein_expert_encoder(input1)
        drug_experts = self.drug_expert_encoder(input2)
        prot_fused, prot_weights = self.prot_gate(prot_experts)
        drug_fused, drug_weights = self.drug_gate(drug_experts)
        cross_p, cross_d, w_p2d, w_d2p = self.expert_cross_att(
            prot_experts, drug_experts
        )
        cross_fused = self.cross_proj(torch.cat([cross_p, cross_d], dim=-1))

        prot_fused = self.prot_branch_norm(prot_fused)
        drug_fused = self.drug_branch_norm(drug_fused)
        cross_fused = self.cross_branch_norm(cross_fused)

        prot_mean, prot_logvar = self.protein_distribution(prot_fused)
        drug_mean, drug_logvar = self.drug_distribution(drug_fused)
        cross_mean, cross_logvar = self.cross_distribution(cross_fused)
        branch_means = torch.cat([prot_mean, drug_mean, cross_mean], dim=-1)
        branch_logvars = torch.cat([prot_logvar, drug_logvar, cross_logvar], dim=-1)
        availability = self._availability(
            input1.shape[0], input1.device, prot_fused.dtype
        )
        router_scales, router_logits = self.reliability_router(
            prot_fused,
            drug_fused,
            cross_fused,
            branch_logvars,
            availability,
            use_router=self.ablation != "router",
            use_uncertainty=self.ablation != "uncertainty",
        )
        combined = torch.cat(
            [
                prot_fused * router_scales[:, 0:1],
                drug_fused * router_scales[:, 1:2],
                cross_fused * router_scales[:, 2:3],
            ],
            dim=-1,
        )
        moe_out, moe_weights = self.moe(combined)
        prediction = self.regressor(moe_out)
        aux = {
            "protein_expert_weights": prot_weights,
            "drug_expert_weights": drug_weights,
            "attention_p2d": w_p2d,
            "attention_d2p": w_d2p,
            "router_scales": router_scales,
            "router_logits": router_logits,
            "branch_means": branch_means,
            "branch_logvars": branch_logvars,
            "availability": availability,
            "moe_weights": moe_weights,
        }
        self.last_aux = aux
        return prediction, aux

    def auxiliary_loss(self, labels, aux=None):
        if self.ablation == "uncertainty" or self.auxiliary_nll_weight <= 0:
            return labels.new_zeros(())
        aux = self.last_aux if aux is None else aux
        if aux is None:
            raise RuntimeError("auxiliary_loss requires a preceding forward pass")
        target = labels.reshape(-1, 1).expand_as(aux["branch_means"])
        logvar = aux["branch_logvars"]
        nll = 0.5 * (torch.exp(-logvar) * (target - aux["branch_means"]).pow(2) + logvar)
        availability = aux["availability"]
        normalized = (nll * availability).sum() / availability.sum().clamp_min(1.0)
        return self.auxiliary_nll_weight * normalized


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    # Kept as a compatibility entry point.  Formal v5 experiments use the
    # artifact-rich train_v5_uar.py driver in this directory.
    legacy.FullRegressionTransformer = FullRegressionTransformer
    legacy.main()


if __name__ == "__main__":
    main()
