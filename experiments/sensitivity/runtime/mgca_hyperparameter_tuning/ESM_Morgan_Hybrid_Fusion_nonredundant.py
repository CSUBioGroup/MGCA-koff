# -*- coding: utf-8 -*-
"""Corrected MGCA training entry point for parameter-matched ablations.

This module reuses the data, feature extraction, training, and reporting logic
from ``local/ESM_Morgan_Hybrid_Fusion.py`` while replacing only the model
architecture used by that entry point.

The current corrected architecture has four deliberately controlled parts:

1. Bidirectional MHA remains the cross-modal core.  Its attended opposite-side
   context is combined multiplicatively with the query-side expert, so the
   cross branch represents interactions instead of a second copy of either
   unimodal value stream.  No additive residual is used.
2. All three final branches use bias-free RMS normalization before fusion.
3. The protein, drug, and cross branches have independent sigmoid reliability
   gates.  The gates are not normalized against one another.
4. Every ablation keeps the final MoE input at 3H and uses a zero mask.  The
   final MoE has two compact experts with hidden width H.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = PROJECT_ROOT / "local" / "ESM_Morgan_Hybrid_Fusion.py"


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("mgca_legacy_training", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import MGCA training logic from {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_module()


def __getattr__(name):
    """Forward data/training utilities required by the tuning driver."""
    return getattr(legacy, name)


class BiasFreeRMSNorm(nn.Module):
    """RMS normalization that preserves an exact zero vector as zero."""

    def __init__(self, hidden_dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


class MHAInteractionFusion(nn.Module):
    """Bidirectional MHA followed by query-context multiplicative interaction."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.attn_p = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_d = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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
            query=prot_experts,
            key=drug_experts,
            value=drug_experts,
            need_weights=True,
        )
        prot_context, w_d2p = self.attn_d(
            query=drug_experts,
            key=prot_experts,
            value=prot_experts,
            need_weights=True,
        )

        # MHA is retained in full, including its learned value projections.
        # Multiplication makes each direction depend jointly on its own query
        # expert and the attended opposite-modality context.  Unlike an
        # additive residual or raw value pooling, neither output is a direct
        # copy of a global unimodal branch.
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


class IndependentBranchReliabilityGate(nn.Module):
    """Scale each branch independently in (0, 2), initialized at exactly 1."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.protein_gate = nn.Linear(hidden_dim, 1)
        self.drug_gate = nn.Linear(hidden_dim, 1)
        self.cross_gate = nn.Linear(hidden_dim, 1)
        for gate in (self.protein_gate, self.drug_gate, self.cross_gate):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    @staticmethod
    def _scale_branch(feature, gate):
        scale = 2.0 * torch.sigmoid(gate(feature))
        return feature * scale, scale

    def forward(self, protein, drug, cross):
        protein, protein_scale = self._scale_branch(protein, self.protein_gate)
        drug, drug_scale = self._scale_branch(drug, self.drug_gate)
        cross, cross_scale = self._scale_branch(cross, self.cross_gate)
        scales = torch.cat([protein_scale, drug_scale, cross_scale], dim=-1)
        return protein, drug, cross, scales


class FullRegressionTransformer(legacy.FullRegressionTransformer):
    """MGCA with non-redundant cross updates and fixed-3H ablations."""

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
    ):
        if ablation not in {"no", "drug", "target", "bicross"}:
            raise ValueError(f"Unsupported ablation: {ablation}")

        # The legacy training entry point still passes its historical value 4.
        # This v3 architecture intentionally fixes the final MoE at 2 experts.
        moe_num_experts = 2

        # Reuse the stable encoders, fusion modules, and regressor.  The legacy
        # cross attention/MoE allocated here are immediately replaced below.
        super().__init__(
            proj_dim1=proj_dim1,
            proj_dim2=proj_dim2,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nums_of_experts=nums_of_experts,
            num_heads=num_heads,
            moe_num_experts=moe_num_experts,
            ablation=ablation,
        )
        self.expert_cross_att = MHAInteractionFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.prot_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.drug_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.cross_branch_norm = BiasFreeRMSNorm(hidden_dim)
        self.branch_reliability = IndependentBranchReliabilityGate(hidden_dim)
        self.moe = legacy.MoEBlock(
            in_dim=hidden_dim * 3,
            out_dim=hidden_dim,
            num_experts=moe_num_experts,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.last_branch_scales = None

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

        prot_fused, drug_fused, cross_fused, branch_scales = self.branch_reliability(
            prot_fused, drug_fused, cross_fused
        )

        # Apply the ablation after the three independent branch-local gates.
        # The zeroed branch cannot influence either remaining gate, while all
        # variants retain identical modules and a fixed 3H MoE input.
        if self.ablation == "drug":
            drug_fused = torch.zeros_like(drug_fused)
        elif self.ablation == "target":
            prot_fused = torch.zeros_like(prot_fused)
        elif self.ablation == "bicross":
            cross_fused = torch.zeros_like(cross_fused)

        self.last_branch_scales = branch_scales.detach()

        combined = torch.cat([prot_fused, drug_fused, cross_fused], dim=-1)
        moe_out, _ = self.moe(combined)
        out = self.regressor(moe_out)
        return out, (prot_weights, drug_weights, w_p2d, w_d2p)


def main():
    # Functions defined in the legacy module resolve this global at runtime.
    legacy.FullRegressionTransformer = FullRegressionTransformer
    legacy.main()


if __name__ == "__main__":
    main()
