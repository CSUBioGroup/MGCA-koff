# -*- coding: utf-8 -*-
"""MGCA-PADG v7: protein-anchored drug gating.

v7 is an independent architecture derived from the v6 ablation evidence. It
keeps the stable ESM2/Morgan four-expert encoders and expert fusion, but removes
the output-level cross residual, explicit product/difference residual features,
residual-target loss and final MoE. Drug information is injected once, as a
bounded representation-level modulation of a protein anchor.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCRIPT = PROJECT_ROOT / "local" / "ESM_Morgan_Hybrid_Fusion.py"
SOURCE_V4_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "ESM_Morgan_Hybrid_Fusion_nonredundant.py"
SOURCE_V5_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "v5_uar" / "ESM_Morgan_Hybrid_Fusion_uar_v5.py"
SOURCE_V6_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "v6_crrf" / "ESM_Morgan_Hybrid_Fusion_crrf_v6.py"
MODEL_VARIANT = "mgca_padg_v7_protein_anchored_bounded_drug_gating"
MAX_GRAD_NORM = 5.0
ABLATIONS = {
    "no",
    "protein_anchor",
    "drug_modulation",
    "reliability_gate",
    "drug_dropout",
    "protein_aux",
    "contribution_penalty",
    "expert_gates",
}


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("mgca_padg_v7_legacy", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import stable MGCA utilities from {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_module()


def __getattr__(name):
    return getattr(legacy, name)


class BiasFreeRMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        return self.scale * x / x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()


class LowCapacityReliabilityGate(nn.Module):
    """Bounded scalar gate driven only by four alignment statistics + mask."""

    def __init__(self, hidden_dim: int, dropout: float, initial_gate: float, gate_cap: float):
        super().__init__()
        if not 0.0 < initial_gate < gate_cap <= 1.0:
            raise ValueError("require 0 < initial_gate < gate_cap <= 1")
        self.initial_gate = float(initial_gate)
        self.gate_cap = float(gate_cap)
        probability = initial_gate / gate_cap
        self.initial_logit = float(math.log(probability / (1.0 - probability)))
        bottleneck = max(hidden_dim // 16, 16)
        self.network = nn.Sequential(
            nn.Linear(5, bottleneck),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, self.initial_logit)

    def forward(self, statistics, adaptive=True):
        if adaptive:
            logits = self.network(statistics)
        else:
            logits = statistics.new_full((statistics.shape[0], 1), self.initial_logit)
        return self.gate_cap * torch.sigmoid(logits), logits


class FullRegressionTransformer(nn.Module):
    """PADG v7 with parameter-matched forward/loss ablations."""

    def __init__(
        self,
        proj_dim1=2560,
        proj_dim2=2048,
        hidden_dim=512,
        dropout=0.1,
        nums_of_experts=4,
        ablation="no",
        protein_aux_weight=0.15,
        contribution_penalty_weight=0.002,
        drug_gate_init=0.2,
        drug_gate_cap=0.5,
        drug_dropout=0.25,
    ):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"Unsupported v7 ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
        if protein_aux_weight < 0 or contribution_penalty_weight < 0:
            raise ValueError("auxiliary weights must be non-negative")
        if not 0.0 <= drug_dropout < 1.0:
            raise ValueError("drug_dropout must be in [0,1)")

        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.protein_aux_weight = float(protein_aux_weight)
        self.contribution_penalty_weight = float(contribution_penalty_weight)
        self.drug_gate_init = float(drug_gate_init)
        self.drug_gate_cap = float(drug_gate_cap)
        self.drug_dropout = float(drug_dropout)
        self.drug_enabled = True

        self.protein_expert_encoder = legacy.MultiExpertEncoder(
            in_dim=proj_dim1, hidden_dim=hidden_dim, max_experts=nums_of_experts
        )
        self.drug_expert_encoder = legacy.MultiExpertEncoder(
            in_dim=proj_dim2, hidden_dim=hidden_dim, max_experts=nums_of_experts
        )
        self.prot_gate = legacy.GatedExpertFusion(nums_of_experts, hidden_dim)
        self.drug_gate = legacy.GatedExpertFusion(nums_of_experts, hidden_dim)

        self.protein_norm = BiasFreeRMSNorm(hidden_dim)
        self.drug_norm = BiasFreeRMSNorm(hidden_dim)
        self.protein_anchor_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_delta_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.anchor_norm = BiasFreeRMSNorm(hidden_dim)
        self.delta_norm = BiasFreeRMSNorm(hidden_dim)
        self.fusion_norm = BiasFreeRMSNorm(hidden_dim)
        self.reliability_gate = LowCapacityReliabilityGate(
            hidden_dim, dropout, drug_gate_init, drug_gate_cap
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.last_aux = None

    def set_drug_enabled(self, enabled: bool):
        self.drug_enabled = bool(enabled)

    def _fuse_experts(self, experts, gate_module):
        gated, weights = gate_module(experts)
        if self.ablation == "expert_gates":
            uniform = experts.new_full(
                (experts.shape[0], experts.shape[1]), 1.0 / experts.shape[1]
            )
            return experts.mean(dim=1), uniform
        return gated, weights

    def _availability(self, batch_size: int, reference):
        if not self.drug_enabled or self.ablation == "drug_modulation":
            return reference.new_zeros((batch_size, 1))
        if self.training and self.ablation != "drug_dropout" and self.drug_dropout > 0:
            keep = (torch.rand((batch_size, 1), device=reference.device) >= self.drug_dropout).to(reference.dtype)
            return keep / (1.0 - self.drug_dropout)
        return reference.new_ones((batch_size, 1))

    def forward(self, input1, input2):
        prot_experts = self.protein_expert_encoder(input1)
        drug_experts = self.drug_expert_encoder(input2)
        protein, protein_weights = self._fuse_experts(prot_experts, self.prot_gate)
        drug, drug_weights = self._fuse_experts(drug_experts, self.drug_gate)
        protein = self.protein_norm(protein)
        drug = self.drug_norm(drug)

        anchor = self.anchor_norm(self.protein_anchor_proj(protein))
        delta = self.delta_norm(self.drug_delta_proj(drug))
        if self.ablation == "protein_anchor":
            anchor = torch.zeros_like(anchor)

        cosine = F.cosine_similarity(anchor, delta, dim=-1).unsqueeze(-1)
        mean_abs_difference = torch.mean(torch.abs(anchor - delta), dim=-1, keepdim=True)
        mean_product = torch.mean(anchor * delta, dim=-1, keepdim=True)
        norm_ratio = torch.log(
            (delta.norm(dim=-1, keepdim=True) + 1e-6)
            / (anchor.norm(dim=-1, keepdim=True) + 1e-6)
        ).clamp(-8.0, 8.0)
        availability = self._availability(anchor.shape[0], anchor)
        gate_statistics = torch.cat(
            [cosine, mean_abs_difference, mean_product, norm_ratio, availability.clamp_max(1.0)],
            dim=-1,
        )
        drug_gate, drug_gate_logits = self.reliability_gate(
            gate_statistics, adaptive=self.ablation != "reliability_gate"
        )
        # Multiplication by zero does not sanitize NaN (NaN * 0 = NaN).  The
        # explicit branch also makes protein-only warm-up/dropout truly isolate
        # the drug path if an upstream diagnostic ever becomes non-finite.
        effective_gate = torch.where(
            availability > 0,
            drug_gate * availability,
            torch.zeros_like(drug_gate),
        )
        drug_contribution = effective_gate * delta
        fused_feature = self.fusion_norm(anchor + drug_contribution)
        prediction = self.regressor(fused_feature)
        protein_prediction = self.regressor(self.fusion_norm(anchor))
        contribution_energy = drug_contribution.pow(2).mean(dim=-1, keepdim=True)
        # sqrt'(0) is infinite.  Dropped samples have exactly zero contribution,
        # so computing the regularizer through sqrt(x)**2 produced 0*inf=NaN.
        contribution_rms = contribution_energy.add(1e-8).sqrt()

        aux = {
            "protein_expert_weights": protein_weights,
            "drug_expert_weights": drug_weights,
            "protein_prediction": protein_prediction,
            "drug_gate": drug_gate,
            "drug_gate_logits": drug_gate_logits,
            "effective_drug_gate": effective_gate,
            "drug_availability": availability,
            "drug_contribution_rms": contribution_rms,
            "drug_contribution_energy": contribution_energy,
            "gate_statistics": gate_statistics,
        }
        self.last_aux = aux
        return prediction, aux

    def loss_components(self, labels, aux=None):
        aux = self.last_aux if aux is None else aux
        if aux is None:
            raise RuntimeError("loss_components requires a preceding forward pass")
        target = labels.reshape(-1, 1)
        zero = target.new_zeros(())
        protein_loss = torch.mean((aux["protein_prediction"] - target).pow(2))
        if self.ablation == "protein_aux" or self.protein_aux_weight <= 0:
            protein_loss = zero
        contribution_penalty = torch.mean(aux["drug_contribution_energy"])
        if (
            self.ablation in {"drug_modulation", "contribution_penalty"}
            or not self.drug_enabled
            or self.contribution_penalty_weight <= 0
        ):
            contribution_penalty = zero
        weighted_protein = self.protein_aux_weight * protein_loss
        weighted_contribution = self.contribution_penalty_weight * contribution_penalty
        return {
            "protein_aux": weighted_protein,
            "contribution_penalty": weighted_contribution,
            "total": weighted_protein + weighted_contribution,
            "unweighted_protein_aux": protein_loss,
            "unweighted_contribution_penalty": contribution_penalty,
        }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    legacy.FullRegressionTransformer = FullRegressionTransformer
    legacy.main()


if __name__ == "__main__":
    main()
