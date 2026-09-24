# -*- coding: utf-8 -*-
"""MGCA-CGRS v10: conservative global residual shrinkage.

v10 preserves the protein anchor, deterministic shared prediction head, and
real low-rank bidirectional joint interaction from v9. It replaces the failed
near-constant sample-wise routers with explicit learnable global shrinkage
scalars, removes protein perturbation and drug dropout, keeps joint-only
dropout, and directly supervises sequential branch utility.
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
SOURCE_V7_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "v7_padg" / "ESM_Morgan_Hybrid_Fusion_padg_v7.py"
SOURCE_V8_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "v8_pajg" / "ESM_Morgan_Hybrid_Fusion_pajg_v8.py"
SOURCE_V9_SCRIPT = PROJECT_ROOT / "mgca_hyperparameter_tuning" / "v9_scrg" / "ESM_Morgan_Hybrid_Fusion_scrg_v9.py"
MODEL_VARIANT = "mgca_cgrs_v10_conservative_global_residual_shrinkage"
MAX_GRAD_NORM = 5.0
ABLATIONS = {
    "no",
    "protein_anchor",
    "drug_correction",
    "joint_interaction",
    "learnable_shrinkage",
    "drug_utility_loss",
    "joint_utility_loss",
    "gate_prior",
    "joint_dropout",
}


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("mgca_cgrs_v10_legacy", LEGACY_SCRIPT)
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


class GlobalShrinkageGate(nn.Module):
    """One bounded scalar shared by every sample in a trained run."""

    def __init__(self, initial: float, cap: float):
        super().__init__()
        if not 0.0 < initial < cap <= 1.0:
            raise ValueError("require 0 < initial gate < gate cap <= 1")
        self.initial = float(initial)
        self.cap = float(cap)
        probability = initial / cap
        initial_logit = float(math.log(probability / (1.0 - probability)))
        self.logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self.register_buffer("initial_logit", torch.tensor(initial_logit, dtype=torch.float32))

    def forward(self, batch_size: int, reference, learnable=True):
        logit = self.logit if learnable else self.initial_logit
        gate = self.cap * torch.sigmoid(logit)
        return gate.expand(batch_size, 1).to(reference), logit.expand(batch_size, 1).to(reference)


class LowRankBidirectionalJointInteraction(nn.Module):
    """Four-by-four bidirectional expert attention in a low-rank subspace."""

    def __init__(self, hidden_dim: int, rank: int, dropout: float):
        super().__init__()
        if rank < 8:
            raise ValueError("joint rank must be at least 8")
        self.rank = rank
        self.protein_q = nn.Linear(hidden_dim, rank, bias=False)
        self.protein_k = nn.Linear(hidden_dim, rank, bias=False)
        self.protein_v = nn.Linear(hidden_dim, rank, bias=False)
        self.drug_q = nn.Linear(hidden_dim, rank, bias=False)
        self.drug_k = nn.Linear(hidden_dim, rank, bias=False)
        self.drug_v = nn.Linear(hidden_dim, rank, bias=False)
        self.output = nn.Sequential(
            nn.Linear(rank * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.output_norm = BiasFreeRMSNorm(hidden_dim)

    def forward(self, protein_tokens, drug_tokens):
        pq, pk, pv = self.protein_q(protein_tokens), self.protein_k(protein_tokens), self.protein_v(protein_tokens)
        dq, dk, dv = self.drug_q(drug_tokens), self.drug_k(drug_tokens), self.drug_v(drug_tokens)
        scale = math.sqrt(self.rank)
        score_p2d = torch.matmul(pq, dk.transpose(-1, -2)) / scale
        score_d2p = torch.matmul(dq, pk.transpose(-1, -2)) / scale
        attention_p2d = torch.softmax(score_p2d, dim=-1)
        attention_d2p = torch.softmax(score_d2p, dim=-1)
        protein_context = torch.matmul(attention_p2d, dv).mean(dim=1)
        drug_context = torch.matmul(attention_d2p, pv).mean(dim=1)
        joint_input = torch.cat(
            [
                protein_context,
                drug_context,
                protein_context * drug_context,
                torch.abs(protein_context - drug_context),
            ],
            dim=-1,
        )
        joint = self.output_norm(self.output(joint_input))
        entropy_p2d = -(attention_p2d.clamp_min(1e-8) * attention_p2d.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1, keepdim=True)
        entropy_d2p = -(attention_d2p.clamp_min(1e-8) * attention_d2p.clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1, keepdim=True)
        confidence = 1.0 - (entropy_p2d + entropy_d2p) / (2.0 * math.log(attention_p2d.shape[-1]))
        return joint, attention_p2d, attention_d2p, confidence.clamp(0.0, 1.0)


class SharedPredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        # Candidate errors supervise sequential correction utility, so repeated
        # calls differ only because their input features differ. Independent
        # head-dropout masks would reintroduce incomparable prediction noise.
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Identity(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.network(x)


class FullRegressionTransformer(nn.Module):
    """Protein anchor plus two globally shrunk residual corrections."""

    def __init__(
        self,
        proj_dim1=2560,
        proj_dim2=2048,
        hidden_dim=512,
        dropout=0.1,
        nums_of_experts=4,
        ablation="no",
        joint_rank=128,
        protein_aux_weight=0.10,
        drug_utility_weight=0.02,
        joint_utility_weight=0.02,
        gate_prior_weight=0.001,
        branch_margin=0.01,
        drug_gate_init=0.10,
        drug_gate_cap=0.30,
        joint_gate_init=0.03,
        joint_gate_cap=0.12,
        joint_branch_dropout=0.15,
    ):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"Unsupported v10 ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
        for value in (protein_aux_weight, drug_utility_weight, joint_utility_weight, gate_prior_weight, branch_margin):
            if value < 0:
                raise ValueError("loss weights and branch margin must be non-negative")
        if not 0.0 <= joint_branch_dropout < 1.0:
            raise ValueError("joint_branch_dropout must be in [0,1)")

        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.protein_aux_weight = float(protein_aux_weight)
        self.drug_utility_weight = float(drug_utility_weight)
        self.joint_utility_weight = float(joint_utility_weight)
        self.gate_prior_weight = float(gate_prior_weight)
        self.branch_margin = float(branch_margin)
        self.drug_gate_init = float(drug_gate_init)
        self.drug_gate_cap = float(drug_gate_cap)
        self.joint_gate_init = float(joint_gate_init)
        self.joint_gate_cap = float(joint_gate_cap)
        self.joint_branch_dropout = float(joint_branch_dropout)
        self.corrections_enabled = True
        self.shrinkage_learnable = True

        self.protein_expert_encoder = legacy.MultiExpertEncoder(proj_dim1, hidden_dim, nums_of_experts)
        self.drug_expert_encoder = legacy.MultiExpertEncoder(proj_dim2, hidden_dim, nums_of_experts)
        self.prot_gate = legacy.GatedExpertFusion(nums_of_experts, hidden_dim)
        self.drug_gate = legacy.GatedExpertFusion(nums_of_experts, hidden_dim)
        self.protein_norm = BiasFreeRMSNorm(hidden_dim)
        self.drug_norm = BiasFreeRMSNorm(hidden_dim)
        self.protein_anchor_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.drug_delta_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.anchor_norm = BiasFreeRMSNorm(hidden_dim)
        self.delta_norm = BiasFreeRMSNorm(hidden_dim)
        self.fusion_norm = BiasFreeRMSNorm(hidden_dim)
        self.joint_interaction = LowRankBidirectionalJointInteraction(hidden_dim, joint_rank, dropout)
        self.drug_shrinkage = GlobalShrinkageGate(drug_gate_init, drug_gate_cap)
        self.joint_shrinkage = GlobalShrinkageGate(joint_gate_init, joint_gate_cap)
        self.shared_prediction_head = SharedPredictionHead(hidden_dim, dropout)
        self.last_aux = None

    def set_corrections_enabled(self, enabled: bool):
        self.corrections_enabled = bool(enabled)

    def set_shrinkage_learnable(self, enabled: bool):
        self.shrinkage_learnable = bool(enabled)

    def _joint_availability(self, batch_size, reference):
        if not self.corrections_enabled or self.ablation == "joint_interaction":
            return reference.new_zeros((batch_size, 1))
        if self.training and self.ablation != "joint_dropout" and self.joint_branch_dropout > 0:
            return (
                torch.rand((batch_size, 1), device=reference.device)
                >= self.joint_branch_dropout
            ).to(reference.dtype)
        return reference.new_ones((batch_size, 1))

    def forward(self, input1, input2):
        protein_experts = self.protein_expert_encoder(input1)
        drug_experts = self.drug_expert_encoder(input2)
        protein, protein_weights = self.prot_gate(protein_experts)
        drug, drug_weights = self.drug_gate(drug_experts)
        protein = self.protein_norm(protein)
        drug = self.drug_norm(drug)
        anchor = self.anchor_norm(self.protein_anchor_proj(protein))
        drug_delta = self.delta_norm(self.drug_delta_proj(drug))
        joint_delta, attention_p2d, attention_d2p, joint_confidence = self.joint_interaction(
            protein_experts, drug_experts
        )
        if self.ablation == "protein_anchor":
            anchor = torch.zeros_like(anchor)

        batch_size = anchor.shape[0]
        drug_availability = (
            anchor.new_zeros((batch_size, 1))
            if not self.corrections_enabled or self.ablation == "drug_correction"
            else anchor.new_ones((batch_size, 1))
        )
        joint_availability = self._joint_availability(batch_size, anchor)
        learnable = self.shrinkage_learnable and self.ablation != "learnable_shrinkage"
        drug_gate, drug_gate_logits = self.drug_shrinkage(batch_size, anchor, learnable=learnable)
        joint_gate, joint_gate_logits = self.joint_shrinkage(batch_size, anchor, learnable=learnable)
        effective_drug_gate = drug_gate * drug_availability
        effective_joint_gate = joint_gate * joint_availability
        drug_contribution = effective_drug_gate * drug_delta
        joint_contribution = effective_joint_gate * joint_delta

        protein_feature = self.fusion_norm(anchor)
        drug_feature = self.fusion_norm(anchor + drug_contribution)
        joint_feature = self.fusion_norm(anchor + drug_contribution + joint_contribution)
        protein_prediction = self.shared_prediction_head(protein_feature)
        drug_prediction = self.shared_prediction_head(drug_feature)
        joint_prediction = self.shared_prediction_head(joint_feature)
        drug_energy = drug_contribution.pow(2).mean(dim=-1, keepdim=True)
        joint_energy = joint_contribution.pow(2).mean(dim=-1, keepdim=True)

        aux = {
            "protein_expert_weights": protein_weights,
            "drug_expert_weights": drug_weights,
            "protein_prediction": protein_prediction,
            "drug_candidate_prediction": drug_prediction,
            "joint_candidate_prediction": joint_prediction,
            "drug_prediction_shift": torch.abs(drug_prediction - protein_prediction).detach(),
            "joint_prediction_shift": torch.abs(joint_prediction - drug_prediction).detach(),
            "drug_gate": drug_gate,
            "joint_gate": joint_gate,
            "drug_gate_logits": drug_gate_logits,
            "joint_gate_logits": joint_gate_logits,
            "effective_drug_gate": effective_drug_gate,
            "effective_joint_gate": effective_joint_gate,
            "drug_availability": drug_availability,
            "joint_availability": joint_availability,
            "drug_contribution_energy": drug_energy,
            "joint_contribution_energy": joint_energy,
            "drug_contribution_rms": drug_energy.add(1e-8).sqrt(),
            "joint_contribution_rms": joint_energy.add(1e-8).sqrt(),
            "joint_attention_confidence": joint_confidence,
            "attention_p2d": attention_p2d,
            "attention_d2p": attention_d2p,
        }
        self.last_aux = aux
        return joint_prediction, aux

    @staticmethod
    def _masked_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def loss_components(self, labels, aux=None):
        aux = self.last_aux if aux is None else aux
        if aux is None:
            raise RuntimeError("loss_components requires a preceding forward pass")
        target = labels.reshape(-1, 1)
        zero = target.new_zeros(())
        protein_error = (aux["protein_prediction"] - target).pow(2)
        drug_error = (aux["drug_candidate_prediction"] - target).pow(2)
        joint_error = (aux["joint_candidate_prediction"] - target).pow(2)
        protein_loss = protein_error.mean()

        drug_utility = self._masked_mean(
            F.relu(drug_error - protein_error.detach() + self.branch_margin),
            aux["drug_availability"],
        )
        joint_utility = self._masked_mean(
            F.relu(joint_error - drug_error.detach() + self.branch_margin),
            aux["joint_availability"],
        )
        if (
            not self.corrections_enabled
            or self.ablation in {"drug_correction", "drug_utility_loss"}
        ):
            drug_utility = zero
        if (
            not self.corrections_enabled
            or self.ablation in {"joint_interaction", "joint_utility_loss"}
        ):
            joint_utility = zero

        gate_prior = zero
        if self.ablation != "drug_correction":
            gate_prior = gate_prior + (
                aux["drug_gate"][0, 0] / self.drug_gate_cap
            ).pow(2)
        if self.ablation != "joint_interaction":
            gate_prior = gate_prior + (
                aux["joint_gate"][0, 0] / self.joint_gate_cap
            ).pow(2)
        if not self.corrections_enabled or self.ablation == "gate_prior":
            gate_prior = zero
        weighted = {
            "protein_aux": self.protein_aux_weight * protein_loss,
            "drug_utility": self.drug_utility_weight * drug_utility,
            "joint_utility": self.joint_utility_weight * joint_utility,
            "gate_prior": self.gate_prior_weight * gate_prior,
        }
        return {
            **weighted,
            "total": sum(weighted.values(), zero),
            "unweighted_protein_aux": protein_loss,
            "unweighted_drug_utility": drug_utility,
            "unweighted_joint_utility": joint_utility,
            "unweighted_gate_prior": gate_prior,
        }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    legacy.FullRegressionTransformer = FullRegressionTransformer
    legacy.main()


if __name__ == "__main__":
    main()
