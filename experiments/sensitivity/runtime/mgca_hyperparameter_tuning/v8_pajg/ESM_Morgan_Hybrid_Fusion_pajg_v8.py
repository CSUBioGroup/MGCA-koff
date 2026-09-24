# -*- coding: utf-8 -*-
"""MGCA-PAJG v8: protein-anchored drug and joint gated corrections.

v8 retains the stable four-view ESM2/Morgan encoders and the successful v7
protein anchor.  It adds a real, low-rank bidirectional joint-interaction branch
and error-aware supervision for two independent bounded gates.  It deliberately
does not restore the v6 output residual or final MoE.
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
MODEL_VARIANT = "mgca_pajg_v8_protein_anchored_error_aware_joint_gating"
MAX_GRAD_NORM = 5.0
ABLATIONS = {
    "no",
    "protein_anchor",
    "drug_correction",
    "joint_interaction",
    "adaptive_router",
    "gate_target",
    "protein_perturbation",
    "drug_dropout",
}


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("mgca_pajg_v8_legacy", LEGACY_SCRIPT)
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


class BoundedScalarGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, initial: float, cap: float):
        super().__init__()
        if not 0.0 < initial < cap <= 1.0:
            raise ValueError("require 0 < initial gate < gate cap <= 1")
        self.initial = float(initial)
        self.cap = float(cap)
        probability = initial / cap
        self.initial_logit = float(math.log(probability / (1.0 - probability)))
        width = max(hidden_dim // 16, 24)
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, self.initial_logit)

    def forward(self, statistics, adaptive=True):
        logits = (
            self.network(statistics)
            if adaptive
            else statistics.new_full((statistics.shape[0], 1), self.initial_logit)
        )
        return self.cap * torch.sigmoid(logits), logits


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


class BranchHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x):
        return self.network(x)


class FullRegressionTransformer(nn.Module):
    """Three-branch PAJG model with parameter-matched forward/loss ablations."""

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
        branch_aux_weight=0.05,
        gate_target_weight=0.02,
        consistency_weight=0.02,
        contribution_penalty_weight=0.001,
        drug_gate_init=0.20,
        drug_gate_cap=0.50,
        joint_gate_init=0.05,
        joint_gate_cap=0.25,
        branch_dropout=0.25,
        protein_expert_perturbation=0.15,
        gate_target_temperature=1.0,
    ):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"Unsupported v8 ablation {ablation!r}; choose from {sorted(ABLATIONS)}")
        for value in (protein_aux_weight, branch_aux_weight, gate_target_weight, consistency_weight, contribution_penalty_weight):
            if value < 0:
                raise ValueError("loss weights must be non-negative")
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError("branch_dropout must be in [0,1)")
        if not 0.0 <= protein_expert_perturbation < 1.0:
            raise ValueError("protein_expert_perturbation must be in [0,1)")
        if gate_target_temperature <= 0:
            raise ValueError("gate_target_temperature must be positive")

        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.protein_aux_weight = float(protein_aux_weight)
        self.branch_aux_weight = float(branch_aux_weight)
        self.gate_target_weight = float(gate_target_weight)
        self.consistency_weight = float(consistency_weight)
        self.contribution_penalty_weight = float(contribution_penalty_weight)
        self.drug_gate_init = float(drug_gate_init)
        self.drug_gate_cap = float(drug_gate_cap)
        self.joint_gate_init = float(joint_gate_init)
        self.joint_gate_cap = float(joint_gate_cap)
        self.branch_dropout = float(branch_dropout)
        self.protein_expert_perturbation = float(protein_expert_perturbation)
        self.gate_target_temperature = float(gate_target_temperature)
        self.corrections_enabled = True
        self.router_adaptive = True

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
        self.drug_router = BoundedScalarGate(5, hidden_dim, dropout, drug_gate_init, drug_gate_cap)
        self.joint_router = BoundedScalarGate(6, hidden_dim, dropout, joint_gate_init, joint_gate_cap)
        self.protein_head = BranchHead(hidden_dim, dropout)
        self.drug_candidate_head = BranchHead(hidden_dim, dropout)
        self.joint_candidate_head = BranchHead(hidden_dim, dropout)
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

    def set_corrections_enabled(self, enabled: bool):
        self.corrections_enabled = bool(enabled)

    def set_router_adaptive(self, enabled: bool):
        self.router_adaptive = bool(enabled)

    def _perturb_protein_experts(self, experts):
        active = (
            self.training
            and self.corrections_enabled
            and self.ablation != "protein_perturbation"
            and self.protein_expert_perturbation > 0
        )
        if not active:
            return experts, experts.new_zeros((experts.shape[0], 1))
        apply = torch.rand((experts.shape[0],), device=experts.device) < self.protein_expert_perturbation
        chosen = torch.randint(experts.shape[1], (experts.shape[0],), device=experts.device)
        mask = experts.new_ones((experts.shape[0], experts.shape[1], 1))
        rows = torch.arange(experts.shape[0], device=experts.device)[apply]
        mask[rows, chosen[apply], 0] = 0.0
        return experts * mask, apply.to(experts.dtype).unsqueeze(-1)

    @staticmethod
    def _statistics(anchor, correction, availability):
        cosine = F.cosine_similarity(anchor, correction, dim=-1).unsqueeze(-1)
        mean_abs = torch.mean(torch.abs(anchor - correction), dim=-1, keepdim=True)
        mean_product = torch.mean(anchor * correction, dim=-1, keepdim=True)
        norm_ratio = torch.log(
            (correction.norm(dim=-1, keepdim=True) + 1e-6)
            / (anchor.norm(dim=-1, keepdim=True) + 1e-6)
        ).clamp(-8.0, 8.0)
        return torch.cat([cosine, mean_abs, mean_product, norm_ratio, availability.clamp_max(1.0)], dim=-1)

    def _availability(self, batch_size, reference):
        if not self.corrections_enabled:
            return reference.new_zeros((batch_size, 1))
        if self.training and self.ablation != "drug_dropout" and self.branch_dropout > 0:
            keep = (torch.rand((batch_size, 1), device=reference.device) >= self.branch_dropout).to(reference.dtype)
            # Do not use inverted-dropout scaling here: the gates are explicit
            # physical contribution bounds and must never exceed their caps.
            return keep
        return reference.new_ones((batch_size, 1))

    def forward(self, input1, input2):
        clean_protein_experts = self.protein_expert_encoder(input1)
        protein_experts, perturbation_active = self._perturb_protein_experts(clean_protein_experts)
        drug_experts = self.drug_expert_encoder(input2)
        protein, protein_weights = self.prot_gate(protein_experts)
        clean_protein, _ = self.prot_gate(clean_protein_experts)
        drug, drug_weights = self.drug_gate(drug_experts)
        protein, clean_protein, drug = self.protein_norm(protein), self.protein_norm(clean_protein), self.drug_norm(drug)
        anchor = self.anchor_norm(self.protein_anchor_proj(protein))
        clean_anchor = self.anchor_norm(self.protein_anchor_proj(clean_protein))
        drug_delta = self.delta_norm(self.drug_delta_proj(drug))
        joint_delta, attention_p2d, attention_d2p, joint_confidence = self.joint_interaction(protein_experts, drug_experts)
        if self.ablation == "protein_anchor":
            anchor = torch.zeros_like(anchor)
            clean_anchor = torch.zeros_like(clean_anchor)

        availability = self._availability(anchor.shape[0], anchor)
        drug_availability = torch.zeros_like(availability) if self.ablation == "drug_correction" else availability
        joint_availability = torch.zeros_like(availability) if self.ablation == "joint_interaction" else availability
        adaptive = self.router_adaptive and self.ablation != "adaptive_router"
        drug_statistics = self._statistics(anchor, drug_delta, drug_availability)
        joint_statistics_base = self._statistics(anchor, joint_delta, joint_availability)
        joint_statistics = torch.cat([joint_statistics_base, joint_confidence], dim=-1)
        drug_gate, drug_gate_logits = self.drug_router(drug_statistics, adaptive=adaptive)
        joint_gate, joint_gate_logits = self.joint_router(joint_statistics, adaptive=adaptive)
        effective_drug_gate = torch.where(drug_availability > 0, drug_gate * drug_availability, torch.zeros_like(drug_gate))
        effective_joint_gate = torch.where(joint_availability > 0, joint_gate * joint_availability, torch.zeros_like(joint_gate))
        drug_contribution = effective_drug_gate * drug_delta
        joint_contribution = effective_joint_gate * joint_delta
        final_feature = self.fusion_norm(anchor + drug_contribution + joint_contribution)
        prediction = self.regressor(final_feature)

        protein_feature = self.fusion_norm(anchor)
        drug_candidate_delta = torch.zeros_like(drug_delta) if self.ablation == "drug_correction" else self.drug_gate_init * drug_delta
        joint_candidate_delta = torch.zeros_like(joint_delta) if self.ablation == "joint_interaction" else self.joint_gate_init * joint_delta
        drug_candidate_feature = self.fusion_norm(anchor + drug_candidate_delta)
        joint_candidate_feature = self.fusion_norm(anchor + drug_candidate_delta + joint_candidate_delta)
        protein_prediction = self.protein_head(protein_feature)
        drug_candidate_prediction = self.drug_candidate_head(drug_candidate_feature)
        joint_candidate_prediction = self.joint_candidate_head(joint_candidate_feature)
        protein_consistency_error = (anchor - clean_anchor.detach()).pow(2).mean(dim=-1, keepdim=True)
        drug_energy = drug_contribution.pow(2).mean(dim=-1, keepdim=True)
        joint_energy = joint_contribution.pow(2).mean(dim=-1, keepdim=True)

        aux = {
            "protein_expert_weights": protein_weights,
            "drug_expert_weights": drug_weights,
            "protein_prediction": protein_prediction,
            "drug_candidate_prediction": drug_candidate_prediction,
            "joint_candidate_prediction": joint_candidate_prediction,
            "protein_consistency_error": protein_consistency_error,
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
            "drug_gate_statistics": drug_statistics,
            "joint_gate_statistics": joint_statistics,
            "joint_attention_confidence": joint_confidence,
            "attention_p2d": attention_p2d,
            "attention_d2p": attention_d2p,
            "protein_perturbation_active": perturbation_active,
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
        drug_branch_loss = torch.mean((aux["drug_candidate_prediction"] - target).pow(2))
        joint_branch_loss = torch.mean((aux["joint_candidate_prediction"] - target).pow(2))
        active_branch_losses = []
        if self.ablation != "drug_correction":
            active_branch_losses.append(drug_branch_loss)
        if self.ablation != "joint_interaction":
            active_branch_losses.append(joint_branch_loss)
        branch_loss = torch.stack(active_branch_losses).mean() if active_branch_losses else zero
        if not self.corrections_enabled:
            branch_loss = zero

        perturbation_mask = aux["protein_perturbation_active"]
        consistency = (
            (aux["protein_consistency_error"] * perturbation_mask).sum()
            / perturbation_mask.sum().clamp_min(1.0)
        )
        if self.ablation == "protein_perturbation" or not self.training:
            consistency = zero

        gate_target_loss = zero
        if self.corrections_enabled and self.router_adaptive and self.ablation not in {"adaptive_router", "gate_target"}:
            with torch.no_grad():
                protein_error = (aux["protein_prediction"] - target).pow(2)
                drug_error = (aux["drug_candidate_prediction"] - target).pow(2)
                joint_error = (aux["joint_candidate_prediction"] - target).pow(2)
                drug_target = torch.sigmoid((protein_error - drug_error) / self.gate_target_temperature)
                joint_target = torch.sigmoid((torch.minimum(protein_error, drug_error) - joint_error) / self.gate_target_temperature)
            losses = []
            if self.ablation != "drug_correction":
                drug_probability = (aux["drug_gate"] / self.drug_gate_cap).clamp(1e-5, 1.0 - 1e-5)
                drug_mask = (aux["drug_availability"] > 0).to(target.dtype)
                drug_bce = F.binary_cross_entropy(drug_probability, drug_target, reduction="none")
                losses.append((drug_bce * drug_mask).sum() / drug_mask.sum().clamp_min(1.0))
            if self.ablation != "joint_interaction":
                joint_probability = (aux["joint_gate"] / self.joint_gate_cap).clamp(1e-5, 1.0 - 1e-5)
                joint_mask = (aux["joint_availability"] > 0).to(target.dtype)
                joint_bce = F.binary_cross_entropy(joint_probability, joint_target, reduction="none")
                losses.append((joint_bce * joint_mask).sum() / joint_mask.sum().clamp_min(1.0))
            if losses:
                gate_target_loss = torch.stack(losses).mean()

        contribution_penalty = torch.mean(
            aux["drug_contribution_energy"] + aux["joint_contribution_energy"]
        )
        if not self.corrections_enabled:
            contribution_penalty = zero
        weighted = {
            "protein_aux": self.protein_aux_weight * protein_loss,
            "branch_aux": self.branch_aux_weight * branch_loss,
            "gate_target": self.gate_target_weight * gate_target_loss,
            "consistency": self.consistency_weight * consistency,
            "contribution_penalty": self.contribution_penalty_weight * contribution_penalty,
        }
        return {
            **weighted,
            "total": sum(weighted.values(), zero),
            "unweighted_protein_aux": protein_loss,
            "unweighted_branch_aux": branch_loss,
            "unweighted_gate_target": gate_target_loss,
            "unweighted_consistency": consistency,
            "unweighted_contribution_penalty": contribution_penalty,
        }


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    legacy.FullRegressionTransformer = FullRegressionTransformer
    legacy.main()


if __name__ == "__main__":
    main()
