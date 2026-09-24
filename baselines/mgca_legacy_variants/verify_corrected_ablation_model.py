#!/usr/bin/env python3
"""Fail-fast structural checks for the corrected MGCA ablation model."""

from __future__ import annotations

import gc

import torch

from ESM_Morgan_Hybrid_Fusion_nonredundant import FullRegressionTransformer


def main() -> None:
    expected_moe_input = 512 * 3
    counts: dict[str, int] = {}
    protein = torch.randn(2, 4, 2560)
    drug = torch.randn(2, 4, 2048)

    for ablation in ("no", "drug", "target", "bicross"):
        model = FullRegressionTransformer(ablation=ablation)
        # Exercise nn.Module._apply through .to(); this catches accidental
        # collisions with PyTorch's internal device/dtype migration hook.
        model = model.to("cpu")
        counts[ablation] = sum(parameter.numel() for parameter in model.parameters())
        actual_moe_input = model.moe.gate[0].in_features
        if actual_moe_input != expected_moe_input:
            raise RuntimeError(
                f"{ablation}: final MoE input is {actual_moe_input}, "
                f"expected fixed 3H={expected_moe_input}"
            )
        if model.moe.num_experts != 2:
            raise RuntimeError(
                f"{ablation}: final MoE has {model.moe.num_experts} experts, expected 2"
            )
        if model.moe.experts[0][0].out_features != 512:
            raise RuntimeError(
                f"{ablation}: final MoE hidden width is not compact H=512"
            )
        model.eval()
        with torch.no_grad():
            prediction, (_, _, w_p2d, w_d2p) = model(protein, drug)
        if prediction.shape != (2, 1):
            raise RuntimeError(f"{ablation}: unexpected prediction shape {prediction.shape}")
        if w_p2d.shape != (2, 4, 4) or w_d2p.shape != (2, 4, 4):
            raise RuntimeError(
                f"{ablation}: unexpected MHA map shapes {w_p2d.shape}, {w_d2p.shape}"
            )
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"{ablation}: non-finite prediction in smoke test")
        del model
        gc.collect()

    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Ablation parameter counts differ: {counts}")

    print("Corrected MGCA ablation preflight OK")
    print(f"  fixed final MoE input: {expected_moe_input} (3H)")
    print(f"  identical trainable parameters: {next(iter(counts.values())):,}")
    print("  final MoE: 2 experts, hidden width H=512")
    print(f"  variants: {counts}")


if __name__ == "__main__":
    main()
