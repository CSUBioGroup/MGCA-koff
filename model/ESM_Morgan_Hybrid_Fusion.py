# -*- coding: utf-8 -*-
"""
ESM2+Morgan 药物冷启动五折交叉验证训练脚本
(Gated Expert Fusion + Expert-level Cross-Att + Final MoE)

结构概要：
1) Protein / Drug 多专家编码 -> prot_experts, drug_experts ∈ R^{B×4×H}
2) 分支A：GatedExpertFusion 得到 prot_fused, drug_fused ∈ R^{B×H}
3) 分支B：专家尺度双向 Cross-Att 得到 x1, x2 ∈ R^{B×H}，拼接→cross_fused ∈ R^{B×H}
4) 三路拼接 combined = [prot_fused; drug_fused; cross_fused] ∈ R^{B×3H}
5) MoEBlock(combined) → moe_out ∈ R^{B×H}
6) 最终 MLP 回归：moe_out → 标量 pkoff
"""

import os
import sys
import math
import time
import random
import argparse
import csv
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# 导入冷启动框架
try:
    sys.path.append(os.path.join(os.path.dirname(__file__),
        'models/moe_kinetic'))
    from cold_start_framework import (
        DrugColdStartStrategy, ProteinColdStartStrategy, DrugProteinPairColdStartStrategy,
        ColdStartCrossValidator, set_seed
    )
except ImportError:
    print("Warning: cold_start_framework not found. Using simple set_seed.")
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# =============================================================================
# 指标
# =============================================================================

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    try:
        pearson = pearsonr(y_true, y_pred)[0]
    except Exception:
        pearson = 0.0
    try:
        spearman = spearmanr(y_true, y_pred)[0]
    except Exception:
        spearman = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman
    }


# =============================================================================
# 特征提取
# =============================================================================

def get_fingerprint(r, mols, n_bits=2048, device="cpu", fingerprint_type="morgan"):
    if fingerprint_type not in {"morgan", "fcfp"}:
        raise ValueError(f"Unsupported fingerprint_type: {fingerprint_type}")

    atom_invariants_generator = (
        rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        if fingerprint_type == "fcfp"
        else None
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=r,
        fpSize=n_bits,
        atomInvariantsGenerator=atom_invariants_generator,
    )

    fps = []
    valid_mask = []
    for mol in mols:
        if mol is None:
            fps.append(np.zeros((n_bits,), dtype=np.uint8))
            valid_mask.append(0)
            continue
        try:
            bv = generator.GetFingerprint(mol)
            arr = np.zeros((n_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(bv, arr)
            fps.append(arr)
            valid_mask.append(1)
        except Exception:
            fps.append(np.zeros((n_bits,), dtype=np.uint8))
            valid_mask.append(0)
    fps = torch.from_numpy(np.stack(fps)).float().to(device)
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)
    return fps, valid_mask


# @torch.no_grad()
# def batch_extract_esm2(fasta_list, tokenizer, model, device, batch_size=8, mix_last_k=4):
#     model.eval()
#     feats = []
#     for i in range(0, len(fasta_list), batch_size):
#         batch = fasta_list[i:i+batch_size]
#         enc = tokenizer(batch, return_tensors="pt", padding=True,
#                         truncation=True, max_length=1024)
#         enc = {k: v.to(device) for k, v in enc.items()}
#         out = model(**enc, output_hidden_states=True)
#         # hs = out.hidden_states[-mix_last_k:]   # list[k] of [B,L,H]
#         hs = out.hidden_states[1, N//3, (N*2)//3, N]
#         hs = ou
#         layer_mean = torch.stack([h.mean(dim=1) for h in hs], dim=1)  # [B,4,H]
#         feats.append(layer_mean)
#     return torch.cat(feats, dim=0)  # [N,4,H]
# @torch.no_grad()
# def batch_extract_esm2(fasta_list, tokenizer, model, device, batch_size=8):
#     model.eval()
#     feats = []
    
#     # 1. 获取总层数 N (例如 33)
#     N = model.config.num_hidden_layers
    
#     # 2. 定义你的 "Wide-Span" 采样策略
#     # 直接使用 N 是安全的 (hidden_states 长度为 N+1)
#     target_layer_indices = [1, N//3, (N*2)//3, N]
    
#     print(f"Feature Extraction Layers: {target_layer_indices}") # 调试用

#     for i in range(0, len(fasta_list), batch_size):
#         batch = fasta_list[i:i+batch_size]
        
#         # 获取 attention_mask 用于后续去除 padding 噪音
#         enc = tokenizer(batch, return_tensors="pt", padding=True,
#                         truncation=True, max_length=1024)
#         enc = {k: v.to(device) for k, v in enc.items()}
        
#         out = model(**enc, output_hidden_states=True)
        
#         # 3. 提取指定层 + Mask Mean Pooling
#         batch_layers = []
#         for idx in target_layer_indices:
#             h = out.hidden_states[idx] # [B, L, H]
            
#             # === 关键修正开始 ===
#             # 将 padding 位置的特征置 0，并不计入分母
#             mask = enc['attention_mask'].unsqueeze(-1) # [B, L, 1]
            
#             # 分子：有效位置求和
#             sum_h = (h * mask).sum(dim=1) # [B, H]
            
#             # 分母：有效长度 (防止除以 0，加 1e-9)
#             lengths = mask.sum(dim=1) + 1e-9 # [B, 1]
            
#             mean_h = sum_h / lengths # [B, H]
#             # === 关键修正结束 ===
            
#             batch_layers.append(mean_h)
            
#         # 堆叠 -> [B, 4, H]
#         layer_stack = torch.stack(batch_layers, dim=1)
#         feats.append(layer_stack)
        
#     return torch.cat(feats, dim=0) # [Total_N, 4, H]
import torch
import numpy as np

ESM_WINDOW_LAYOUTS = ("even_span_v2", "legacy_anchors_v1")
DEFAULT_ESM_WINDOW_LAYOUT = "even_span_v2"


def build_esm_depth_windows(
    num_hidden_layers: int,
    window_size: int,
    window_layout: str = DEFAULT_ESM_WINDOW_LAYOUT,
):
    """Return four valid, auditable ESM2 layer windows.

    ``legacy_anchors_v1`` preserves the historical bottom/one-third/two-thirds/
    top starts. ``even_span_v2`` places four starts evenly across every legal
    start position. The latter keeps all four experts distinct for large
    windows such as 16 on ESM2-t36.
    """
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise ValueError(
            f"num_hidden_layers must be a positive integer, got: {num_hidden_layers}"
        )
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got: {window_size}")
    if window_size > num_hidden_layers:
        raise ValueError(
            f"window_size={window_size} exceeds ESM2 depth={num_hidden_layers}"
        )
    if window_layout not in ESM_WINDOW_LAYOUTS:
        raise ValueError(
            f"Unsupported ESM window layout {window_layout!r}; "
            f"choose from {ESM_WINDOW_LAYOUTS}"
        )

    depth = num_hidden_layers
    if window_layout == "legacy_anchors_v1":
        starts = [1, depth // 3, (depth * 2) // 3, depth - window_size + 1]
    else:
        # Valid starts are 1..(depth-window_size+1). Use deterministic
        # integer half-up rounding to distribute four starts over that span.
        span = depth - window_size
        starts = [1 + (expert_index * span + 1) // 3 for expert_index in range(4)]

    windows = [list(range(start, start + window_size)) for start in starts]
    if any(index < 1 or index > depth for window in windows for index in window):
        raise ValueError(
            f"window_size={window_size} is incompatible with depth={depth} "
            f"under layout={window_layout}: {windows}"
        )
    if len({tuple(window) for window in windows}) != 4:
        raise ValueError(
            f"layout={window_layout} cannot form four distinct windows for "
            f"depth={depth}, window_size={window_size}: {windows}"
        )
    return windows


@torch.no_grad()
def batch_extract_esm2(
    fasta_list,
    tokenizer,
    model,
    device,
    batch_size=8,
    window_size=2,
    window_layout=DEFAULT_ESM_WINDOW_LAYOUT,
):
    """
    Strided Window Averaging Strategy (Aligned with Integer Division):
    Extracts features by averaging a small window of adjacent layers at 4 distinct depths.
    
    Logic aligns with: [Bottom, 1/3 Depth, 2/3 Depth, Top]
    
    Args:
        window_size (int): Number of adjacent layers to average (default=2).
    """
    model.eval()
    feats = []
    
    # 1. 获取总层数 N (e.g., 33)
    N = model.config.num_hidden_layers
    
    # 2. 定义 4 个窗口的起始位置 (核心修改处)
    # 这里放弃了浮点数间隔计算，改用严谨的整除定位，确保与你手动指定的逻辑一致。
    
    # Window 1: 底部 (从 Layer 1 开始)
    # e.g., [1, 2]
    w1 = list(range(1, 1 + window_size))
    
    # Window 2: 1/3 处 (整除)
    # e.g., N=33 -> 11 -> [11, 12]
    start_2 = N // 3
    w2 = list(range(start_2, start_2 + window_size))
    
    # Window 3: 2/3 处 (整除)
    # e.g., N=33 -> 22 -> [22, 23]
    start_3 = (N * 2) // 3
    w3 = list(range(start_3, start_3 + window_size))
    
    # Window 4: 顶部 (以 N 结束)
    # e.g., N=33 -> [32, 33]
    w4 = list(range(N - window_size + 1, N + 1))
    
    # 最终的层索引组
    target_windows = build_esm_depth_windows(N, window_size, window_layout)
    
    # 打印日志确认 (只打印一次)
    print(f"👉 采用跨步窗口平均策略 (Window Size={window_size}):")
    labels = ["Bottom (Seq)", "Low-Mid (Struct)", "High-Mid (Pocket)", "Top (Semantic)"]
    for i, w in enumerate(target_windows):
        print(f"   Expert {i+1} [{labels[i]}]: Layers {w} (Avg of {len(w)})")

    # 校验：防止配置错误导致索引越界
    if any(idx > N for w in target_windows for idx in w):
        raise ValueError(f"Window size {window_size} is too large for model depth {N} with this spacing!")

    # 3. 批处理提取
    total_batches = math.ceil(len(fasta_list) / batch_size) if fasta_list else 0
    batch_indices = range(0, len(fasta_list), batch_size)
    if tqdm is not None:
        batch_indices = tqdm(
            batch_indices,
            total=total_batches,
            desc=f"Extracting ESM2 ({len(fasta_list)} seqs)",
            unit="batch",
        )
    else:
        print(f"Extracting ESM2: {len(fasta_list)} sequences in {total_batches} batches")

    for i in batch_indices:
        batch = fasta_list[i:i+batch_size]
        
        # Tokenization
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        # Mask 处理 (去除 Padding 噪音)
        mask = enc['attention_mask'].unsqueeze(-1).float() # [B, L, 1]
        lengths = mask.sum(dim=1) + 1e-9
        
        out = model(**enc, output_hidden_states=True)
        
        batch_experts = []
        
        # 遍历 4 个窗口
        for window_indices in target_windows:
            # 收集该窗口内的层向量
            window_vectors = []
            
            for layer_idx in window_indices:
                h = out.hidden_states[layer_idx] # [B, L, H]
                
                # Mask Mean Pooling (序列维度平均)
                # (h * mask).sum -> 仅计算非Padding区域
                layer_rep = (h * mask).sum(dim=1) / lengths # [B, H]
                window_vectors.append(layer_rep)
            
            # Stack -> [B, Window_Size, H]
            window_stack = torch.stack(window_vectors, dim=1)
            
            # Window Mean (深度维度平均) -> [B, H]
            expert_feat = window_stack.mean(dim=1)
            
            batch_experts.append(expert_feat)
            
        # 堆叠最终的 4 个 Experts -> [B, 4, H]
        expert_stack = torch.stack(batch_experts, dim=1)
        feats.append(expert_stack)
        
    return torch.cat(feats, dim=0)

def _esm_cache_path_for_window(
    cache_path: str,
    window_size: int,
    window_layout: str = DEFAULT_ESM_WINDOW_LAYOUT,
) -> str:
    """Return an idempotent cache path recording window size and layout."""
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got: {window_size}")
    if window_layout not in ESM_WINDOW_LAYOUTS:
        raise ValueError(
            f"Unsupported ESM window layout {window_layout!r}; "
            f"choose from {ESM_WINDOW_LAYOUTS}"
        )
    root, ext = os.path.splitext(cache_path)
    root = re.sub(r"__ws\d+(?:__wl[a-zA-Z0-9_-]+)?$", "", root)
    # Historical legacy caches used only __wsN. Every non-legacy layout
    # carries an explicit suffix so a changed layer layout can never reuse a
    # numerically compatible but scientifically different cache.
    layout_suffix = (
        "" if window_layout == "legacy_anchors_v1" else f"__wl{window_layout}"
    )
    return root + f"__ws{window_size}{layout_suffix}" + (ext or ".pt")


def get_default_esm_cache_path(
    csv_path: str,
    window_size: int = 2,
    window_layout: str = DEFAULT_ESM_WINDOW_LAYOUT,
) -> str:
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    return _esm_cache_path_for_window(
        os.path.join(csv_dir, f"{csv_stem}.pt"), window_size, window_layout
    )


def get_combined_esm_cache_path(
    csv_paths,
    window_size: int = 2,
    window_layout: str = DEFAULT_ESM_WINDOW_LAYOUT,
) -> str:
    abs_paths = [os.path.abspath(path) for path in csv_paths]
    dirs = [os.path.dirname(path) for path in abs_paths]
    cache_dir = dirs[0] if all(path_dir == dirs[0] for path_dir in dirs) else os.getcwd()
    stems = [os.path.splitext(os.path.basename(path))[0] for path in abs_paths]
    return _esm_cache_path_for_window(
        os.path.join(cache_dir, "__".join(stems) + ".pt"),
        window_size,
        window_layout,
    )


def _pick_column(fieldnames, target_name):
    lower_map = {name.lower(): name for name in fieldnames}
    return lower_map.get(target_name.lower())


def read_labeled_rows(csv_path: str):
    print(f"Loading data: {csv_path}")
    rows: List[Tuple[str, str, float]] = []
    with open(csv_path, mode='r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")

        fasta_col = _pick_column(reader.fieldnames, 'FASTA')
        smiles_col = _pick_column(reader.fieldnames, 'SMILES')
        label_col = _pick_column(reader.fieldnames, 'pkoff')
        if not fasta_col or not smiles_col or not label_col:
            raise ValueError(
                f"CSV must contain FASTA, SMILES/smiles, pkoff columns: {csv_path}; "
                f"actual columns: {reader.fieldnames}"
            )

        for row in reader:
            fasta = row[fasta_col].strip()
            smiles = row[smiles_col].strip()
            label = float(row[label_col])
            rows.append((fasta, smiles, label))
    return rows


def preprocess_rows(
    rows: List[Tuple[str, str, float]],
    esm2_path: str,
    device: torch.device,
    esm_cache=None,
    cache_label="dataset",
    fingerprint_type="morgan",
    window_size=2,
    window_layout=DEFAULT_ESM_WINDOW_LAYOUT,
):
    print(f"Preprocessing {cache_label}: {len(rows)} samples")

    smiles_list = [r[1] for r in rows]
    fasta_list  = [r[0] for r in rows]
    labels      = np.array([r[2] for r in rows], dtype=float)

    if esm_cache is None:
        raise ValueError("esm_cache must be provided to preprocess_rows")
    esm_cache = _esm_cache_path_for_window(esm_cache, window_size, window_layout)
    fasta_en = None
    if os.path.exists(esm_cache):
        print(f"Loading cached ESM2 features: {esm_cache}")
        cached = torch.load(esm_cache, map_location=device)
        if cached.shape[0] == len(fasta_list):
            fasta_en = cached
        else:
            print(
                f"Cache sample count mismatch: cache={cached.shape[0]}, "
                f"rows={len(fasta_list)}; extracting ESM2 again."
            )

    if fasta_en is None:
        print("Extracting ESM2 features. This can take a long time...")
        tokenizer = AutoTokenizer.from_pretrained(esm2_path)
        model     = AutoModelForMaskedLM.from_pretrained(esm2_path).to(device)
        fasta_en  = batch_extract_esm2(
            fasta_list,
            tokenizer,
            model,
            device,
            batch_size=int(os.environ.get('ESM_BATCH_SIZE', '1')),
            window_size=window_size,
            window_layout=window_layout,
        )
        os.makedirs(os.path.dirname(os.path.abspath(esm_cache)), exist_ok=True)
        temporary_cache = str(esm_cache) + '.tmp.' + str(os.getpid())
        torch.save(fasta_en.cpu(), temporary_cache)
        os.replace(temporary_cache, esm_cache)
        fasta_en = fasta_en.to(device)

    print(f"ESM2 feature shape: {fasta_en.shape}")  # [N,4,2560]

    fp_label = "FCFP" if fingerprint_type == "fcfp" else "Morgan"
    print(f"Extracting {fp_label} fingerprints...")
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    smiles_en_list = []
    for r in range(4):
        fp, _ = get_fingerprint(r, mols, device=device, fingerprint_type=fingerprint_type)
        smiles_en_list.append(fp.unsqueeze(1))   # [N,1,2048]
    smiles_en = torch.cat(smiles_en_list, dim=1)  # [N,4,2048]
    print(f"{fp_label} fingerprint shape: {smiles_en.shape}")

    y = torch.from_numpy(labels).float().to(device)
    return fasta_en, smiles_en, y, rows


def load_and_preprocess_data(
    csv_path: str,
    esm2_path: str,
    device: torch.device,
    esm_cache=None,
    fingerprint_type="morgan",
    window_size=2,
    window_layout=DEFAULT_ESM_WINDOW_LAYOUT,
):
    rows = read_labeled_rows(csv_path)
    if esm_cache is None:
        esm_cache = get_default_esm_cache_path(
            csv_path, window_size=window_size, window_layout=window_layout
        )
    return preprocess_rows(
        rows,
        esm2_path,
        device,
        esm_cache=esm_cache,
        cache_label=os.path.basename(csv_path),
        fingerprint_type=fingerprint_type,
        window_size=window_size,
        window_layout=window_layout,
    )


# =============================================================================
# 模型结构：Mask / MultiExpert / Gated Fusion / Cross-Att / MoE
# =============================================================================

class MaskGenerator(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.W = nn.Linear(proj_dim, 1)

    def forward(self, X):
        # X: [B,S,D]
        B, S, D = X.size()
        X_flat = X.view(-1, D)               # [B*S, D]
        logits = self.W(X_flat).view(B, S, 1)
        mask   = torch.softmax(logits, dim=1)
        return X * mask


class MultiExpertEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_experts=4):
        super().__init__()
        self.max_experts = max_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(max_experts)
        ])

    def forward(self, x):
        # x: [B,S,in_dim], S<=max_experts (这里 S=4)
        B, S, D = x.size()
        outs = []
        for i in range(S):
            outs.append(self.experts[i](x[:, i, :]))  # [B,H]
        return torch.stack(outs, dim=1)               # [B,S,H]


class GatedExpertFusion(nn.Module):
    """自适应门控融合 (expert 级 MoE，用于 Layer/Radius 权重学习)"""
    def __init__(self, num_experts, hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim  = hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(num_experts * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, experts_out):
        # experts_out: [B,N,H]
        B, N, H = experts_out.size()
        flat = experts_out.view(B, N * H)        # [B,N*H]
        weights = self.gate(flat)               # [B,N]
        weights_exp = weights.unsqueeze(-1)     # [B,N,1]
        fused = (experts_out * weights_exp).sum(dim=1)  # [B,H]
        return fused, weights                   # [B,H], [B,N]


class ExpertBiCrossAttention(nn.Module):
    """
    专家尺度的双向 Cross-Attention:
      prot_experts: [B,Np,H]
      drug_experts: [B,Nd,H]
      输出:
        x1: prot←drug, [B,H]
        x2: drug←prot, [B,H]
    """
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.2):
        super().__init__()
        self.attn_p = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_d = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, prot_experts, drug_experts):
        # attn_output: [B, L, H], attn_output_weights: [B, L, S]
        out_p, w_p2d = self.attn_p(query=prot_experts, key=drug_experts, value=drug_experts, need_weights=True)
        out_d, w_d2p = self.attn_d(query=drug_experts, key=prot_experts, value=prot_experts, need_weights=True)
        
        x1 = out_p.mean(dim=1)  # [B,H]
        x2 = out_d.mean(dim=1)  # [B,H]
        
        # 将 Attention Weights 一并返回用于可视化
        return x1, x2, w_p2d, w_d2p


# class ExpertBiCrossAttention(nn.Module):
#     """
#     专家尺度的双向 Cross-Attention（带 Residual + LayerNorm）:
#       prot_experts: [B,Np,H]
#       drug_experts: [B,Nd,H]
#       输出:
#         x1: prot←drug, [B,H]
#         x2: drug←prot, [B,H]
#     """
#     def __init__(self, hidden_dim=512, num_heads=8, dropout=0.2):
#         super().__init__()
#         self.attn_p = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.attn_d = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )

#         # ===== 新增：残差里的 LayerNorm + Dropout =====
#         self.norm_p = nn.LayerNorm(hidden_dim)   # 给 protein 路径用
#         self.norm_d = nn.LayerNorm(hidden_dim)   # 给 drug 路径用
#         self.dropout = nn.Dropout(dropout)
#     def forward(self, prot_experts, drug_experts):
#         """
#         prot_experts: [B, Np, H]
#         drug_experts: [B, Nd, H]
#         """
#         # attn_output: [B, L, H], attn_output_weights: [B, L, S]

#         # 注意力输出命名为 attn_p / attn_d，更清晰
#         attn_p, w_p2d = self.attn_p(
#             query=prot_experts,
#             key=drug_experts,
#             value=drug_experts,
#             need_weights=True
#         )  # attn_p: [B, Np, H], w_p2d: [B, Np, Nd]

#         attn_d, w_d2p = self.attn_d(
#             query=drug_experts,
#             key=prot_experts,
#             value=prot_experts,
#             need_weights=True
#         )  # attn_d: [B, Nd, H], w_d2p: [B, Nd, Np]

#         # ===== 关键修改：Residual + LayerNorm =====
#         # out_p = LN(prot_experts + Dropout(attn_p))
#         out_p = self.norm_p(prot_experts + self.dropout(attn_p))
#         # out_d = LN(drug_experts + Dropout(attn_d))
#         out_d = self.norm_d(drug_experts + self.dropout(attn_d))

#         # 然后再在 expert 维度上做 mean pooling
#         x1 = out_p.mean(dim=1)  # [B,H]，prot←drug
#         x2 = out_d.mean(dim=1)  # [B,H]，drug←prot

#         # 保持返回值不变，方便你后面做可视化
#         return x1, x2, w_p2d, w_d2p


class MoEBlock(nn.Module):
    """
    最后一层 MoE：
      输入: combined ∈ R^{B×(3H)}
      内部: num_moe_experts 个 MLP expert + gating
      输出: moe_out ∈ R^{B×H}
    """
    def __init__(self, in_dim, out_dim, num_experts=4, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: [B,in_dim]
        gate_w = self.gate(x)  # [B,num_experts]
        expert_outs = [expert(x) for expert in self.experts]  # list of [B,out_dim]
        expert_stack = torch.stack(expert_outs, dim=1)        # [B,num_experts,out_dim]
        gate_exp = gate_w.unsqueeze(-1)                       # [B,num_experts,1]
        out = (expert_stack * gate_exp).sum(dim=1)            # [B,out_dim]
        return out, gate_w


class FullRegressionTransformer(nn.Module):
    def __init__(self,
                 proj_dim1=2560,
                 proj_dim2=2048,
                 hidden_dim=512,
                 dropout=0.1,
                 nums_of_experts=4,
                 num_heads=8,
                 moe_num_experts=4,
                 ablation='no'):
        super().__init__()
        self.ablation = ablation
        self.hidden_dim = hidden_dim
        self.n_experts  = nums_of_experts

        # Mask
        self.mask1 = MaskGenerator(proj_dim=proj_dim1)
        self.mask2 = MaskGenerator(proj_dim=proj_dim2)

        # 多专家编码
        self.protein_expert_encoder = MultiExpertEncoder(
            in_dim=proj_dim1, hidden_dim=hidden_dim, max_experts=nums_of_experts
        )
        self.drug_expert_encoder = MultiExpertEncoder(
            in_dim=proj_dim2, hidden_dim=hidden_dim, max_experts=nums_of_experts
        )

        # Gated Fusion
        self.prot_gate = GatedExpertFusion(num_experts=nums_of_experts,
                                           hidden_dim=hidden_dim)
        self.drug_gate = GatedExpertFusion(num_experts=nums_of_experts,
                                           hidden_dim=hidden_dim)

        # 专家尺度 Cross-Att
        self.expert_cross_att = ExpertBiCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.cross_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== 新增：最终 MoE 层 (特征选择) =====
        self.moe = MoEBlock(
            in_dim=hidden_dim * 3 if ablation == 'no' else hidden_dim * 2,
            out_dim=hidden_dim,
            num_experts=moe_num_experts,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )
      
        # 最终回归 MLP（输入维度 H）
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Adam 优化器会对 LayerNorm 有一定影响
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, input1, input2):
        """
        input1: [B,4,2560]  ESM2 最后4层
        input2: [B,4,2048]  Morgan r=0..3
        """
        # 1) Mask
        # input1 = self.mask1(input1)
        # input2 = self.mask2(input2)

        # 2) 多专家编码
        prot_experts = self.protein_expert_encoder(input1)  # [B,4,H]
        drug_experts = self.drug_expert_encoder(input2)     # [B,4,H]

        # 3) 分支 A：Gated Fusion
        prot_fused, prot_weights = self.prot_gate(prot_experts)  # [B,H], [B,4]
        drug_fused, drug_weights = self.drug_gate(drug_experts)  # [B,H], [B,4]

        # 4) 分支 B：专家尺度 Cross-Att
        # 接收额外的 attention weights 用于可视化: w_p2d [B,4,4], w_d2p [B,4,4]
        x1, x2, w_p2d, w_d2p = self.expert_cross_att(prot_experts, drug_experts)  
        cross_feat  = torch.cat([x1, x2], dim=-1)                   # [B,2H]
        cross_fused = self.cross_proj(cross_feat)                   # [B,H]

        # 5) 三路拼接 + 最终 MoE
        combined = None
        if self.ablation == 'no':
            combined = torch.cat([prot_fused, drug_fused, cross_fused], dim=-1)  # [B,3H]
        elif self.ablation == 'drug':
            combined = torch.cat([prot_fused, cross_fused], dim=-1)  # [B,3H]
        elif self.ablation == 'target':
            combined = torch.cat([drug_fused, cross_fused], dim=-1)  # [B,3H]
        elif self.ablation == 'bicross':
            combined = torch.cat([prot_fused, drug_fused], dim=-1)  # [B,3H]

        moe_out, moe_weights = self.moe(combined)                            # [B,H], [B,num_moe]
        
        # 6) 最终回归
        out = self.regressor(moe_out)  # [B,1]

        # 返回所有权重用于分析: Gate权重 (w_p, w_d) 和 Attention权重 (w_p2d, w_d2p)
        return out, (prot_weights, drug_weights, w_p2d, w_d2p)


# =============================================================================
# 可视化：专家权重 & Attention Map
# =============================================================================

# def analyze_and_plot_weights(prot_weights_list, drug_weights_list, save_dir, fold_id):
#     all_prot_w = np.concatenate(prot_weights_list, axis=0)  # [N,4]
#     all_drug_w = np.concatenate(drug_weights_list, axis=0)  # [N,4]

#     avg_prot_w = all_prot_w.mean(axis=0)
#     avg_drug_w = all_drug_w.mean(axis=0)

#     prot_labels = ["ESM Layer -4", "ESM Layer -3", "ESM Layer -2", "ESM Layer -1"]
#     drug_labels = ["Morgan r=0", "Morgan r=1", "Morgan r=2", "Morgan r=3"]

#     plt.figure(figsize=(12,5))

#     plt.subplot(1,2,1)
#     sns.heatmap(avg_prot_w.reshape(-1,1), annot=True, cmap="Blues", fmt=".3f",
#                 yticklabels=prot_labels, xticklabels=["Weight"])
#     plt.title(f"Fold {fold_id}: Protein Layer Importance (Gating)")

#     plt.subplot(1,2,2)
#     sns.heatmap(avg_drug_w.reshape(-1,1), annot=True, cmap="Oranges", fmt=".3f",
#                 yticklabels=drug_labels, xticklabels=["Weight"])
#     plt.title(f"Fold {fold_id}: Drug Radius Importance (Gating)")

#     plt.tight_layout()
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_expert_weights.png"), dpi=300)
#     plt.close()

#     np.save(os.path.join(save_dir, f"fold_{fold_id}_prot_weights.npy"), all_prot_w)
#     np.save(os.path.join(save_dir, f"fold_{fold_id}_drug_weights.npy"), all_drug_w)

# def analyze_and_plot_cross_attention(p2d_weights_list, d2p_weights_list, save_dir, fold_id):
#     """
#     可视化 Cross-Attention 热力图:
#     p2d: Prot attends to Drug (Row: Prot Layers, Col: Drug Radii)
#     d2p: Drug attends to Prot (Row: Drug Radii, Col: Prot Layers)
#     """
#     # p2d_weights_list: List of [B, 4, 4] -> Concat to [N, 4, 4]
#     all_p2d = np.concatenate(p2d_weights_list, axis=0)
#     all_d2p = np.concatenate(d2p_weights_list, axis=0)
    
#     # 计算全局平均 Pattern
#     avg_p2d = all_p2d.mean(axis=0) # [4, 4]
#     avg_d2p = all_d2p.mean(axis=0) # [4, 4]

#     prot_labels = ["ESM L-4", "ESM L-3", "ESM L-2", "ESM L-1"]
#     drug_labels = ["Morgan r=0", "Morgan r=1", "Morgan r=2", "Morgan r=3"]

#     plt.figure(figsize=(14, 6))

#     # Plot 1: Prot Attention (rows) to Drug (cols)
#     plt.subplot(1, 2, 1)
#     sns.heatmap(avg_p2d, annot=True, fmt=".3f", cmap="viridis",
#                 xticklabels=drug_labels, yticklabels=prot_labels)
#     plt.title(f"Fold {fold_id}: Prot Attends Drug (P->D)")
#     plt.xlabel("Drug Experts (Key/Value)")
#     plt.ylabel("Protein Experts (Query)")

#     # Plot 2: Drug Attention (rows) to Prot (cols)
#     plt.subplot(1, 2, 2)
#     sns.heatmap(avg_d2p, annot=True, fmt=".3f", cmap="magma",
#                 xticklabels=prot_labels, yticklabels=drug_labels)
#     plt.title(f"Fold {fold_id}: Drug Attends Prot (D->P)")
#     plt.xlabel("Protein Experts (Key/Value)")
#     plt.ylabel("Drug Experts (Query)")

#     plt.tight_layout()
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_cross_attn_map.png"), dpi=300)
#     plt.close()
# def analyze_and_plot_weights(prot_weights_list, drug_weights_list, save_dir, fold_id):
#     all_prot_w = np.concatenate(prot_weights_list, axis=0)  # [N,4]
#     all_drug_w = np.concatenate(drug_weights_list, axis=0)  # [N,4]

#     avg_prot_w = all_prot_w.mean(axis=0)
#     avg_drug_w = all_drug_w.mean(axis=0)

#     prot_labels = ["ESM Layer -4", "ESM Layer -3", "ESM Layer -2", "ESM Layer -1"]
#     drug_labels = ["Morgan r=0", "Morgan r=1", "Morgan r=2", "Morgan r=3"]

#     plt.figure(figsize=(12,5))

#     plt.subplot(1,2,1)
#     ax1 = sns.heatmap(
#         avg_prot_w.reshape(-1,1),
#         annot=True,
#         cmap="Blues",
#         fmt=".3f",
#         yticklabels=prot_labels,
#         xticklabels=["Weight"],
#         annot_kws={"size": 13}  # ★ 调整数字标注字号
#     )
#     plt.title(f"Fold {fold_id}: Protein Layer Importance (Gating)", fontsize=14)
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)

#     plt.subplot(1,2,2)
#     ax2 = sns.heatmap(
#         avg_drug_w.reshape(-1,1),
#         annot=True,
#         cmap="Oranges",
#         fmt=".3f",
#         yticklabels=drug_labels,
#         xticklabels=["Weight"],
#         annot_kws={"size": 13}  # ★ 调整数字标注字号
#     )
#     plt.title(f"Fold {fold_id}: Drug Radius Importance (Gating)", fontsize=14)
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)

#     plt.tight_layout()
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_expert_weights.png"), dpi=300)
#     plt.close()

#     np.save(os.path.join(save_dir, f"fold_{fold_id}_prot_weights.npy"), all_prot_w)
#     np.save(os.path.join(save_dir, f"fold_{fold_id}_drug_weights.npy"), all_drug_w)


# def analyze_and_plot_cross_attention(p2d_weights_list, d2p_weights_list, save_dir, fold_id):
#     """
#     可视化 Cross-Attention 热力图:
#     p2d: Prot attends to Drug (Row: Prot Layers, Col: Drug Radii)
#     d2p: Drug attends to Prot (Row: Drug Radii, Col: Prot Layers)
#     """
#     # p2d_weights_list: List of [B, 4, 4] -> Concat to [N, 4, 4]
#     all_p2d = np.concatenate(p2d_weights_list, axis=0)
#     all_d2p = np.concatenate(d2p_weights_list, axis=0)
    
#     # 计算全局平均 Pattern
#     avg_p2d = all_p2d.mean(axis=0) # [4, 4]
#     avg_d2p = all_d2p.mean(axis=0) # [4, 4]

#     prot_labels = ["ESM L-4", "ESM L-3", "ESM L-2", "ESM L-1"]
#     drug_labels = ["Morgan r=0", "Morgan r=1", "Morgan r=2", "Morgan r=3"]

#     plt.figure(figsize=(14, 6))

#     # Plot 1: Prot Attention (rows) to Drug (cols)
#     plt.subplot(1, 2, 1)
#     ax1 = sns.heatmap(
#         avg_p2d,
#         annot=True,
#         fmt=".3f",
#         cmap="viridis",
#         xticklabels=drug_labels,
#         yticklabels=prot_labels,
#         annot_kws={"size": 13}  # ★ 调整数字标注字号
#     )
#     plt.title(f"Fold {fold_id}: Prot Attends Drug (P->D)", fontsize=14)
#     plt.xlabel("Drug Experts (Key/Value)", fontsize=12)
#     plt.ylabel("Protein Experts (Query)", fontsize=12)
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)

#     # Plot 2: Drug Attention (rows) to Prot (cols)
#     plt.subplot(1, 2, 2)
#     ax2 = sns.heatmap(
#         avg_d2p,
#         annot=True,
#         fmt=".3f",
#         cmap="magma",
#         xticklabels=prot_labels,
#         yticklabels=drug_labels,
#         annot_kws={"size": 13}  # ★ 调整数字标注字号
#     )
#     plt.title(f"Fold {fold_id}: Drug Attends Prot (D->P)", fontsize=14)
#     plt.xlabel("Protein Experts (Key/Value)", fontsize=12)
#     plt.ylabel("Drug Experts (Query)", fontsize=12)
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)

#     plt.tight_layout()
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_cross_attn_map.png"), dpi=300)
#     plt.close()
def analyze_and_plot_weights(prot_weights_list, drug_weights_list, save_dir, fold_id):
    all_prot_w = np.concatenate(prot_weights_list, axis=0)  # [N,4]
    all_drug_w = np.concatenate(drug_weights_list, axis=0)  # [N,4]

    avg_prot_w = all_prot_w.mean(axis=0)
    avg_drug_w = all_drug_w.mean(axis=0)

    prot_labels = ["ESM Layer -4", "ESM Layer -3", "ESM Layer -2", "ESM Layer -1"]
    drug_labels = ["Morgan r=0", "Morgan r=1", "Morgan r=2", "Morgan r=3"]

    # ★ 全局放大 seaborn 字体
    sns.set(font_scale=1.5)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        avg_prot_w.reshape(-1, 1),
        annot=True,
        cmap="Blues",
        fmt=".3f",
        yticklabels=prot_labels,
        xticklabels=["Weight"],
        annot_kws={"size": 18}  # ★ 数字标注字号：非常大
    )
    plt.title(f"Fold {fold_id}: Protein Layer Importance (Gating)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(1, 2, 2)
    sns.heatmap(
        avg_drug_w.reshape(-1, 1),
        annot=True,
        cmap="Oranges",
        fmt=".3f",
        yticklabels=drug_labels,
        xticklabels=["Weight"],
        annot_kws={"size": 18}  # ★ 数字标注字号：非常大
    )
    plt.title(f"Fold {fold_id}: Drug Radius Importance (Gating)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_expert_weights.png"), dpi=300)
    plt.close()

    np.save(os.path.join(save_dir, f"fold_{fold_id}_prot_weights.npy"), all_prot_w)
    np.save(os.path.join(save_dir, f"fold_{fold_id}_drug_weights.npy"), all_drug_w)


def analyze_and_plot_cross_attention(p2d_weights_list, d2p_weights_list, save_dir, fold_id):
    """
    可视化 Cross-Attention 热力图:
    p2d: Prot attends to Drug (Row: Prot Layers, Col: Drug Radii)
    d2p: Drug attends to Prot (Row: Drug Radii, Col: Prot Layers)
    """
    all_p2d = np.concatenate(p2d_weights_list, axis=0)
    all_d2p = np.concatenate(d2p_weights_list, axis=0)
    
    avg_p2d = all_p2d.mean(axis=0)  # [4, 4]
    avg_d2p = all_d2p.mean(axis=0)  # [4, 4]

    prot_labels = ["ESM L-4", "ESM L-3", "ESM L-2", "ESM L-1"]
    drug_labels = ["r=0", "r=1", "r=2", "r=3"]

    # ★ 同样放大 seaborn 字体
    sns.set(font_scale=1.5)

    plt.figure(figsize=(14, 6))

    # Prot Attention -> Drug
    plt.subplot(1, 2, 1)
    sns.heatmap(
        avg_p2d,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=drug_labels,
        yticklabels=prot_labels,
        annot_kws={"size": 18}  # ★ 数字标注字号
    )
    plt.title(f"Fold {fold_id}: Prot Attends Drug (P->D)", fontsize=20)
    plt.xlabel("Drug Experts (Key/Value)", fontsize=16)
    plt.ylabel("Protein Experts (Query)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Drug Attention -> Prot
    plt.subplot(1, 2, 2)
    sns.heatmap(
        avg_d2p,
        annot=True,
        fmt=".3f",
        cmap="magma",
        xticklabels=prot_labels,
        yticklabels=drug_labels,
        annot_kws={"size": 18}  # ★ 数字标注字号
    )
    plt.title(f"Fold {fold_id}: Drug Attends Prot (D->P)", fontsize=20)
    plt.xlabel("Protein Experts (Key/Value)", fontsize=16)
    plt.ylabel("Drug Experts (Query)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"fold_{fold_id}_cross_attn_map.png"), dpi=600)
    plt.close()


# =============================================================================
# 训练 & 主流程
# =============================================================================

class ESM2MorganDataset(Dataset):
    def __init__(self, smiles, fasta, labels):
        self.smiles = smiles
        self.fasta  = fasta
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.smiles[idx], self.fasta[idx], self.labels[idx]


def train_one_fold(
    model, train_loader, val_loader, device, epochs, lr, val_freq=10,
    weight_decay=1e-2, patience=0,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float('inf')
    best_state    = None
    evaluations_without_improvement = 0

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for smiles_batch, fasta_batch, labels_batch in train_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch  = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)

            optimizer.zero_grad()
            preds, _ = model(fasta_batch, smiles_batch) # 忽略辅助输出
            loss = criterion(preds.reshape(-1), labels_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if epoch % val_freq == 0 or epoch == epochs:
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for smiles_batch, fasta_batch, labels_batch in val_loader:
                    smiles_batch = smiles_batch.float().to(device)
                    fasta_batch  = fasta_batch.float().to(device)
                    labels_batch = labels_batch.float().to(device)
                    preds, _ = model(fasta_batch, smiles_batch)
                    val_preds.append(preds.reshape(-1).cpu().numpy())
                    val_trues.append(labels_batch.cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_trues = np.concatenate(val_trues)
            metrics   = compute_metrics(val_trues, val_preds)
            rmse      = metrics["rmse"]
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            print(f"Epoch {epoch}: Val RMSE={rmse:.6f}, Pearson={metrics['pearson']:.4f} (Best RMSE={best_val_rmse:.6f})")
            if patience > 0 and evaluations_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: no validation improvement "
                    f"for {evaluations_without_improvement} evaluations."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 最终评估
    model.eval()

    # Train metrics
    train_preds, train_trues = [], []
    with torch.no_grad():
        for smiles_batch, fasta_batch, labels_batch in train_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch  = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)
            preds, _ = model(fasta_batch, smiles_batch)
            train_preds.append(preds.reshape(-1).cpu().numpy())
            train_trues.append(labels_batch.cpu().numpy())
    train_metrics = compute_metrics(np.concatenate(train_trues), np.concatenate(train_preds))

    # Val metrics + 权重 + Attention Map
    val_preds, val_trues = [], []
    prot_weights_list, drug_weights_list = [], []
    # 新增：收集 Cross Attention 权重
    p2d_attn_list, d2p_attn_list = [], [] 

    with torch.no_grad():
        for smiles_batch, fasta_batch, labels_batch in val_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch  = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)
            
            # 解包：w_p, w_d 是 Gating 权重; w_p2d, w_d2p 是 Attention 权重
            preds, (w_p, w_d, w_p2d, w_d2p) = model(fasta_batch, smiles_batch)
            
            val_preds.append(preds.reshape(-1).cpu().numpy())
            val_trues.append(labels_batch.cpu().numpy())
            
            prot_weights_list.append(w_p.cpu().numpy())
            drug_weights_list.append(w_d.cpu().numpy())
            
            # 收集 attn 权重
            p2d_attn_list.append(w_p2d.cpu().numpy())
            d2p_attn_list.append(w_d2p.cpu().numpy())
            
    val_metrics = compute_metrics(np.concatenate(val_trues), np.concatenate(val_preds))

    # 返回增加了 attn 权重列表
    return train_metrics, val_metrics, np.concatenate(val_trues), np.concatenate(val_preds), \
           prot_weights_list, drug_weights_list, p2d_attn_list, d2p_attn_list


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', type=str, default='dataset/data.csv')
    parser.add_argument('--train_csv', type=str, default=None,
                        help='固定划分模式: 训练集 CSV，需包含 FASTA, SMILES, pkoff')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='固定划分模式: 验证集 CSV，需包含 FASTA, SMILES, pkoff')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='固定划分模式: 测试集 CSV，需包含 FASTA, SMILES, pkoff')
    parser.add_argument('--split_name', type=str, default=None,
                        help='固定划分模式的输出子目录名称，默认由 train_csv 文件名推断')
    parser.add_argument('--esm2_path', type=str, default='../pretrained_model/esm2_t36')
    parser.add_argument('--output_dir', type=str, default='outputs_esm2_morgan_gated_crossatt_moe')
    parser.add_argument('--fingerprint_type', type=str, default='morgan',
                        choices=['morgan', 'fcfp'],
                        help='Drug fingerprint type: morgan=ECFP-like Morgan, fcfp=feature-based Morgan')
    parser.add_argument('--window_size', type=int, default=2,
                        help='每个ESM2深度专家平均的相邻层数；缓存文件按__wsN区分')
    parser.add_argument(
        '--window_layout',
        type=str,
        default=DEFAULT_ESM_WINDOW_LAYOUT,
        choices=ESM_WINDOW_LAYOUTS,
        help='ESM2 depth-window placement strategy; even_span_v2 is the final default',
    )
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.17)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--patience', type=int, default=0,
                        help='验证评估连续多少次无提升后早停；0表示禁用')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cold_start_mode', type=str, default='drug',
                        choices=['drug', 'target', 'pair'])
    parser.add_argument('--ablation', type=str, default='no', choices=['no', 'drug', 'target', 'bicross'])
    parser.add_argument('--n_splits', type=int, default=5)
    
    parser.add_argument('--save_models', type=_parse_bool, default=True, help='是否保存模型权重及配置')

    
    return parser.parse_args()


def _infer_split_name(train_csv):
    stem = os.path.splitext(os.path.basename(train_csv))[0]
    if stem.startswith("train_"):
        return stem[len("train_"):]
    return stem


def run_fixed_split(args, device, total_start):
    split_name = args.split_name or _infer_split_name(args.train_csv)
    print(f"使用固定 train/val/test 划分: {split_name}")

    train_rows = read_labeled_rows(args.train_csv)
    val_rows = read_labeled_rows(args.val_csv)
    test_rows = read_labeled_rows(args.test_csv)
    n_train, n_val, n_test = len(train_rows), len(val_rows), len(test_rows)
    print(f"Fixed split sizes: train={n_train}, val={n_val}, test={n_test}")

    merged_rows = train_rows + val_rows + test_rows
    merged_cache = get_combined_esm_cache_path(
        [args.train_csv, args.val_csv, args.test_csv],
        window_size=args.window_size,
        window_layout=args.window_layout,
    )
    fasta_en, smiles_en, y, _ = preprocess_rows(
        merged_rows,
        args.esm2_path,
        device,
        esm_cache=merged_cache,
        cache_label=f"{split_name} train+val+test",
        fingerprint_type=args.fingerprint_type,
        window_size=args.window_size,
        window_layout=args.window_layout,
    )

    train_start = 0
    val_start = n_train
    test_start = n_train + n_val
    train_end = n_train
    val_end = n_train + n_val
    test_end = n_train + n_val + n_test

    train_fasta = fasta_en[train_start:train_end]
    train_smiles = smiles_en[train_start:train_end]
    train_y = y[train_start:train_end]
    val_fasta = fasta_en[val_start:val_end]
    val_smiles = smiles_en[val_start:val_end]
    val_y = y[val_start:val_end]
    test_fasta = fasta_en[test_start:test_end]
    test_smiles = smiles_en[test_start:test_end]
    test_y = y[test_start:test_end]

    if len(train_y) != n_train or len(val_y) != n_val or len(test_y) != n_test:
        raise RuntimeError("Fixed split slicing failed; split lengths do not match original CSV sizes.")

    train_ds = ESM2MorganDataset(train_smiles, train_fasta, train_y)
    val_ds = ESM2MorganDataset(val_smiles, val_fasta, val_y)
    test_ds = ESM2MorganDataset(test_smiles, test_fasta, test_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    fold_start = time.time()
    model = FullRegressionTransformer(
        proj_dim1=2560,
        proj_dim2=2048,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        nums_of_experts=4,
        num_heads=8,
        moe_num_experts=4,
        ablation=args.ablation
    ).to(device)

    train_met, val_met, _, _, _, _, _, _ = train_one_fold(
        model, train_loader, val_loader, device, args.epochs, args.lr,
        val_freq=args.val_freq, weight_decay=args.weight_decay, patience=args.patience,
    )

    model.eval()
    test_preds, test_trues = [], []
    prot_w_test, drug_w_test = [], []
    p2d_attn_test, d2p_attn_test = [], []

    with torch.no_grad():
        for smiles_batch, fasta_batch, labels_batch in test_loader:
            smiles_batch = smiles_batch.float().to(device)
            fasta_batch = fasta_batch.float().to(device)
            labels_batch = labels_batch.float().to(device)
            preds, (w_p, w_d, w_p2d, w_d2p) = model(fasta_batch, smiles_batch)

            test_preds.append(preds.reshape(-1).cpu().numpy())
            test_trues.append(labels_batch.cpu().numpy())
            prot_w_test.append(w_p.cpu().numpy())
            drug_w_test.append(w_d.cpu().numpy())
            p2d_attn_test.append(w_p2d.cpu().numpy())
            d2p_attn_test.append(w_d2p.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_trues = np.concatenate(test_trues)
    test_metrics = compute_metrics(test_trues, test_preds)
    fold_dur = time.time() - fold_start
    total_dur = time.time() - total_start

    print(f"{split_name} Results:")
    print(f"  Train - RMSE: {train_met['rmse']:.4f}, R2: {train_met['r2']:.4f}, PCC: {train_met['pearson']:.4f}")
    print(f"  Val   - RMSE: {val_met['rmse']:.4f}, R2: {val_met['r2']:.4f}, PCC: {val_met['pearson']:.4f}")
    print(f"  Test  - RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}, PCC: {test_metrics['pearson']:.4f}")

    analyze_and_plot_weights(prot_w_test, drug_w_test, args.output_dir, 1)
    analyze_and_plot_cross_attention(p2d_attn_test, d2p_attn_test, args.output_dir, 1)

    if args.save_models:
        save_name = f"model_{split_name}.pt"
        save_path = os.path.join(args.output_dir, save_name)
        checkpoint = {
            'fold': split_name,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'model_state_dict': model.state_dict(),
            'config': {
                'proj_dim1': 2560,
                'proj_dim2': 2048,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'nums_of_experts': 4,
                'num_heads': 8,
                'moe_num_experts': model.moe.num_experts,
                'ablation': args.ablation,
                'fingerprint_type': args.fingerprint_type,
                'window_size': args.window_size,
                'window_layout': args.window_layout,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'val_freq': args.val_freq,
                'patience': args.patience,
            },
            'metrics': {
                'test_rmse': test_metrics['rmse'],
                'test_r2': test_metrics['r2']
            }
        }
        torch.save(checkpoint, save_path)
        print(f"模型已保存: {save_path}")

    keys = ['mse', 'rmse', 'mae', 'r2', 'pearson', 'spearman']
    ts = time.strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(args.output_dir, f'metrics_{split_name}_{ts}.txt')

    with open(metrics_file, 'w') as f:
        f.write(f"ESM2+Morgan Gated+CrossAtt+MoE - fixed split {split_name}\n")
        f.write(f"ESM2 window size: {args.window_size}\n")
        f.write(f"ESM2 window layout: {args.window_layout}\n")
        layout_suffix = (
            "" if args.window_layout == "legacy_anchors_v1"
            else f"__wl{args.window_layout}"
        )
        f.write(f"ESM2 cache suffix: __ws{args.window_size}{layout_suffix}.pt\n")
        f.write(f"Train CSV: {args.train_csv}\n")
        f.write(f"Val CSV: {args.val_csv}\n")
        f.write(f"Test CSV: {args.test_csv}\n")
        f.write(f"Total Duration: {total_dur:.2f}s\n")
        f.write(f"Train Duration: {fold_dur:.2f}s\n\n")
        header = ["Split", "Duration"]
        for phase in ['train', 'val', 'test']:
            for k in keys:
                header.append(f"{phase}_{k}")
        f.write("\t".join(header) + "\n")
        row = [split_name, f"{fold_dur:.2f}"]
        for metrics in [train_met, val_met, test_metrics]:
            for k in keys:
                row.append(f"{metrics[k]:.6f}")
        f.write("\t".join(row) + "\n")

    pred_file = os.path.join(args.output_dir, f'test_predictions_{split_name}_{ts}.txt')
    np.savetxt(pred_file, np.vstack([test_trues, test_preds]).T,
               header='True\tPred', delimiter='\t')

    print(f"\n结果已保存至: {args.output_dir}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    fixed_split_mode = any([args.train_csv, args.val_csv, args.test_csv])
    if fixed_split_mode and not all([args.train_csv, args.val_csv, args.test_csv]):
        raise ValueError("固定划分模式需要同时提供 --train_csv, --val_csv, --test_csv")

    if args.ablation != 'no':
        args.output_dir = args.output_dir + "_ablation_" + args.ablation
    fingerprint_suffix = "_" + args.fingerprint_type.lower()
    if (args.fingerprint_type != 'morgan'
            and not args.output_dir.lower().endswith(fingerprint_suffix)):
        args.output_dir = args.output_dir + "_" + args.fingerprint_type

    if fixed_split_mode:
        split_name = args.split_name or _infer_split_name(args.train_csv)
        args.output_dir = os.path.join(args.output_dir, "fixed_split", split_name)
    else:
        args.output_dir = os.path.join(args.output_dir, args.cold_start_mode)

    os.makedirs(args.output_dir, exist_ok=True)
    total_start = time.time()

    if fixed_split_mode:
        run_fixed_split(args, device, total_start)
        return

    # 1. 加载数据
    fasta_en, smiles_en, y, rows = load_and_preprocess_data(
        args.dataset_csv, args.esm2_path, device,
        fingerprint_type=args.fingerprint_type,
        window_size=args.window_size,
        window_layout=args.window_layout,
    )

    # 2. 冷启动划分
    mode_names = {'drug': '药物', 'target': '蛋白', 'pair': '药物-蛋白对'}
    print(f"加载 {mode_names.get(args.cold_start_mode,'药物')} 冷启动划分...")
    try:
        from generate_folds import load_folds
        folds_path = f"{args.cold_start_mode}_cold/unified_folds_{args.cold_start_mode}.pkl"
        folds = load_folds(folds_path)
    except Exception:
        print("未找到划分文件，使用随机 KFold (仅调试用)")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        folds = list(kf.split(np.arange(len(y))))

    print(f"开始训练，共 {len(folds)} 折...")
    all_metrics = []
    all_preds   = np.zeros((len(y),), dtype=np.float32)
    all_trues   = np.zeros((len(y),), dtype=np.float32)
    
    # 定义容器，用于存储每一折的平均注意力矩阵
    fold_p2d_matrices = [] 
    fold_d2p_matrices = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_id+1} / {len(folds)} ===")
        fold_start = time.time()

        train_ds = ESM2MorganDataset(smiles_en[train_idx], fasta_en[train_idx], y[train_idx])
        test_ds  = ESM2MorganDataset(smiles_en[test_idx],  fasta_en[test_idx],  y[test_idx])

        train_sub_len = int(0.9 * len(train_ds))
        train_sub_ds, val_sub_ds = torch.utils.data.random_split(
            train_ds,
            [train_sub_len, len(train_ds) - train_sub_len],
            generator=torch.Generator().manual_seed(args.seed)
        )

        train_loader = DataLoader(train_sub_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_sub_ds,   batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds,      batch_size=args.batch_size, shuffle=False)

        model = FullRegressionTransformer(
            proj_dim1=2560,
            proj_dim2=2048,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            nums_of_experts=4,
            num_heads=8,
            moe_num_experts=4,
            ablation=args.ablation
        ).to(device)

        # 接收额外的 Attn List
        train_met, val_met, _, _, prot_w_val, drug_w_val, p2d_attn_val, d2p_attn_val = train_one_fold(
            model, train_loader, val_loader, device, args.epochs, args.lr,
            val_freq=args.val_freq, weight_decay=args.weight_decay, patience=args.patience,
        )

        # 测试集 (同样收集权重用于分析，可选)
        model.eval()
        test_preds, test_trues = [], []
        prot_w_test, drug_w_test = [], []
        p2d_attn_test, d2p_attn_test = [], []
        
        with torch.no_grad():
            for smiles_batch, fasta_batch, labels_batch in test_loader:
                smiles_batch = smiles_batch.float().to(device)
                fasta_batch  = fasta_batch.float().to(device)
                labels_batch = labels_batch.float().to(device)
                preds, (w_p, w_d, w_p2d, w_d2p) = model(fasta_batch, smiles_batch)
                
                test_preds.append(preds.reshape(-1).cpu().numpy())
                test_trues.append(labels_batch.cpu().numpy())
                prot_w_test.append(w_p.cpu().numpy())
                drug_w_test.append(w_d.cpu().numpy())
                p2d_attn_test.append(w_p2d.cpu().numpy())
                d2p_attn_test.append(w_d2p.cpu().numpy())

        test_preds = np.concatenate(test_preds)
        test_trues = np.concatenate(test_trues)
        test_metrics = compute_metrics(test_trues, test_preds)

        all_trues[test_idx] = test_trues
        all_preds[test_idx] = test_preds

        fold_dur = time.time() - fold_start

        print(f"Fold {fold_id+1} Results:")
        print(f"  Train - RMSE: {train_met['rmse']:.4f}, R2: {train_met['r2']:.4f}, PCC: {train_met['pearson']:.4f}")
        print(f"  Val   - RMSE: {val_met['rmse']:.4f}, R2: {val_met['r2']:.4f}, PCC: {val_met['pearson']:.4f}")
        print(f"  Test  - RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}, PCC: {test_metrics['pearson']:.4f}")

        record = {
            "fold": fold_id+1,
            "duration": fold_dur,
        }
        for phase, metrics in [('train', train_met), ('val', val_met), ('test', test_metrics)]:
            for k, v in metrics.items():
                record[f"{phase}_{k}"] = v
        all_metrics.append(record)

        # 1. 专家权重图（基于 gate fusion 的权重） - 使用测试集数据
        analyze_and_plot_weights(prot_w_test, drug_w_test, args.output_dir, fold_id+1)

        current_fold_p2d_mean = np.concatenate(p2d_attn_test, axis=0).mean(axis=0)
        current_fold_d2p_mean = np.concatenate(d2p_attn_test, axis=0).mean(axis=0)
        
        fold_p2d_matrices.append(current_fold_p2d_mean)
        fold_d2p_matrices.append(current_fold_d2p_mean)
        
        # 2. Cross-Attention 热力图 (新增功能) - 使用测试集数据
        analyze_and_plot_cross_attention(p2d_attn_test, d2p_attn_test, args.output_dir, fold_id+1)

        # === 新增：模型保存逻辑 ===
        if args.save_models:
            save_name = f"model_{args.cold_start_mode}_fold{fold_id+1}.pt"
            save_path = os.path.join(args.output_dir, save_name)
            
            checkpoint = {
                'fold': fold_id + 1,
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'model_state_dict': model.state_dict(),
                # 关键：保存配置以便推理时重建模型
                'config': {
                    'proj_dim1': 2560,       # ESM2-t36 固定维度
                    'proj_dim2': 2048,       # Morgan 指纹维度
                    'hidden_dim': args.hidden_dim,
                    'dropout': args.dropout,
                    'nums_of_experts': 4,    # 你在 main 中硬编码的值
                    'num_heads': 8,          # 你在 main 中硬编码的值
                    'moe_num_experts': model.moe.num_experts,
                    'fingerprint_type': args.fingerprint_type,
                    'window_size': args.window_size,
                    'window_layout': args.window_layout,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'val_freq': args.val_freq,
                    'patience': args.patience,
                },
                'metrics': {
                    'test_rmse': test_metrics['rmse'],
                    'test_r2': test_metrics['r2']
                }
            }
            
            torch.save(checkpoint, save_path)
            print(f"✅ 模型已保存: {save_path}")
            
    total_dur = time.time() - total_start
    overall   = compute_metrics(all_trues, all_preds)

    # =========================================================
    #  新增：绘制五折汇总的 Grand Average Heatmap
    # =========================================================
    print("绘制五折汇总热力图 (Grand Average)...")
    
    # 1. 计算五折的平均值 (Grand Mean)
    # stack 后形状: [5, 4, 4] -> mean -> [4, 4]
    grand_avg_p2d = np.stack(fold_p2d_matrices).mean(axis=0)
    grand_avg_d2p = np.stack(fold_d2p_matrices).mean(axis=0)
    
    # 2. 绘制并保存
    prot_labels = ["ESM L-4", "ESM L-3", "ESM L-2", "ESM L-1"]
    drug_labels = ["r=0", "r=1", "r=2", "r=3"]
    
    plt.figure(figsize=(14, 6))
    
    # P->D
    plt.subplot(1, 2, 1)
    sns.heatmap(grand_avg_p2d, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=drug_labels, yticklabels=prot_labels)
    plt.title(f"Grand Average (5-Fold): Prot Attends Drug")
    plt.xlabel("Drug Experts (Scales)")
    plt.ylabel("Protein Experts (Depths)")
    
    # D->P
    plt.subplot(1, 2, 2)
    sns.heatmap(grand_avg_d2p, annot=True, fmt=".3f", cmap="magma",
                xticklabels=prot_labels, yticklabels=drug_labels)
    plt.title(f"Grand Average (5-Fold): Drug Attends Prot")
    plt.xlabel("Protein Experts (Depths)")
    plt.ylabel("Drug Experts (Scales)")
    
    plt.tight_layout()
    path = os.path.join(args.output_dir, "grand_average_attention_map_" + args.cold_start_mode + "_cold.png")
    plt.savefig(path, dpi=600)
    plt.close()
    
    print(f"全局平均热力图已保存至: {path}")
    # =========================================================
    
    keys = ['mse','rmse','mae','r2','pearson','spearman']
    avg_metrics = {}
    for phase in ['train','val','test']:
        for k in keys:
            avg_metrics[f"avg_{phase}_{k}"] = np.mean([m[f"{phase}_{k}"] for m in all_metrics])

    ts = time.strftime('%Y%m%d_%H%M%S')
    metrics_file = os.path.join(args.output_dir,
                                f'metrics_{args.cold_start_mode}_{ts}.txt')

    print("\n" + "="*60)
    print("折间平均指标：")
    print(f"Train - RMSE: {avg_metrics['avg_train_rmse']:.4f}, R2: {avg_metrics['avg_train_r2']:.4f}, PCC: {avg_metrics['avg_train_pearson']:.4f}")
    print(f"Val   - RMSE: {avg_metrics['avg_val_rmse']:.4f}, R2: {avg_metrics['avg_val_r2']:.4f}, PCC: {avg_metrics['avg_val_pearson']:.4f}")
    print(f"Test  - RMSE: {avg_metrics['avg_test_rmse']:.4f}, R2: {avg_metrics['avg_test_r2']:.4f}, PCC: {avg_metrics['avg_test_pearson']:.4f}")
    print("-"*60)
    print("OOF 总体指标：")
    print(f"MSE: {overall['mse']:.4f}")
    print(f"RMSE: {overall['rmse']:.4f}")
    print(f"MAE: {overall['mae']:.4f}")
    print(f"R2: {overall['r2']:.4f}")
    print(f"Pearson: {overall['pearson']:.4f}")
    print(f"Spearman: {overall['spearman']:.4f}")
    print("="*60)

    with open(metrics_file, 'w') as f:
        f.write(f"ESM2+Morgan Gated+CrossAtt+MoE - {args.cold_start_mode} Cold Start\n")
        f.write(f"Total Duration: {total_dur:.2f}s\n\n")
        header = ["Fold","Duration"]
        for phase in ['train','val','test']:
            for k in keys:
                header.append(f"{phase}_{k}")
        f.write("\t".join(header) + "\n")
        for m in all_metrics:
            row = [str(m['fold']), f"{m['duration']:.2f}"]
            for phase in ['train','val','test']:
                for k in keys:
                    row.append(f"{m[f'{phase}_{k}']:.6f}")
            f.write("\t".join(row)+"\n")
        f.write("\nAverage Metrics:\n")
        for k,v in avg_metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\nOverall OOF Metrics:\n")
        for k,v in overall.items():
            f.write(f"{k}: {v:.6f}\n")

    pred_file = os.path.join(args.output_dir,
                             f'oof_predictions_{args.cold_start_mode}_{ts}.txt')
    np.savetxt(pred_file, np.vstack([all_trues, all_preds]).T,
               header='True\tPred', delimiter='\t')

    print(f"\n结果已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
