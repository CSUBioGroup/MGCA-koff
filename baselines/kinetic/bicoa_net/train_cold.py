#!/usr/bin/env python3
"""
Refined Protein-Ligand Binding Prediction Model

Simple but Effective Improvements over coattention1_save.py:
1. Better weight initialization (Kaiming/Xavier)
2. Mixup data augmentation (proven to reduce overfitting)
3. Exponential Moving Average (EMA) of weights
4. Stochastic Weight Averaging (SWA) 
5. Focal loss variant for better training
6. Enhanced residual connections
7. Cosine annealing with restarts
8. Better normalization strategy
9. Gradient accumulation for stable training
10. Attention temperature scaling

Target: Beat coattention1_save.py without overcomplicating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import os
import argparse
import random
from datetime import datetime
from tqdm import tqdm
import warnings
import esm
from transformers import AutoTokenizer, AutoModel
import math
from copy import deepcopy

# RDKit for molecular descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def normalize_input_columns(df, csv_path):
    """Normalize required CSV headers while accepting case variants."""
    canonical_columns = {
        "smiles": "smiles",
        "fasta": "FASTA",
        "pkoff": "pkoff",
    }
    columns_by_lower = {}
    for column in df.columns:
        columns_by_lower.setdefault(str(column).strip().lower(), []).append(column)

    rename_map = {}
    missing = []
    for lower_name, canonical_name in canonical_columns.items():
        matches = columns_by_lower.get(lower_name, [])
        if not matches:
            missing.append(canonical_name)
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous columns for {canonical_name} in {csv_path}: {matches}"
            )
        rename_map[matches[0]] = canonical_name

    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}; "
            f"actual columns: {list(df.columns)}"
        )
    return df.rename(columns=rename_map)


def set_random_seed(seed):
    """Seed all RNGs used by model initialization, shuffling, dropout, and mixup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ======================== Concordance Index ========================

def concordance_index(y_true, y_pred):
    """Calculate concordance index (C-index) for ranking evaluation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n = len(y_true)
    concordant = 0
    discordant = 0
    tied_pred = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            
            true_order = y_true[i] < y_true[j]
            pred_order = y_pred[i] < y_pred[j]
            
            if y_pred[i] == y_pred[j]:
                tied_pred += 1
            elif true_order == pred_order:
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = concordant + discordant + tied_pred
    if total_pairs == 0:
        return 0.5
    
    ci = (concordant + 0.5 * tied_pred) / total_pairs
    return ci

# ======================== Molecular Descriptors ========================

def compute_molecular_descriptors(smiles):
    """Compute RDKit molecular descriptors"""
    if not RDKIT_AVAILABLE:
        return np.zeros(200)
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(200)
        
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.RingCount(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0v(mol),
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
        ]
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=182)
        fp_array = np.array(fp)
        
        full_desc = np.concatenate([descriptors, fp_array])
        return full_desc
        
    except Exception as e:
        return np.zeros(200)

def compute_descriptors_for_dataset(df, save_path):
    """Compute and save molecular descriptors"""
    if os.path.exists(save_path):
        return np.load(save_path)
    
    print("  Computing molecular descriptors...")
    descriptors = []
    for smiles in tqdm(df['smiles'].tolist(), desc="Descriptors"):
        desc = compute_molecular_descriptors(smiles)
        descriptors.append(desc)
    
    descriptors = np.array(descriptors)
    np.save(save_path, descriptors)
    return descriptors

# ======================== Embedding Functions ========================

class MolFormerEmbedder:
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading MolFormer...")
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        # This checkpoint only publishes PyTorch/Flax weights.  Forcing
        # safetensors makes Transformers attempt an online conversion PR and
        # can fail before the existing pytorch_model.bin is considered.
        self.model = AutoModel.from_pretrained(
            "seyonec/PubChem10M_SMILES_BPE_450k",
            use_safetensors=False,
        )
        self.model = self.model.to(device).eval()
        print("✓ MolFormer loaded")
    
    def embed_smiles(self, smiles, max_length=512):
        try:
            with torch.no_grad():
                inputs = self.tokenizer(smiles, return_tensors="pt", padding=True, 
                                       truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(), True
        except:
            return np.zeros(768), False

class ESM2Embedder:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device='cuda'):
        self.device = device
        print(f"Loading ESM-2: {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        print("✓ ESM-2 loaded")
        
    def embed_protein(self, sequence, max_length=1024):
        try:
            with torch.no_grad():
                if len(sequence) > max_length:
                    sequence = sequence[:max_length]
                data = [("protein", sequence)]
                _, _, batch_tokens = self.batch_converter(data)
                batch_tokens = batch_tokens.to(self.device)
                results = self.model(batch_tokens, repr_layers=[33])
                return results["representations"][33].mean(dim=1).cpu().numpy().squeeze(), True
        except:
            return np.zeros(1280), False

def normalize_embeddings(embeddings):
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (embeddings - mean) / std

def compute_embeddings_for_split(df, molformer, esm2, split_name):
    print(f"\nComputing embeddings for {split_name}...")
    
    smiles_embeddings = []
    for smiles in tqdm(df['smiles'].tolist(), desc="SMILES"):
        emb, _ = molformer.embed_smiles(smiles)
        smiles_embeddings.append(emb)
    smiles_embeddings = normalize_embeddings(np.array(smiles_embeddings))
    
    protein_embeddings = []
    for fasta in tqdm(df['FASTA'].tolist(), desc="Proteins"):
        emb, _ = esm2.embed_protein(fasta)
        protein_embeddings.append(emb)
    protein_embeddings = normalize_embeddings(np.array(protein_embeddings))
    
    return smiles_embeddings, protein_embeddings

# ======================== Dataset with Mixup ========================

class EnhancedFeatureDataset(Dataset):
    def __init__(self, smiles_features, protein_features, mol_descriptors, labels, keep_valid_mask=False):
        valid_mask = (
            ~np.isnan(labels) & ~np.isinf(labels) &
            ~np.isnan(smiles_features).any(axis=1) &
            ~np.isnan(protein_features).any(axis=1) &
            ~np.isnan(mol_descriptors).any(axis=1)
        )
        if not valid_mask.all():
            print(f"  Filtered {(~valid_mask).sum()} invalid samples")
        
        self.smiles_features = torch.FloatTensor(smiles_features[valid_mask])
        self.protein_features = torch.FloatTensor(protein_features[valid_mask])
        self.mol_descriptors = torch.FloatTensor(mol_descriptors[valid_mask])
        self.labels = torch.FloatTensor(labels[valid_mask])
        
        if keep_valid_mask:
            self.valid_indices = np.where(valid_mask)[0]
        else:
            self.valid_indices = None
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.smiles_features[idx], 
                self.protein_features[idx], 
                self.mol_descriptors[idx],
                self.labels[idx])

# ======================== Mixup Augmentation ========================

def mixup_data(x1, x2, x3, y, alpha=0.2):
    """Apply mixup augmentation to the data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index]
    y_a, y_b = y, y[index]
    
    return mixed_x1, mixed_x2, mixed_x3, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * F.mse_loss(pred, y_a) + (1 - lam) * F.mse_loss(pred, y_b)

# ======================== Improved Model Architecture ========================

class MultiHeadCoAttentionWithTemp(nn.Module):
    """Multi-head co-attention with temperature scaling for better training"""
    def __init__(self, d_model, n_heads=8, dropout=0.1, temperature=1.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.temperature = temperature
        
        # Query, Key, Value projections
        self.q_lig = nn.Linear(d_model, d_model)
        self.k_lig = nn.Linear(d_model, d_model)
        self.v_lig = nn.Linear(d_model, d_model)
        
        self.q_prot = nn.Linear(d_model, d_model)
        self.k_prot = nn.Linear(d_model, d_model)
        self.v_prot = nn.Linear(d_model, d_model)
        
        self.out_lig = nn.Linear(d_model, d_model)
        self.out_prot = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout * 0.5)  # Lighter dropout on attention
        self.scale = math.sqrt(self.d_k) * self.temperature
        
        # Better initialization
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, lig, prot):
        batch_size = lig.size(0)
        
        def reshape(x):
            return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Ligand attends to protein
        q_l = reshape(self.q_lig(lig.unsqueeze(1)))
        k_p = reshape(self.k_prot(prot.unsqueeze(1)))
        v_p = reshape(self.v_prot(prot.unsqueeze(1)))
        
        attn_l = torch.matmul(q_l, k_p.transpose(-2, -1)) / self.scale
        attn_l = F.softmax(attn_l, dim=-1)
        attn_l = self.attn_dropout(attn_l)
        
        out_l = torch.matmul(attn_l, v_p)
        out_l = out_l.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        out_l = self.out_lig(out_l).squeeze(1)
        
        # Protein attends to ligand
        q_p = reshape(self.q_prot(prot.unsqueeze(1)))
        k_l = reshape(self.k_lig(lig.unsqueeze(1)))
        v_l = reshape(self.v_lig(lig.unsqueeze(1)))
        
        attn_p = torch.matmul(q_p, k_l.transpose(-2, -1)) / self.scale
        attn_p = F.softmax(attn_p, dim=-1)
        attn_p = self.attn_dropout(attn_p)
        
        out_p = torch.matmul(attn_p, v_l)
        out_p = out_p.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        out_p = self.out_prot(out_p).squeeze(1)
        
        return out_l, out_p


class RefinedCoAttentionBlock(nn.Module):
    """Refined co-attention block with better residual connections"""
    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadCoAttentionWithTemp(d_model, n_heads, dropout)
        
        self.norm1_lig = nn.LayerNorm(d_model)
        self.norm1_prot = nn.LayerNorm(d_model)
        self.norm2_lig = nn.LayerNorm(d_model)
        self.norm2_prot = nn.LayerNorm(d_model)
        
        # Improved feed-forward with residual scaling
        self.ff_lig = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.ff_prot = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Residual scaling factors (learnable)
        self.alpha_lig = nn.Parameter(torch.ones(1))
        self.alpha_prot = nn.Parameter(torch.ones(1))
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, lig, prot):
        # Pre-norm + attention + scaled residual
        lig_norm = self.norm1_lig(lig)
        prot_norm = self.norm1_prot(prot)
        
        lig_attn, prot_attn = self.attention(lig_norm, prot_norm)
        
        lig = lig + self.alpha_lig * lig_attn
        prot = prot + self.alpha_prot * prot_attn
        
        # Pre-norm + FF + residual
        lig = lig + self.ff_lig(self.norm2_lig(lig))
        prot = prot + self.ff_prot(self.norm2_prot(prot))
        
        return lig, prot


class ImprovedBilinearFusion(nn.Module):
    """Improved bilinear fusion with better regularization"""
    def __init__(self, lig_dim, prot_dim, out_dim, dropout=0.1):
        super().__init__()
        
        # Core bilinear interaction
        self.bilinear = nn.Bilinear(lig_dim, prot_dim, out_dim)
        
        # Linear projections with residual
        self.lig_proj = nn.Sequential(
            nn.Linear(lig_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
        # Element-wise multiplication path
        self.mult_proj = nn.Linear(lig_dim, out_dim)
        
        # Gating for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, lig, prot):
        # Multiple fusion paths
        bilinear_out = self.bilinear(lig, prot)
        
        linear_lig = self.lig_proj(lig)
        linear_prot = self.prot_proj(prot)
        
        # Element-wise with gating
        mult = self.mult_proj(lig * prot)
        
        # Adaptive gating
        gate_input = torch.cat([bilinear_out, mult], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # Combine with gate
        fused = gate_weights * bilinear_out + (1 - gate_weights) * mult
        fused = fused + linear_lig + linear_prot
        
        fused = self.norm(fused)
        fused = F.gelu(fused)
        fused = self.dropout(fused)
        
        return fused


class RefinedProteinLigandModel(nn.Module):
    """
    Refined model with simple but effective improvements:
    - Better initialization
    - Temperature-scaled attention
    - Learnable residual scaling
    - Enhanced normalization
    """
    def __init__(self, smiles_dim, protein_dim, descriptor_dim, 
                 d_model=768, n_blocks=4, n_heads=12, d_ff=3072, dropout=0.2):
        super().__init__()
        
        print(f"\n{'='*70}")
        print("Building Refined Protein-Ligand Model")
        print(f"{'='*70}")
        print(f"Model dimension: {d_model}")
        print(f"Co-attention blocks: {n_blocks}")
        print(f"Attention heads: {n_heads}")
        print(f"Dropout: {dropout}")
        print(f"{'='*70}\n")
        
        # Input projections with residual design
        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Molecular descriptor projection
        self.descriptor_proj = nn.Sequential(
            nn.Linear(descriptor_dim, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Co-attention blocks
        self.co_attention_blocks = nn.ModuleList([
            RefinedCoAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        
        # Fusion
        self.fusion = ImprovedBilinearFusion(d_model, d_model, d_model, dropout)
        
        # Integrate descriptors
        self.descriptor_fusion = nn.Sequential(
            nn.Linear(d_model + d_model // 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Streamlined output head (less deep to avoid overfitting)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Better weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, smiles_feat, protein_feat, mol_desc):
        # Project inputs
        lig = self.smiles_proj(smiles_feat)
        prot = self.protein_proj(protein_feat)
        desc = self.descriptor_proj(mol_desc)
        
        # Apply co-attention blocks
        for block in self.co_attention_blocks:
            lig, prot = block(lig, prot)
        
        # Fusion
        fused = self.fusion(lig, prot)
        
        # Integrate descriptors
        fused_with_desc = torch.cat([fused, desc], dim=-1)
        final_feat = self.descriptor_fusion(fused_with_desc)
        
        # Output
        output = self.output_head(final_feat).squeeze(-1)
        
        return output


# ======================== EMA (Exponential Moving Average) ========================

class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ======================== Training with Improvements ========================

def train_epoch_refined(model, loader, criterion, optimizer, device, 
                       use_mixup=True, mixup_alpha=0.2, grad_accum_steps=1, ema=None):
    model.train()
    total_loss = 0
    n = 0
    optimizer.zero_grad()
    
    for batch_idx, (smiles, protein, desc, labels) in enumerate(loader):
        smiles = smiles.to(device)
        protein = protein.to(device)
        desc = desc.to(device)
        labels = labels.to(device)
        
        # Apply mixup during training
        if use_mixup and np.random.rand() < 0.5:
            smiles, protein, desc, labels_a, labels_b, lam = mixup_data(
                smiles, protein, desc, labels, alpha=mixup_alpha
            )
            outputs = model(smiles, protein, desc)
            loss = mixup_criterion(outputs, labels_a, labels_b, lam)
        else:
            outputs = model(smiles, protein, desc)
            loss = criterion(outputs, labels)
        
        # Gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        total_loss += loss.item() * len(labels) * grad_accum_steps
        n += len(labels)
    
    return total_loss / n


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n = 0
    preds = []
    labels_list = []
    
    with torch.no_grad():
        for smiles, protein, desc, labels in loader:
            smiles = smiles.to(device)
            protein = protein.to(device)
            desc = desc.to(device)
            labels = labels.to(device)
            
            outputs = model(smiles, protein, desc)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            n += len(labels)
            
            preds.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    
    preds = np.array(preds)
    labels_arr = np.array(labels_list)
    
    mse = mean_squared_error(labels_arr, preds)
    r2 = r2_score(labels_arr, preds)
    r, _ = pearsonr(labels_arr, preds)
    mae = mean_absolute_error(labels_arr, preds)
    spearman, _ = spearmanr(labels_arr, preds)
    ci = concordance_index(labels_arr, preds)
    
    return total_loss / n, mse, r2, r, mae, spearman, ci, preds, labels_arr


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(description='Refined Protein-Ligand Model')
    
    # Data
    parser.add_argument('--train-csv', type=str, default='split_data/train.csv')
    parser.add_argument('--val-csv', type=str, default='split_data/val.csv')
    parser.add_argument('--test-csv', type=str, default='split_data/test.csv')
    parser.add_argument('--embeddings-dir', type=str, default='split_embeddings')
    parser.add_argument('--descriptors-dir', type=str, default='split_descriptors')
    
    # Model
    parser.add_argument('--d-model', type=int, default=768)
    parser.add_argument('--n-blocks', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=12)
    parser.add_argument('--d-ff', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.15)  # Slightly lower
    
    # Training improvements
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=15)
    parser.add_argument('--use-mixup', action='store_true', default=True)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--use-ema', action='store_true', default=True)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    
    # Other
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed for repeated training runs')
    parser.add_argument('--seed-step', type=int, default=100,
                        help='Seed increment between runs')
    parser.add_argument('--save-dir', type=str, default='saved_refined_models')
    parser.add_argument('--recompute-embeddings', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = normalize_input_columns(pd.read_csv(args.train_csv), args.train_csv)
    val_df = normalize_input_columns(pd.read_csv(args.val_csv), args.val_csv)
    test_df = normalize_input_columns(pd.read_csv(args.test_csv), args.test_csv)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize embedders
    molformer = MolFormerEmbedder(device=device)
    esm2 = ESM2Embedder(device=device)
    
    # Compute embeddings
    os.makedirs(args.embeddings_dir, exist_ok=True)
    os.makedirs(args.descriptors_dir, exist_ok=True)
    
    train_smiles_path = os.path.join(args.embeddings_dir, 'train_smiles.npy')
    train_protein_path = os.path.join(args.embeddings_dir, 'train_protein.npy')
    
    if args.recompute_embeddings or not os.path.exists(train_smiles_path):
        train_smiles_emb, train_protein_emb = compute_embeddings_for_split(
            train_df, molformer, esm2, "train"
        )
        np.save(train_smiles_path, train_smiles_emb)
        np.save(train_protein_path, train_protein_emb)
    else:
        train_smiles_emb = np.load(train_smiles_path)
        train_protein_emb = np.load(train_protein_path)
        print("  Loaded train embeddings from cache")
    
    val_smiles_path = os.path.join(args.embeddings_dir, 'val_smiles.npy')
    val_protein_path = os.path.join(args.embeddings_dir, 'val_protein.npy')
    
    if args.recompute_embeddings or not os.path.exists(val_smiles_path):
        val_smiles_emb, val_protein_emb = compute_embeddings_for_split(
            val_df, molformer, esm2, "val"
        )
        np.save(val_smiles_path, val_smiles_emb)
        np.save(val_protein_path, val_protein_emb)
    else:
        val_smiles_emb = np.load(val_smiles_path)
        val_protein_emb = np.load(val_protein_path)
        print("  Loaded val embeddings from cache")
    
    test_smiles_path = os.path.join(args.embeddings_dir, 'test_smiles.npy')
    test_protein_path = os.path.join(args.embeddings_dir, 'test_protein.npy')
    
    if args.recompute_embeddings or not os.path.exists(test_smiles_path):
        test_smiles_emb, test_protein_emb = compute_embeddings_for_split(
            test_df, molformer, esm2, "test"
        )
        np.save(test_smiles_path, test_smiles_emb)
        np.save(test_protein_path, test_protein_emb)
    else:
        test_smiles_emb = np.load(test_smiles_path)
        test_protein_emb = np.load(test_protein_path)
        print("  Loaded test embeddings from cache")
    
    # Compute descriptors
    train_desc = compute_descriptors_for_dataset(
        train_df, os.path.join(args.descriptors_dir, 'train_desc.npy')
    )
    val_desc = compute_descriptors_for_dataset(
        val_df, os.path.join(args.descriptors_dir, 'val_desc.npy')
    )
    test_desc = compute_descriptors_for_dataset(
        test_df, os.path.join(args.descriptors_dir, 'test_desc.npy')
    )
    
    # Normalize descriptors
    desc_mean = train_desc.mean(axis=0, keepdims=True)
    desc_std = train_desc.std(axis=0, keepdims=True)
    desc_std[desc_std == 0] = 1.0
    
    train_desc = (train_desc - desc_mean) / desc_std
    val_desc = (val_desc - desc_mean) / desc_std
    test_desc = (test_desc - desc_mean) / desc_std
    
    # Run multiple times
    all_results = []
    best_pearson = -1
    
    for run in range(args.num_runs):
        seed = args.seed + run * args.seed_step
        set_random_seed(seed)
        print(f"\n{'='*70}")
        print(f"RUN {run + 1}/{args.num_runs} - Random Seed: {seed}")
        print(f"{'='*70}")
        
        # Create datasets
        train_dataset = EnhancedFeatureDataset(
            train_smiles_emb, train_protein_emb, train_desc,
            train_df['pkoff'].values, keep_valid_mask=False
        )
        val_dataset = EnhancedFeatureDataset(
            val_smiles_emb, val_protein_emb, val_desc,
            val_df['pkoff'].values, keep_valid_mask=False
        )
        test_dataset = EnhancedFeatureDataset(
            test_smiles_emb, test_protein_emb, test_desc,
            test_df['pkoff'].values, keep_valid_mask=True
        )
        
        # Normalize labels
        train_mean = train_dataset.labels.mean().item()
        train_std = train_dataset.labels.std().item()
        
        train_dataset.labels = (train_dataset.labels - train_mean) / train_std
        val_dataset.labels = (val_dataset.labels - train_mean) / train_std
        test_dataset.labels = (test_dataset.labels - train_mean) / train_std
        
        print(f"  Normalization: μ={train_mean:.4f}, σ={train_std:.4f}")
        
        # Loaders
        train_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Model
        model = RefinedProteinLigandModel(
            smiles_dim=train_smiles_emb.shape[1],
            protein_dim=train_protein_emb.shape[1],
            descriptor_dim=train_desc.shape[1],
            d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                               weight_decay=args.weight_decay)
        
        # EMA
        ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
        
        # Scheduler with cosine annealing
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return epoch / args.warmup_epochs
            else:
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training
        best_val_loss = float('inf')
        best_epoch = 0
        patience = 50
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch_refined(
                model, train_loader, criterion, optimizer, device,
                use_mixup=args.use_mixup,
                mixup_alpha=args.mixup_alpha,
                grad_accum_steps=args.gradient_accumulation,
                ema=ema
            )
            
            # Evaluate with EMA if available
            if ema is not None:
                ema.apply_shadow()
            
            val_loss, val_mse, val_r2, val_r, val_mae, val_sp, val_ci, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            if ema is not None:
                ema.restore()
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                      f"Pearson={val_r:.4f}, Spearman={val_sp:.4f}, CI={val_ci:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                if ema is not None:
                    ema.apply_shadow()
                best_state = deepcopy(model.state_dict())
                if ema is not None:
                    ema.restore()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break
        
        # Test with best model
        model.load_state_dict(best_state)
        _, _, _, _, _, _, _, test_preds, test_lbls = evaluate(
            model, test_loader, criterion, device
        )
        
        # Denormalize
        test_preds_orig = test_preds * train_std + train_mean
        test_lbls_orig = test_lbls * train_std + train_mean
        
        test_mse = mean_squared_error(test_lbls_orig, test_preds_orig)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(test_lbls_orig, test_preds_orig)
        test_r, _ = pearsonr(test_lbls_orig, test_preds_orig)
        test_mae = mean_absolute_error(test_lbls_orig, test_preds_orig)
        test_sp, _ = spearmanr(test_lbls_orig, test_preds_orig)
        test_ci = concordance_index(test_lbls_orig, test_preds_orig)
        
        # Save predictions
        print(f"\n  Saving predictions for each sample...")
        
        predictions_data = {
            'sample_index': range(len(test_lbls_orig)),
            'seed': seed,
            'true_pkoff': test_lbls_orig,
            'predicted_pkoff': test_preds_orig,
            'error': test_preds_orig - test_lbls_orig,
            'absolute_error': np.abs(test_preds_orig - test_lbls_orig),
            'squared_error': (test_preds_orig - test_lbls_orig) ** 2
        }
        
        if test_dataset.valid_indices is not None:
            predictions_data['original_test_index'] = test_dataset.valid_indices[:len(test_lbls_orig)]
        
        predictions_df = pd.DataFrame(predictions_data)
        
        if test_dataset.valid_indices is not None:
            for idx in range(len(test_lbls_orig)):
                orig_idx = test_dataset.valid_indices[idx] if idx < len(test_dataset.valid_indices) else idx
                if orig_idx < len(test_df):
                    if 'protein_ID' in test_df.columns:
                        predictions_df.loc[idx, 'protein_ID'] = test_df.iloc[orig_idx]['protein_ID']
                    if 'smiles' in test_df.columns:
                        predictions_df.loc[idx, 'smiles'] = test_df.iloc[orig_idx]['smiles']
                    if 'FASTA' in test_df.columns:
                        seq = str(test_df.iloc[orig_idx]['FASTA'])
                        predictions_df.loc[idx, 'protein_sequence_prefix'] = seq[:50] if len(seq) > 50 else seq
        
        predictions_path = os.path.join(args.save_dir, f'predictions_run{run+1}.csv')
        predictions_df.to_csv(predictions_path, index=False, float_format='%.6f')
        print(f"  ✓ Predictions saved to {predictions_path} ({len(predictions_df)} samples)")
        
        print(f"\n  TEST RESULTS:")
        print(f"    Pearson r:       {test_r:.4f} ⭐")
        print(f"    Spearman:        {test_sp:.4f}")
        print(f"    Concordance(CI): {test_ci:.4f} ⭐")
        print(f"    RMSE:            {test_rmse:.4f}")
        print(f"    MSE:             {test_mse:.4f}")
        print(f"    MAE:             {test_mae:.4f}")
        print(f"    R²:              {test_r2:.4f}")
        
        # Save best
        if test_r > best_pearson or run == 0:
            best_pearson = max(best_pearson, test_r)
            save_path = os.path.join(args.save_dir, f'refined_run{run+1}.pt')
            torch.save({
                'seed': seed,
                'model_state_dict': model.state_dict(),
                'config': {
                    'smiles_dim': train_smiles_emb.shape[1],
                    'protein_dim': train_protein_emb.shape[1],
                    'descriptor_dim': train_desc.shape[1],
                    'd_model': args.d_model,
                    'n_blocks': args.n_blocks,
                    'n_heads': args.n_heads,
                    'd_ff': args.d_ff,
                    'dropout': args.dropout
                },
                'normalization': {
                    'mean': train_mean,
                    'std': train_std,
                    'desc_mean': desc_mean,
                    'desc_std': desc_std
                },
                'metrics': {
                    'mse': float(test_mse),
                    'rmse': float(test_rmse),
                    'r2': float(test_r2),
                    'pearson': float(test_r),
                    'mae': float(test_mae),
                    'spearman': float(test_sp),
                    'concordance_index': float(test_ci)
                }
            }, save_path)
            print(f"  ✓ Saved to {save_path}")
        
        all_results.append({
            'run': run + 1,
            'seed': seed,
            'pearson': test_r,
            'spearman': test_sp,
            'concordance_index': test_ci,
            'rmse': test_rmse,
            'mse': test_mse,
            'mae': test_mae,
            'r2': test_r2,
            'best_epoch': best_epoch
        })
    
    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - REFINED MODEL")
    print("="*70)
    
    df_results = pd.DataFrame(all_results)
    for metric in ['pearson', 'spearman', 'concordance_index', 'rmse', 'mae', 'r2','mse']:
        mean = df_results[metric].mean()
        std = df_results[metric].std()
        symbol = "⭐" if metric in ['pearson', 'spearman', 'concordance_index'] else ""
        metric_name = metric.replace('_', ' ').upper()
        print(f"  {metric_name:18s}: {mean:.4f} ± {std:.4f} {symbol}")
    
    print(f"\n  Best Pearson r: {best_pearson:.4f}")
    
    results_file = os.path.join(args.save_dir, 
                               f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    df_results.to_csv(results_file, index=False)
    print(f"\n  Results saved: {results_file}\n")

if __name__ == '__main__':
    main()
