# -*- coding: utf-8 -*-
"""
药物冷启动 - 单次 9:1 训练:验证（MOE 模型），并在独立测试集上评估

数据：
- 训练/验证：dataset/data.csv ；列名必须为 FASTA, SMILES, pkoff
- 独立测试：<PRIVATE_DATA_ROOT>/PDBbind-koff-680/output_pkoff_clean.csv ；同三列

特性：
- 复用 cold_start_framework 的 KineticsDataset / collate_fn / train_one_fold / compute_metrics
- 配体：mol2vec（necessary_files/model_300dim.pkl）+ Morgan 键为词向量索引 → 定长片段序列
- 蛋白：FASTA 三肽（necessary_files/res_list3.txt）→ 定长索引序列
- 模型：model_bimodal_regression_moe.moe
- 训练 50 轮（可改 --epochs）
- 评估 Train / Valid / Independent Test；另保存独测 CSV：fasta,smiles,lable,pred

用法示例：
python train_holdout_moe.py \
  --dataset_csv dataset/data.csv \
  --ind_test_csv <PRIVATE_DATA_ROOT>/PDBbind-koff-680/output_pkoff_clean.csv \
  --necessary_files_folder necessary_files \
  --device cuda:0
"""

import os
import sys
import csv
import time
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from gensim.models import word2vec
from rdkit import RDLogger
from tqdm.auto import tqdm
import warnings

sys.dont_write_bytecode = True

# 关闭噪声日志
warnings.filterwarnings("ignore", category=DeprecationWarning)
RDLogger.DisableLog('rdApp.*')

# === 导入模型与工具 ===
from model_bimodal_regression_moe import moe
from cold_start_framework import (
    set_seed, smiles2fragvec, fasta2resseq3,
    KineticsDataset, collate_fn, compute_metrics, train_one_fold
)

# ----------------------- 工具函数 -----------------------
def read_rows(csv_path: str) -> List[Tuple[str, str, float]]:
    """
    读取 CSV 的三列：FASTA, SMILES, pkoff
    返回 [(fasta, smiles, label), ...]
    """
    rows: List[Tuple[str, str, float]] = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        try:
            fasta_idx = header.index('FASTA')
            smiles_idx = header.index('SMILES')
            label_idx = header.index('pkoff')
        except ValueError:
            raise RuntimeError('CSV must have columns: FASTA, SMILES, pkoff')
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip('\n').split(',')
            # 若 SMILES 内含逗号，保守重组
            if len(parts) > 3:
                fasta = parts[fasta_idx]
                label = float(parts[-1])
                smiles = ','.join(parts[1:-1]) if smiles_idx == 1 else parts[smiles_idx]
            else:
                fasta, smiles, label = parts[fasta_idx], parts[smiles_idx], float(parts[label_idx])
            rows.append((fasta, smiles, float(label)))
    return rows


# ----------------------- 参数 -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description='MOE 单次 9:1 训练:验证 + 独立测试')
    parser.add_argument('--dataset_csv', type=str, default='../../dataset/data.csv')
    parser.add_argument('--ind_test_csv', type=str, default='<PRIVATE_DATA_ROOT>/PDBbind-koff-680/output_pkoff_clean.csv')
    parser.add_argument('--necessary_files_folder', type=str, default='necessary_files')
    parser.add_argument('--output_dir', type=str, default='outputs_moe_holdout')

    # 模型/数据超参
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--x1_num', type=int, default=74)
    parser.add_argument('--x1_dim', type=int, default=300)
    parser.add_argument('--x2_num', type=int, default=500)
    parser.add_argument('--x2_dim', type=int, default=30)
    parser.add_argument('--hid_dim', type=int, default=30)

    # 训练超参
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    # 选项
    parser.add_argument('--save_model', action='store_true', help='保存最佳模型（以验证集RMSE挑选）')
    return parser.parse_args()


# ----------------------- 主流程 -----------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    nec_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.necessary_files_folder))

    # 载入外部资源
    print("加载 mol2vec / 三肽词表 ...")
    mol2vec = word2vec.Word2Vec.load(os.path.join(nec_dir, 'model_300dim.pkl'))
    with open(os.path.join(nec_dir, 'res_list3.txt'), mode='r', encoding='utf-8') as fs:
        res_list = [line.strip() for line in fs.readlines()]

    # 读取主数据（用于 9:1 训练:验证）
    print(f"\n读取训练/验证数据: {args.dataset_csv}")
    rows = read_rows(args.dataset_csv)
    N = len(rows)
    print(f"样本总数: {N}")

    # 9:1 划分
    n_train = int(round(N * 0.9))
    n_val = N - n_train
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_tensor, val_tensor = random_split(range(N), [n_train, n_val], generator=g)
    train_rows = [rows[i] for i in list(train_tensor)]
    val_rows   = [rows[i] for i in list(val_tensor)]
    print(f"训练集: {len(train_rows)} | 验证集: {len(val_rows)}")

    # 数据集 & DataLoader
    train_dataset = KineticsDataset(train_rows, mol2vec, res_list, args.x1_num, args.x1_dim, args.x2_num)
    val_dataset   = KineticsDataset(val_rows,   mol2vec, res_list, args.x1_num, args.x1_dim, args.x2_num)

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=pin, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin, collate_fn=collate_fn)

    # 独立测试集
    print(f"\n读取独立测试集: {args.ind_test_csv}")
    test_rows = read_rows(args.ind_test_csv)
    print(f"独测样本数: {len(test_rows)}")
    test_dataset = KineticsDataset(test_rows, mol2vec, res_list, args.x1_num, args.x1_dim, args.x2_num)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=pin, collate_fn=collate_fn)

    # 构建模型
    model = moe(num_experts=args.num_experts,
                drop_r=args.dropout,
                res_list_num=len(res_list),
                x1_dim=args.x1_dim,
                x2_dim=args.x2_dim,
                hid_dim=args.hid_dim).to(device)

    # 训练（使用 cold_start_framework.train_one_fold）
    print("\n开始训练（9:1 验证）...")
    best_train_metrics, best_val_metrics, val_trues, val_preds = train_one_fold(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, val_freq=args.val_freq
    )

    # 评估：验证（最佳）与训练（结束后全量）
    print("\n验证集指标（训练过程中最佳）:")
    print(f"  Val MSE: {best_val_metrics['mse']:.6f} | RMSE: {best_val_metrics['rmse']:.6f} | "
          f"MAE: {best_val_metrics['mae']:.6f} | R2: {best_val_metrics['r2']:.6f}")

    print("训练集指标（训练结束后在整个训练集评估一次）:")
    model.eval()
    tr_preds, tr_trues, tr_valids = [], [], []
    with torch.no_grad():
        for x1, x2, y, v in tqdm(train_loader, desc="Eval Train", leave=False):
            x1 = x1.to(device); x2 = x2.to(device); y = y.to(device); v = v.to(device)
            pred = model(x1, x2).reshape(-1)
            tr_preds.append(pred.detach().cpu().numpy())
            tr_trues.append(y.detach().cpu().numpy())
            tr_valids.append(v.detach().cpu().numpy())
    tr_preds = np.concatenate(tr_preds, axis=0)
    tr_trues = np.concatenate(tr_trues, axis=0)
    tr_valids = np.concatenate(tr_valids, axis=0)
    tr_mask = tr_valids.reshape(-1) > 0.5
    train_metrics = compute_metrics(tr_trues[tr_mask], tr_preds[tr_mask]) if tr_mask.any() else {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    print(f"  Train MSE: {train_metrics['mse']:.6f} | RMSE: {train_metrics['rmse']:.6f} | "
          f"MAE: {train_metrics['mae']:.6f} | R2: {train_metrics['r2']:.6f}")

    # 时间戳（用于落盘文件名）
    ts = time.strftime('%Y%m%d_%H%M%S')

    # 评估：独立测试集
    print("\n评估独立测试集 ...")
    te_preds, te_trues, te_valids = [], [], []
    with torch.no_grad():
        for x1, x2, y, v in tqdm(test_loader, desc="Eval Test", leave=False):  # collate_fn 返回4元组
            x1 = x1.to(device); x2 = x2.to(device); y = y.to(device); v = v.to(device)
            pred = model(x1, x2).reshape(-1)
            te_preds.append(pred.detach().cpu().numpy())
            te_trues.append(y.detach().cpu().numpy())
            te_valids.append(v.detach().cpu().numpy())
    te_preds = np.concatenate(te_preds, axis=0)
    te_trues = np.concatenate(te_trues, axis=0)
    te_valids = np.concatenate(te_valids, axis=0)
    test_mask = te_valids.reshape(-1) > 0.5
    test_metrics = compute_metrics(te_trues[test_mask], te_preds[test_mask]) if test_mask.any() else {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    print(f"  Test  MSE: {test_metrics['mse']:.6f} | RMSE: {test_metrics['rmse']:.6f} | "
          f"MAE: {test_metrics['mae']:.6f} | R2: {test_metrics['r2']:.6f}")

    # 落盘：指标
    metrics_txt = os.path.join(args.output_dir, f'moe_holdout_metrics_{ts}.txt')
    with open(metrics_txt, 'w', encoding='utf-8') as f:
        f.write('MOE Holdout (9:1) with Independent Test\n')
        f.write('='*70 + '\n')
        f.write(f"dataset_csv: {args.dataset_csv}\n")
        f.write(f"ind_test_csv: {args.ind_test_csv}\n")
        f.write(f"epochs: {args.epochs}\n\n")

        f.write('Train Metrics:\n')
        for k in ['mse', 'rmse', 'mae', 'r2']:
            f.write(f"{k.upper()}\t{train_metrics[k]:.6f}\n")
        f.write('\nValid Metrics (best during training):\n')
        for k in ['mse', 'rmse', 'mae', 'r2']:
            f.write(f"{k.upper()}\t{best_val_metrics[k]:.6f}\n")
        f.write('\nTest  Metrics:\n')
        for k in ['mse', 'rmse', 'mae', 'r2']:
            f.write(f"{k.upper()}\t{test_metrics[k]:.6f}\n")

    # 落盘：预测（验证）
    np.savetxt(os.path.join(args.output_dir, f'pred_valid_{ts}.txt'),
               np.vstack([val_trues, val_preds]).T, header='true\tpred',
               fmt='%.6f', delimiter='\t', comments='')

    # 落盘：预测（独测 txt）
    np.savetxt(os.path.join(args.output_dir, f'pred_test_{ts}.txt'),
               np.vstack([te_trues[test_mask], te_preds[test_mask]]).T, header='true\tpred',
               fmt='%.6f', delimiter='\t', comments='')

    # 额外：保存独立测试集预测到 CSV（列：fasta,smiles,lable,pred）
    te_csv_rows = read_rows(args.ind_test_csv)              # 原 CSV 顺序
    te_fasta_all  = [r[0] for r in te_csv_rows]
    te_smiles_all = [r[1] for r in te_csv_rows]

    if len(te_fasta_all) != len(te_preds):
        print("⚠ 警告：独测 CSV 行数与预测数不一致，将按有效 mask 对齐截取。")

    idx_mask = np.where(test_mask)[0].tolist()
    te_fasta  = [te_fasta_all[i]  for i in idx_mask]
    te_smiles = [te_smiles_all[i] for i in idx_mask]

    csv_path = os.path.join(args.output_dir, f'pred_test_{ts}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['fasta', 'smiles', 'lable', 'pred'])   # 按你的要求使用 lable
        for fa, smi, yt, yp in zip(te_fasta, te_smiles, te_trues[test_mask], te_preds[test_mask]):
            writer.writerow([fa, smi, f"{float(yt):.6f}", f"{float(yp):.6f}"])
    print(f"\n已保存独立测试集 CSV：{csv_path}")

    # 保存模型
    if args.save_model:
        model_path = os.path.join(args.output_dir, f'moe_best_{ts}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"已保存模型到：{model_path}")

    print("\n=== 完成 ===")
    print(f"指标与预测已保存至：{args.output_dir}")


if __name__ == '__main__':
    main()
