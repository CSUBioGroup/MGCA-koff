# MGCA v10 无上限相对权重 + 统一超参数搜索

独立实验目录：`v10_unbounded_jointtune`。旧 v10、旧调参、敏感性结果均不修改、不复用训练结果。`sources/original_v10_model.py` 是完整原模型副本；实际新模型为 `runtime/mgca_hyperparameter_tuning/v10_unbounded/model.py`。

## 1. 模型与本次变更

保持 v10 的四深度 ESM2 蛋白专家、Morgan 半径 0–3 药物专家、模态内 gated fusion、低秩真实双向专家交互、三路特征归一化、同一个确定性预测 MLP、蛋白辅助损失、两项顺序 utility 损失及 joint-only 分支 dropout。

融合为：

`z = RMSNorm(h_protein + softplus(raw_drug) * h_drug + mask_joint * softplus(raw_joint) * h_joint)`

Protein 的系数 1 是相对权重的参照；Drug/Joint 可以超过 1。不做三权重 softmax，不引入新 MoE，也不是逐样本不确定性路由。原 v10 的蛋白锚点结构及顺序辅助监督仍存在，并未宣称改成完全对称模型。

本次修改明确包括：

1. 删除 `drug_gate_cap`、`joint_gate_cap`，两门改为稳定 softplus；初始化用其稳定逆函数精确设置。
2. 删除原按 cap 归一化的 gate prior；保留输出中的 `gate_prior=0` 仅便于旧统计字段读取，实际不产生任何梯度。
3. 两个初始化加入联合 HPO；Protein 系数不另加搜索维度。
4. 仅两个 softplus 原始坐标的 AdamW weight decay 固定为 0。对负坐标施加 decay 会把权重推向 `softplus(0)=log(2)`，故明确排除；其余网络参数 decay 仍参与搜索。
5. 原 v10 连续训练日程保持：epoch 1–5 仅蛋白锚点；6–10 启用固定初值修正；11+ 学习两标量，只有这一段 checkpoint 可被选为最佳。保持一个 optimizer，不新增 router recalibration 或根据性能通关的阶段。若要取消该日程，必须作为另一个实验配置，不应与本次机制变更混淆。

注意：取消上限不保证标量远离初值，也不保证性能提高。初值候选仍来自已开展的敏感性范围，可能偏向小残差；本轮并未声称穷尽所有权重尺度。允许在运行前用配置改变正值初始候选（包括大于 1）；开始后禁止改变科学配置复用结果。

## 2. 统一搜索与正式实验

每数据集独立搜索以下七项：

| 参数 | 搜索范围 |
|---|---|
| learning rate | log-uniform 1e-5–5e-4 |
| weight decay（不含两个标量） | 0 / 1e-6 / 1e-5 / 1e-4 / 1e-3 / 1e-2 |
| batch size | 16 / 32 / 64 |
| dropout | .05 / .10 / .15 / .17 / .20 / .25 / .30 |
| ESM2 window size | 1 / 2 / 4 / 6 / 8 |
| ESM2 window layout | even_span_v2 (four legal starts distributed as evenly as integer layers permit) |
| Drug 初始相对权重 | .05 / .10 / .20 |
| Joint 初始相对权重 | .015 / .03 / .06 |

Stage 1：warm run1、seed42、30 次串行 Optuna TPE，目标仅 validation MSE。

Stage 2：最优的 5 个唯一配置，warm run1–5 复评。run1 使用与 Stage 1 相同的任务目录并验证后跳过，不重复训练。按五次 validation MSE 均值、样本 SD、trial 编号依次排序。冻结每数据集自己的 best_params.json/sh 和 config ID。

**不同时改变选参目标**：本版不是三协议联合选参，不读取 test 进行 HPO。调参预检查也只打开 train/validation。全部数据集参数冻结后，才创建正式 test manifest、读取 test 文件进行独立审计和最终评估。

默认正式实验：两个数据集 × 三协议 × Full/去 Protein/去 Drug/去 Joint × 五次 = **120 个任务**，其中 Full 30、消融 90。

- Warm、drug-cold：原五套划分，seed42 固定。
- Protein-cold：原固定 split，seeds42/142/242/342/442。
- 每个数据集所有协议和所有消融均使用该数据集 Full 选出的同一套参数，不单独给消融选参。
- 分支消融仅关闭对应融合分支；Joint 保留时仍读取两种模态专家。因此“去 Protein 分支”并非“完全不输入蛋白”。四个实例参数量一致：13,392,395。
- 上述原模型与新模型的比较同时涉及标量参数化、先验删除、标量 decay 排除及联合选参，不能归因于单一因素。
- 搜索维度增加但默认预算仍为 30，属于有预算的搜索结果，不宣称数学全局最优。若提高预算，应开始前确定，并与对照模型的选择预算一并披露。

## 3. 一键运行（Linux 服务器）

把代码目录解压至 `<BIO_PROJECT_ROOT>/mgca_hyperparameter_tuning/`。数据仍使用 Bio 目录原 CSV；ESM2 权重不包含在代码包中。检查 `experiment_config.sh` 的 ESM2_PATH 和 PROJECT_ROOT。

```bash
cd <BIO_PROJECT_ROOT>/mgca_hyperparameter_tuning/v10_unbounded_jointtune
bash run_all_unbounded.sh
```

默认 RUN_JOBS=5、CANDIDATE_REVIEW_JOBS=5；共享 GPU，并非独占计时。ESM2 缓存预计算始终串行，ESM_BATCH_SIZE=1，避免已有服务器的 ESM 批量提取 OOM；训练 batch 仍由 HPO 决定，不会自动减小。

```bash
# 仅生成不可变计划，不加载 PyTorch/ESM2，不读 test：
PHASE=plan bash run_all_unbounded.sh
# 独立恢复阶段：
PHASE=tuning bash run_all_unbounded.sh
PHASE=formal bash run_all_unbounded.sh
PHASE=ablation bash run_all_unbounded.sh
PHASE=summary bash run_all_unbounded.sh
# OOM 后保留原参数，降低任务并发恢复：
RUN_JOBS=2 CANDIDATE_REVIEW_JOBS=2 bash run_all_unbounded.sh
```

阶段为 all/plan/preflight/tuning/formal/ablation/summary。正式或消融阶段需两个数据集都已有冻结参数。仅调整并发和 MIN_FREE_GIB 不改变科学配置；模型/依赖代码、数据、HPO设置、训练配置或 ESM 内容改变时拒绝复用旧结果。不要删除锁文件或完成标记来强行绕过校验。

## 4. 依赖与运行校验

面向现有 Python 3.8+ / PyTorch 环境，依赖 numpy、pandas、scipy、scikit-learn、rdkit、transformers、matplotlib、seaborn、tqdm、optuna。无需安装多个历史模型目录；所需 legacy 工具已随包复制。无需先升级服务器依赖；首次 preflight 记录并锁定关键版本和 ESM 权重哈希。冷启动框架 optional import 的 simple set_seed 警告不影响此独立 trainer 的显式随机种子控制。

预检查验证模型/消融参数量、有限值、ESM 文件、原划分及磁盘；冷启动实体重叠和损坏数据为错误。既有 warm canonical-equivalent 重叠保留为明确警告，不伪装成严格 canonical OOF。Protein-cold 检查 exact-sequence 不重叠，不冒称完成同源聚类审计。

只因 OOM、非有限预测、损坏文件、配置变化或真实资源不足等执行/数据错误而失败；没有性能门或消融效果安全门。失败 trial 原样保留，续跑不以新 trial 替代。Optuna 使用 proposal journal + 确定性 ask/tell 重放恢复 TPE RNG；SQLite 保存所有已完成 trial，尚未完成的 trial 由 proposals/*.json 与该 run 的 last_state.pt 保存。

每个输出目录有 OS 运行锁、每个 run/cache 有独立锁。训练前串行生成缓存；并发训练只加载已封存缓存，不并发加载 ESM2。缓存原子写入，有内容哈希和形状/有限值校验；未封存的中断缓存仅在该目录改名保留，绝不自动删除用户旧数据。

## 5. 留存与汇总

每个 run 保存最佳模型；未完成时保存 last_state（model、optimizer、AMP scaler、Python/NumPy/Torch/CUDA/DataLoader RNG、历史与最佳 epoch），完成且评估写出后移除已不需要的 last_state。保存 train/validation/test 逐样本预测（调参没有 test）、六项指标、epoch 历史、全局权重与贡献、双向注意力、专家权重、分支预测/增益 NPZ、环境、运行命令、各次日志、失败原因、耗时/显存、identity 与文件 SHA256 完成封条。

完整性检查后才写 `verified.complete.json`。只以经过哈希验证的封条判断可复用；底层 `.complete` 不是顶层的完整性依据。不存在“只看到一个 metrics.json 就自动跳过”。旧 checkpoint 不纳入本次科学实验。

`outputs/unbounded_warm_hpo_v1/summary/` 包含：

- test_mean_sd.csv / test_per_run.csv / report.md：两个数据集全部正式结果与三项消融。
- completion_matrix.csv / audit.json：预期与完成矩阵；未完成阶段明确标注。
- paired_mse_differences.csv / paired_statistics.json：配对差值、run bootstrap、按 raw-pair 聚类的样本 bootstrap、精确符号置换检验与跨所有 MSE 消融比较的 Holm 校正。
- gate_epoch_trajectories.csv：权重轨迹、分支贡献、损失、验证指标与计时。
- pooled_ensemble_metrics.csv / test_coverage_audit.csv / predictions/：protein-cold 五 seed ensemble；warm/drug 的原始 pair 覆盖审计及 pooled 预测，不未经核验就标为严格 canonical OOF。

样本 bootstrap 条件于已拟合的这些模型，不是新训练 seeds 或新蛋白簇的置信区间。n=5 的双侧精确符号检验最小 p=.0625，无显著差异不能推出等效。

## 6. 测试及代码打包

```bash
python -m unittest test_model_contract test_training_resume test_workflow -v
python build_release.py
```

测试含真实 PyTorch 网络/反向传播与 crash/resume；为了离线执行，用 AST 加载实际模块类，ESM2 特征由合成 fixture 替代。流程测试用合成训练评分核验 HPO，不把它当成真实性能结果。本地测试不能替代服务器的 CUDA/ESM2 真实训练验证。

所有历史测试反馈已参与模型开发，本轮属于后续开发实验。严禁根据正式 test 再覆盖或挑选训练结果；若要继续改动，应另建协议与输出目录。
