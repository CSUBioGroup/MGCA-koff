# MGCA v10：补充上限与初始化敏感性实验

这是固定 v10 架构的补充实验，不是 v19，也不是重新挑选一个 test 最好的模型。
模型文件完整复制自 v10，模型 SHA256 保持
`e6efef46abec20d1b25305c69ca099e2a5f84f6906f7bcb3f9b1e6c1786d7f7b`。
本目录自带最小运行依赖源码和两个数据集的原最优超参数 JSON；不需要读取 v10 旧 outputs，不写入旧模型、旧结果或数据目录。
仍需服务器已有的两个数据集划分、ESM2 权重以及原 v10 Python 环境。

## 1. 一键运行

将本目录放在 `<BIO_PROJECT_ROOT>/mgca_hyperparameter_tuning/v10_sensitivity`：

```bash
cd <BIO_PROJECT_ROOT>/mgca_hyperparameter_tuning/v10_sensitivity
bash run_all_sensitivity.sh
```

先检查 `experiment_config.sh` 的 `PROJECT_ROOT`、`ESM2_PATH` 和磁盘位置。默认 ESM2 路径是
`/root/private_data/DP/pretrained_model/esm2_t36`，如果实际路径不同，修改配置或设置环境变量。

```bash
ESM2_PATH=/你的实际路径/esm2_t36 bash run_all_sensitivity.sh
```

默认顺序：冻结 plan → 数据/模型/GPU/ESM 哈希预检查 → 22 个唯一 train+val 特征缓存串行预计算 → 每组最多 5 路训练 → 完整汇总。
没有 Optuna、pilot 性能安全门、路由重校准，也没有根据结果自动晋级或淘汰配置。
保留 v10 原训练日程：1–5 epoch anchor、6–10 固定系数、11 起学习系数，同一 optimizer 连续训练。
删除原日程会改变待解释的模型，因此本补充实验不删除它。

仅生成并冻结计划（不需要加载 GPU/ESM，不训练）：

```bash
PHASE=plan bash run_all_sensitivity.sh
```

其他阶段：`PHASE=preflight|cache|run|summary`。
`run` 会先校验/补齐缓存；`summary` 不需要导入 PyTorch，但需要现有数据文件的哈希匹配冻结计划。

## 2. 固定实验网格

| 实验组 | Drug 上限 | Joint 上限 | Drug 初始化 | Joint 初始化 | 组合数 |
|---|---|---|---|---|---|
| caps | 0.15 / 0.30 / 0.60 | 0.06 / 0.12 / 0.24 | 0.10 | 0.03 | 9 |
| initialization | 0.30 | 0.12 | 0.05 / 0.10 / 0.20 | 0.015 / 0.03 / 0.06 | 9 |

两组共用原配置 `(cap_d,cap_j,init_d,init_j)=(0.30,0.12,0.10,0.03)`，因此只有 **17 个唯一配置**。
默认 **17 × 2 数据集 × 3 协议 × 5 次 = 510 次训练**，其中原配置 30 次只跑一次并在两组表格中共用。
不是上限与初始化四维全因子实验，不能从本设计推断所有四维交互。

可选等上限对照：在首次冻结计划前设 `INCLUDE_EQUAL_CAPS=true`，追加 `(0.15,0.15)` 与 `(0.30,0.30)`，初始化不变。
此时 19 个配置，共 570 次训练。这只能检查若干等上限替代，不是完整对称架构：protein anchor 与初始化不对称仍然存在。
默认关闭以保持与已讨论的两项敏感性实验一致。

所有倍率都在新增运行前一次性冻结；不能在看完结果后删除差的网格点。
已有 v10 的学习率、weight decay、batch、dropout、窗口、损失权重和训练轮数/早停保持不变。
2773 dropout=0.15；KinetX dropout=0.17。没有把新扫描与原 HPO 混合。

默认 fold/seed 精确沿用原 benchmark：warm/drug 五个 split、seed 42；protein 固定 split、seeds 42/142/242/342/442。
五次 split 不应称为五个独立 seed。
如需 10 次确认，必须在新输出目录运行：

```bash
OUTPUT_ROOT="$PWD/outputs/sensitivity_10runs_v1" \
  SEED_POLICY=paired_seeds N_RUNS=10 bash run_all_sensitivity.sh
```

此时所有协议使用 `42+100*(run-1)`，warm/drug 循环五个 fold 两遍，共 1020 次训练。
同一个 run 的所有配置使用相同 fold/seed；增加 seed 不会产生新独立测试集。

## 3. 如何解释上限

原参数化为 `alpha = cap * sigmoid(logit)`，初始化固定为上述 alpha 数值。
原 gate prior 为 `0.001 * [(alpha_d/cap_d)^2 + (alpha_j/cap_j)^2]`。
所以改变 cap 同时改变：可达范围、logit 的初始化/局部梯度、对 alpha 的有效正则强度。
例如将某一 cap 减半，该分支同一 alpha 下的正则惩罚变为四倍。
本实现忠实保留这一耦合，报告应称为“整体上限参数化敏感性”，不能称为隔离后的纯上限效应。
若要隔离正则影响，应另行预定义固定分母对照，不能把其结果混入本次网格。

“训练后权重没有接近 cap”不等于 cap 没有作用；“不同设置差异不显著”也不证明等效。
本代码不设置事后随意的“稳定/通过”阈值，不自动生成“0.30、0.12 最优”的结论。

## 4. 不访问 test

规划器只生成 train 和 val 路径；训练入口强制 `--selection-only` 并拒绝 test 参数。
特征缓存只含 train+val，ESM 路径和内容哈希写入独立清单。
不导入会扫描 test 的旧 preflight，不复用旧正式训练的 test 指标。
原配置也在本次协议下重新训练，避免将旧正式训练的缓存/软件状态差异混进配对对照。
使用同一 validation 早停后再报告其 MSE，因此这是带有 checkpoint 选择的验证敏感性，不是无偏泛化估计。
历史 test 已反馈 v10 的架构设计，本实验不能把历史 benchmark 重新变成“未查看过的 test”。

## 5. 恢复、锁与资源

`RUN_JOBS=5` 只控制调度；可以降到 1 后原地续跑：

```bash
RUN_JOBS=1 bash run_all_sensitivity.sh
```

运行前冻结科学配置、代码、数据、原超参文件、输出/缓存路径和相关运行环境。
改变这些内容会拒绝复用原目录；`RUN_JOBS`、`PHASE` 和磁盘最低余量不改变科学配置。
因此若后续迁移缓存或代码，也应新开目录，不手工修改 plan/identity。
设备或 AMP 的改变同样不在原输出目录复用。
顶层和每个 worker 都使用 OS 文件锁，异常退出后自动释放；不要靠删除 lock 文件来解锁活跃进程。

未完成任务从 last_state.pt 恢复模型、optimizer、AMP scaler、RNG 和 DataLoader generator。
修复仅在训练器副本中：RNG 状态在 CPU 恢复；早停后中断的任务不会多训练一轮；耗时跨恢复累计按 epoch 统计。
未定义相关系数保存为 null，不伪装成 0；非有限预测或 MSE 仍作为真实失败处理。
完成后原训练器移除自身 last_state.pt（约 160 MiB 量级的优化器恢复文件），保留 best_model.pt；不会清理原 v10。

全部默认任务仅最佳模型大约占 26 GiB（每个约 51 MiB，实际以 checkpoint_size_bytes 为准），还需缓存、预测、日志和临时恢复文件空间。
建议至少预留 40 GiB，具体取决于现有文件系统情况。默认低于 5 GiB 停止启动下一组，不删除数据、不自动压缩、不降低 batch size。
OOM/运行失败保存日志并让其他已规划配置继续；最终返回非零退出码。降低并发后重跑即可。
这是执行故障处理，不是性能门槛。
单 GPU 并发耗时不能和独占 GPU 串行基线直接比较；同时保存配置并发数及有完整起止时间的实际进程重叠统计。

## 6. 输出

`outputs/sensitivity_v1/summary/`：

- `report.html`：可直接浏览的完整矩阵报告，原配置灰底，不自动标红某个“获胜”配置。
- `2773_warm_caps.svg` 等 12 张可编辑矢量矩阵图（对应默认两数据集、三协议、两扫描组）。
- `validation_mean_sd.csv`：六指标 mean、sample SD、有效 n，以及最终系数/相对初始化变化/上限占比/耗时。
- `validation_per_run.csv`：所有 run 的验证结果与系数。
- `paired_vs_reference.csv`：逐 run 相对原配置差值，负 ΔMSE 表示该配置验证误差更低。
- `paired_statistics.csv`：配对均值差、描述性 run-bootstrap 区间、双侧精确符号置换、Holm 校正。
- `gate_epoch_trajectories.csv`：逐 epoch 系数、梯度、loss、validation、阶段、耗时记录。
- `completion_matrix.csv`、`audit.json`：包括未完成/无效任务，不能用仅完成部分宣称全网格稳定。
- `process_attempts.csv`：恢复前后的所有进程尝试、时间和并发信息。

统计家族预设为所有非原配置 × 数据集 × 协议（默认 16×2×3=96 项，原配置在两组中去重）。
只有整个家族完成后才给 Holm p；不因失败或未完成缩小比较家族。
默认 n=5 双侧精确符号置换最小 p=0.0625；不显著不能作为参数鲁棒/等效的证据。
bootstrap 重采样 paired run，区间条件于现有 fold/seed，不是样本级或新靶点人群区间；fold 共享训练样本也限制独立性解释。
各协议分别报告，不将三种不同难度的原始 MSE 任意混成一个新选参目标。

每个 run 保留：最佳 checkpoint、输入/模型/代码身份、环境、历史、train/validation 逐样本预测、validation 分支输出和注意力 NPZ、命令、stdout/stderr、每次退出码和失败原因。
原 `.complete` 之外再加 `verified.complete.json`，校验样本顺序、标签、预测一致性、MSE、数组维度/有限值以及所有核心文件 SHA256。
只对已验证完整且哈希一致的结果跳过。删除 best_model.pt 后该目录不再是可续跑的完整归档。

## 7. 测试与环境

推荐使用原成功运行 v10 的 Python 3.8 / PyTorch 1.12.1+cu113 / Transformers 4.40.1 / RDKit 2024.03.5 环境。
还需 numpy、pandas、scipy、scikit-learn、matplotlib、seaborn、tqdm。脚本不会自动升级/安装这些包。
对新 PyTorch 的本地受信 checkpoint 显式指定 weights_only=False；不要用该入口加载不可信来源的 checkpoint。

```bash
python -m unittest test_sensitivity -v
python test_model_contract.py
python test_training_resume.py
PHASE=preflight bash run_all_sensitivity.sh
```

前者测试计划、去重、seed、禁用 test、哈希、锁、缺失结果、统计和汇总。
第二项用真实 PyTorch 和原源码抽取的模型/编码器类检查全部 19 个可用设置的 forward/backward、初始化及 13,392,395 参数一致性；不需要下载 ESM2。
第三项用合成缓存特征运行实际训练器，在第 7 和第 12 epoch 注入中断，检查继续训练后的最佳权重与不中断训练逐 tensor 完全相同，以及早停后不多训练一轮。
这些不是 GPU/真实 ESM2 端到端测试；最后一项在服务器验证真实环境、模型、数据与 GPU。
本地代码交付未启动 510 次正式训练，也不预先声称任何参数组更稳健。
