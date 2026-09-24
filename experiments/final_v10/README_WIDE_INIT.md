# 同一宽初值网格重跑（v2）

这是一套新的超参数搜索协议，不修改模型或旧实验结果。请使用新增入口，不要用旧入口启动本轮。

```bash
cd <BIO_PROJECT_ROOT>/mgca_hyperparameter_tuning/v10_unbounded_jointtune
bash run_all_wide_init.sh
```

Drug 和 Joint 都使用以下实际初始相对权重（不是 softplus 的原始坐标）：

`0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5`

较小权重区域保留更细的刻度，同时覆盖更大的残差贡献。相同的网格消除了两路候选范围预先不同的设置，但模型仍保留 Protein=1 的参照及原顺序辅助监督，并非完全对称架构。0.5 是初值候选最大值，不是训练上限。

默认结果目录：`outputs/unbounded_warm_hpo_wide_init_v2/`。旧目录 `outputs/unbounded_warm_hpo_v1/` 不修改。模型、LR/weight decay/batch/dropout/window 的搜索范围、训练日程、seeds、数据划分和 warm-only validation 目标全部保持原设置。

每数据集仍为 30 次串行 TPE + Top-5 五次复评，随后两个数据集三协议的 Full 和三项分支消融共 120 个正式任务。最多 100 个调参/复评训练任务加 120 个正式任务（重复配置可复用）。RUN_JOBS=5，可配置。

两路有 8×8=64 种初值组合；30 次 TPE 并非完整网格遍历，也不能保证找到全局最优。预算不在本次默认修改中增加。若开始前决定提高预算，必须固定新预算并使用独立输出目录，例如：

```bash
TUNING_TRIALS=60 OUTPUT_ROOT="$PWD/outputs/unbounded_warm_hpo_wide_init_t60_v2" bash run_all_wide_init.sh
```

即使 60 次也不保证遍历全部组合，因为 TPE 同时搜索其他参数且允许重复初值组合。

可先仅生成计划检查：

```bash
PHASE=plan bash run_all_wide_init.sh
```

恢复未完成阶段（仍使用新增入口）：

```bash
PHASE=tuning bash run_all_wide_init.sh
PHASE=formal bash run_all_wide_init.sh
PHASE=ablation bash run_all_wide_init.sh
PHASE=summary bash run_all_wide_init.sh
RUN_JOBS=2 CANDIDATE_REVIEW_JOBS=2 bash run_all_wide_init.sh
```

环境变量优先于配置默认值。运行前请确认没有遗留的旧 OUTPUT_ROOT/CACHE_ROOT/DRUG_INIT_CHOICES/JOINT_INIT_CHOICES；科学配置改变但指定旧目录时，哈希检查会拒绝混用。默认缓存也使用新目录；旧缓存不删除、不自动迁移。

本地仅验证 Bash 语法、配置继承、真实 train/validation 路径计划与无上限标量初始化，未启动服务器 GPU 训练。本轮仍是历史测试反馈之后的新开发实验，不改变原测试集已被评估过的事实。
