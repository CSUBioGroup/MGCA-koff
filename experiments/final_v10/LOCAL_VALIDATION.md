# 本地验证记录（2026-09-10）

## 已完成

- `python -m unittest test_model_contract test_training_resume test_workflow -v`：**8 项测试全部通过**，最终整套执行约 18.9 秒。
- 真实 PyTorch 前向/反向：9 组初始化 × Full/去 Protein/去 Drug/去 Joint；所有参数量为 13,392,395，损失/梯度有限。
- 正值无上限门：初始化从 1e-7 至 100（包括大于 1）；精确初始化、正梯度、非法初值拒绝。
- 实际 trainer + 合成 ESM2/Morgan 特征：在 epoch 7 与早停 epoch 12 持久化后模拟异常，恢复与不中断的最佳 state_dict 和预测完全相同；没有额外早停 epoch。
- 检查 checkpoint 中 optimizer 两组 decay：普通参数遵循设定 .0001，两个融合坐标为 0。
- 实际 trainer 的正式 test 预测及注意力 NPZ 导出、逐样本/指标校验、完成封条验证通过；修改已封存 metrics 文件后正确拒绝复用。
- Optuna SQLite + 合成目标：第 4 个 trial 中断后重放，12 个提议（包含超出 startup 的 TPE 提议）与不中断序列相同，最终冻结参数相同；选择阶段若调用 test_path 测试立即失败，本次未触发。
- 正式任务组合：每数据集 60 个任务；Full/消融使用相同冻结参数；warm/drug 固定 seed，protein 固定 fold 配五 seeds。
- 完整合成结果矩阵：60/60 汇总、12 个 mean/SD 分组、4 个 protein-cold ensemble、9 个配对消融统计正常；无结果的矩阵报告 0/60 而不是成功。
- 本地真实数据只执行 plan：22 组 train/validation、44 个文件路径及 SHA256 可读；未加载 ESM2，也未打开 test CSV。临时计划放在 preview/，不包含在发布包中。
- 全部 13 个 Python 文件通过 Python 3.8 语法解析；两个 Bash 脚本通过 `bash -n`。
- 原 v10 模型与 sources/original_v10_model.py 哈希一致：`e6efef46abec20d1b25305c69ca099e2a5f84f6906f7bcb3f9b1e6c1786d7f7b`；原文件未修改。

## 边界

本地使用 CPU PyTorch 2.14.0、Optuna 4.9.0。为避免下载模型及绘图库依赖，网络与 trainer 测试从实际源文件 AST 加载真实定义，ESM2 特征使用合成 fixture。流程测试中的评分也是合成的，不是实验性能。

没有完成服务器 Python 3.8 / CUDA / ESM2 权重的端到端真实训练，也没有声称新模型已经优于 v10。运行服务器一键脚本时，preflight 会再次核验真实依赖、GPU、数据与模型权重。
