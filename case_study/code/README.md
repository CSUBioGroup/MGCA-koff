# MGCA 最终冻结版：完整 2773 case study

本包冻结 `unbounded_warm_hpo_wide_init_v2`，不是新的模型版本。模型对外仍称 MGCA。
模型文件和 ESM2/Morgan 实现逐字节复制自该轮实际 benchmark 代码，不再修改架构或重新选参。
两个数据集的最优参数、科学代码哈希、计划 ID、现有五次消融记录均保存在 `frozen/`。

## 冻结决定

- 2773：Drug/Joint 初值 0.30/0.30，LR=5.9260857797183374e-5，WD=1e-4，batch=64，dropout=0.15，ESM2 window=2，`even_span_v2` 对应第 1--2、12--13、24--25 和 35--36 层。
- KinetX：Drug/Joint 初值 0.40/0.05，LR=2.432667795899065e-5，WD=1e-3，batch=16，dropout=0.10，window=8。该配置只归档，本包不训练 KinetX case 模型。
- Protein 系数为 1，两个残差系数为全局 softplus 无上限参数；不恢复旧 cap 或 cap prior。
- 保持两组 AdamW：普通参数使用选定 WD，两个 softplus 原始标量坐标的 WD=0。
- benchmark 与消融均沿用当前五次结果，不补十次，不借用旧版显著性。
- case study 只用完整 2773 训练五个模型，seeds=42/142/242/342/442，AMP=false。
- 固定 33 epochs：由本轮已冻结 winner 的五次 warm 验证最佳 epoch `[49,32,33,48,18]` 的中位数确定。
- 前 5 epochs anchor-only，6–10 epochs 固定残差系数，11–33 epochs 学习系数；不中断 optimizer，不用案例标签选 epoch。
- 固定最终 epoch checkpoint，五个均报告，不按案例表现筛选。没有安全/性能门，只检查输入、数值、哈希和执行完整性。

## 服务器一键运行

将整个文件夹上传至 `<BIO_PROJECT_ROOT>/case_study_mgca_final`：

```bash
cd <BIO_PROJECT_ROOT>/case_study_mgca_final
bash run_all_case_study.sh
```

默认 `RUN_JOBS=5`，可改为 1–5。只影响五个 checkpoint 的调度，不改变种子和 batch。
默认输出 `<BIO_PROJECT_ROOT>/case_study_outputs/mgca_final_unbounded_2773_v1`，绝不使用旧 `v10_2773`。
输入 case CSV、2773 原始 cohort 重建所需 CSV、模型和案例依赖脚本已包含在本包中；无需旧 Python 脚本目录。
ESM2 大权重不打包，默认 `${PROJECT_ROOT}/../pretrained_model/esm2_t36`。

```bash
ESM2_PATH=/absolute/path/to/esm2_t36 RUN_JOBS=3 bash run_all_case_study.sh
```

依赖沿用已完成 benchmark 的 Python/PyTorch/Transformers/RDKit 环境，还需 matplotlib、Pillow、gemmi 或 biopython。
本包不自动升级服务器依赖。若缺少 mmCIF 解析器，可在运行环境安装兼容的 gemmi。
ESM2 文件 SHA256 必须匹配原 benchmark 中记录的权重；不接受换编码器后继续复用缓存。

## 完整流程

1. 核验发布包和冻结配置；获取输出锁，拒绝跨代码/配置混跑。
2. 从冻结 warm run1 三个 split 重建 2773 全数据 cohort，并核对原始 koff.csv 和既有四目标曝光审计。
   这是训练全数据 refit，不是把 benchmark test 再用于 checkpoint 选择。
3. 核验 ESM2 权重、设备、输入和软件环境；串行提取窗口 2 的特征缓存。
4. 五个 seed 固定训练 33 epochs；留存完整模型、恢复状态、RNG、历史、训练预测、注意力/系数诊断。
5. 重算四目标面板和 K4DD 六目标面板的逐样本、逐 seed、均值/SD、ensemble、误差和排名；审计训练集曝光。
6. Factor Xa 与 DPP-4：重新提取 pre-pooling token 状态，执行单/双遮挡、五 seed 稳定性。
   主窗口16、步长4；窗口8和32、zero baseline 为保留的敏感性设置。
7. 每个面板/目标生成预测和排名图，两个遮挡目标生成 protein profile、Morgan bit bar、双遮挡矩阵。
8. Factor Xa：固定 `factor_xa_05`，复用包内经 SHA256 固定的 1NFU docking/redocking **几何数据**。
   不复用旧模型遮挡得分、旧结果着色图或旧模型与结构的联合结论。
   用本轮遮挡重新计算 residue mapping、contact 对照、结构着色、sequence panel 和审计。
9. 检查五个 checkpoint、两套面板、两个遮挡目标、图和结构分析的来源及文件哈希；生成 `summary/final_audit.json`、`final_report.md`、`.complete`。

结构绘图默认 `STRUCTURE_RENDERER=auto`：有 PyMOL 时输出 PyMOL 结构图，否则生成明确标注的 C-alpha/配体原子三维图。
数值结构分析始终执行，不会因为没有 PyMOL 跳过。PyMOL 脚本也保留，不能将替代图描述成 PyMOL 渲染。
可以指定另外的 PyMOL Python 环境：

```bash
STRUCTURE_PYTHON_BIN=/opt/conda/bin/python PHASE=structure bash run_all_case_study.sh
```

如服务器需要 libstdc++ 兼容处理，请在该独立绘图环境处理；本包不修改系统库。
几何数据本身不是本轮重做 docking 的结果，最终报告会明确注明复用来源。

## 断点恢复

`experiment_config.sh` 是允许调整的部署配置，不参与科学代码拒复用校验；模型和 JSON 科学设置仍严格校验。
两项双遮挡示例固定为 `factor_xa_05` 和 `dpp4_12`。后者沿用历史上“实测 pKoff 最高”的展示对象，
不根据本轮预测或遮挡效果重选，也不用于选择 checkpoint。可用 RDKit Draw 时额外生成配体环境 SVG；缺少绘图库时保留条形图并记录原因。

再次运行同一命令即可。已通过哈希验证的完整阶段跳过；中断训练从 `last_state.pt` 恢复 optimizer/scaler/RNG。
遇 OOM 不修改 batch，降低 RUN_JOBS 后续跑。
已完成文件被删除或损坏时明确报错，不静默重算替换；源码/输入/模型更改后拒绝复用当前结果。
训练日志在 `logs/train_seed_*.log`，各命令/开始结束时间/退出码在同名 JSON。
耗时记录并发配置；共享 GPU 进程时间不能当作独占 GPU 性能比较。

支持阶段：`plan preflight train four_target k4dd factor_xa dpp4 plots structure summary`，默认 `all`。

```bash
PHASE=plan bash run_all_case_study.sh     # 仅包/数据/协议检查，不加载 ESM2 或训练
PHASE=train RUN_JOBS=1 bash run_all_case_study.sh
PHASE=factor_xa bash run_all_case_study.sh
PHASE=summary bash run_all_case_study.sh
```

`all` 的数值阶段齐备后才生成最终完成标志。各 PHASE 完成不等于全流程完成。
部分内部文件名保留 `_v10` 以兼容已验证案例 API；所有入口均加载本包冻结的无上限模型，旧 checkpoint 会被拒绝。

## 论文交接

只有本轮服务器输出完成并分析后，才把新的案例数值写入正文/补充材料。
现有旧 v10 case study 仅保留为历史结果，不能改标签归入新模型。
五 seed SD 不是校准预测不确定性；post-ESM occlusion 不是原生残基—原子注意力；docking 不证明物理解离机制。
四目标与七目标是回顾性案例，不能因为冻结协议而称为全新的前瞻性验证集。

本地交付包括代码/冻结/数据检查和 CPU 合成测试，不包含已经完成的真实 GPU case 训练结果。
