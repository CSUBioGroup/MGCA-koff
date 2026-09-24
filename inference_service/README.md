# KinetX 全量重拟合 + 常驻预热 FastAPI

## 固定模型与训练范围

本包使用已冻结的最终无上限 softplus MGCA，不修改模型架构、不重新调参，也不覆盖 2773 权重或论文结果。训练数据为此前 KinetX 实验使用的完整清洗子集 **5,446 行**，将 warm run1 的 train/val/test 合并为一个全量训练集，保留重复测量。不是原始 `KinetX.csv` 的 5,624 行；该原始文件含序列问题，不静默混入。

固定 lr=2.432667795899065e-5、weight_decay=0.001、batch_size=16、dropout=0.1、ESM window=8、drug/joint 初始化=0.4/0.05、protein 系数=1。ESM2 使用 `even_span_v2`，对应第 1--8、10--17、20--27 和 29--36 层；其缓存使用显式 `__wleven_span_v2` 后缀，旧 `legacy_anchors_v1` 缓存与 checkpoint 不可复用。选定候选的五次纯验证最佳 epoch 是 88/56/45/55/46，沿用中位数规则固定 **55 epochs**；本次只训练用户指定的 **seed=43**。训练不中途挑选，不构建 validation/test loader；全量数据训练误差不作为泛化指标。

保留原实现连续训练日程：epoch1-5 protein warmup，6-10 固定系数联合分支，11-55 学习系数；这是一次连续优化，不是重新调参或额外校准阶段。全程 float32，无 AMP、无系数上限、无 scalar weight decay，梯度裁剪5。哈希改变拒绝续跑。

## 1. 上传并一键训练、打包

将 `mgca_kinetx_full_service_seed43_code.zip` 上传到原服务器，使用之前成功运行 MGCA 的环境：

```bash
cd <BIO_PROJECT_ROOT>
unzip mgca_kinetx_full_service_seed43_code.zip
cd mgca_kinetx_full_service

# 核对路径。此路径沿用此前 case-study 的默认 ESM2 路径。
export ESM2_PATH=/root/private_data/DP/pretrained_model/esm2_t36
bash run_train_and_package.sh
```

运行顺序：核对冻结代码/数据/ESM2 -> 串行全量特征提取 -> seed=43 单任务重拟合 -> 验证checkpoint重载 -> 生成带一个checkpoint的部署ZIP。设备和路径可在 `config.sh` 修改；不允许改科学配置JSON来绕过哈希。

```bash
# 中断后直接重跑同一命令，将从最后一个完整epoch继续。
bash run_train_and_package.sh

# 单独准备特征 / 训练 / 打包
bash run_train_and_package.sh --phase prepare
bash run_train_and_package.sh --phase train
bash run_train_and_package.sh --phase package
```

seed=43 保存逐epoch loss/系数/耗时、全量训练预测、最终checkpoint、last_state（optimizer和RNG）及哈希清单。中断从最后完整epoch恢复；已验证完成则跳过。未完成run保留失败日志。OOM不会自动改batch size或科学配置。

输出示例：

```text
outputs/checkpoints/seed_43/mgca_fullKinetX_seed_43_epoch55.pt
outputs/deploy/<bundle_id>/mgca_kinetx_api.zip
```

部署ZIP包含seed=43的一个最终权重、冻结模型/配置、服务脚本、完整清洗数据蛋白的预计算特征、API复现检查数据。不包含 optimizer、last_state、训练CSV、日志等中间文件。代码包本身没有尚未训练的新权重。

默认不复制大型ESM2。部署到原服务器或已有相同ESM2文件的机器即可直接引用。如需把ESM2也打进ZIP，在打包时设置（会明显增大文件和磁盘需求）：

```bash
INCLUDE_ESM=1 bash run_train_and_package.sh --phase package
```

每个 bundle_id 对应独立内容，不覆盖之前部署包。保留训练输出用于续跑/审计，但不必全部上传到服务机器。

## 2. 一键启动服务（启动中自动预热）

把训练完成的 `mgca_kinetx_api.zip` 放到目标机器，解压到一个新目录：

```bash
unzip mgca_kinetx_api.zip
cd mgca_kinetx_api

# 在已有成功运行MGCA的Python环境中，仅增加服务依赖。
python -m pip install -r requirements_service.txt
export ESM2_PATH=/root/private_data/DP/pretrained_model/esm2_t36
bash start_service.sh
```

不主动升级torch/CUDA/Transformers/RDKit等模型环境。模型依赖沿用成功运行的环境（torch1.12.1+cu113、transformers4.40.1、RDKit2024.03.5等），原工具文件也会导入 numpy/pandas/scipy/sklearn/matplotlib/seaborn/tqdm。仅包住推理逻辑，不需PyMOL/Vina/绘图系统库。

服务前台启动，看到 `READY` 和 Uvicorn `Application startup complete` 后才可请求。可用 tmux 或已有服务管理器常驻；不自动创建系统服务、改防火墙或暴露公网。

预热包含：

1. 核对所有checkpoint/代码/ESM2哈希。
2. ESM2与seed=43的一个MGCA模型加载到指定设备，之后常驻，不按请求重载。
3. 载入训练时计算的全部蛋白特征（LRU默认可容纳1024种蛋白）。
4. 执行一次真实ESM2前向，以及batch1和配置batch大小的5模型前向，预热GPU计算。

单Uvicorn worker，GPU请求串行执行；正在推理时其他请求返回429，客户端退避重试。不设置多个worker重复装载大模型。启动哈希检查和加载需要时间；**预热不意味着从未见过的新蛋白无需ESM编码**。新蛋白第一次请求编码一次，后续重复使用缓存。服务重启会恢复训练蛋白缓存；其他新蛋白/药物LRU仅存在内存，不无限增长磁盘。

## 3. 健康检查、数值复现与筛选

另开终端：

```bash
curl http://127.0.0.1:8000/readyz
python verify_api.py

# 本包内真实样例：一条蛋白和一个已知输入分子。
curl -X POST http://127.0.0.1:8000/screen \
  -H 'Content-Type: application/json' \
  --data-binary @examples/screen.json
```

`verify_api.py` 将seed=43的API输出与训练后打包时计算的CPU参考预测逐行比对（绝对容差1e-4，容许CPU/GPU浮点差异）。不通过时检查环境和checkpoint，不能据此调模型。

`POST /screen` 请求格式（用真实完整序列替换示意值）：

```json
{
  "fasta": "ACDEFGHIKLMNPQRSTVWY",
  "smiles": ["CCO", "CCN"],
  "top_k": 2,
  "allow_truncation": false
}
```

结果按pKoff从高到低排序（预测解离越慢越靠前），包含seed=43的预测pKoff、变换后的koff、原始索引ID和编码/推理耗时。`top_k`只裁剪返回结果，所有候选都先打分。不输出跨seed标准差或集成均值。

`POST /predict` 支持不同蛋白-药物对并保持输入顺序：

```json
{"pairs":[{"sample_id":"pair1","fasta":"ACDEFGHIKLMNPQRSTVWY","smiles":"CCO"}],"allow_truncation":false}
```

输入不需要标签。非法序列、SMILES或重复sample_id返回422，整个请求失败，不静默删行或修复分子。默认拒绝长度超过1022残基的序列；显式 `allow_truncation=true` 才按冻结tokenizer max_length1024（含特殊token）截断并标记。训练则保留原实验同样的截断政策，并在features_manifest统计。

## 4. 大库筛选（客户端分批，保存所有结果）

准备一个单记录FASTA文件 `target.fasta` 和逐行SMILES文件 `library.smi`：

```bash
python client_screen.py --fasta target.fasta --smiles library.smi \
  --batch-size 512 --output screening_results.csv
```

客户端自动分批调用 `/predict`，返回429/503时有限次数退避，最后全库排序保存CSV；不会只保存每批top-k。`.progress.json`只报告进度，不是可续跑预测文件；客户端中断需重新调用（服务已缓存该蛋白，重算会较快）。输出已存在会拒绝覆盖。

## 5. 端口、安全和性能配置

默认绑定127.0.0.1:8000。远程访问建议SSH转发：

```bash
ssh -L 8000:127.0.0.1:8000 用户@服务器
```

确需局域网绑定，必须在shell设置API_KEY；不要把密钥写进ZIP：

```bash
export API_KEY='替换为随机长密钥'
HOST=0.0.0.0 PORT=8000 bash start_service.sh
# 客户端传 X-API-Key，client_screen.py/verify_api.py自动读取API_KEY环境变量。
```

这是受信任环境中的研究筛选服务，不是经过安全审计的公网多租户产品；公网需要TLS反向代理、限流和网络访问控制。默认关闭Swagger/OpenAPI公开页面。

在 `service_config.sh` 调整 INFER_BATCH_SIZE（默认64）、MAX_PAIRS（每请求最多4096）、MAX_PROTEIN_CACHE（1024）、MAX_DRUG_CACHE（10000）。请求体默认限制8MiB；OOM不自动更改模型精度/batch。服务机器需足够显存常驻fp32 ESM2及一个MGCA，实际延迟/吞吐量必须以目标机器实测，不承诺固定毫秒数。

pKoff=-log10(koff/s^-1)。本服务只有一个seed=43 checkpoint，不提供集成方差、已校准不确定性或预测区间。模型已用全量KinetX训练，不能再将其在原KinetX test上的表现报告为独立测试结果。

服务生命周期参考 FastAPI 官方文档：https://fastapi.tiangolo.com/advanced/events/ ；单worker内存注意事项：https://fastapi.tiangolo.com/deployment/concepts/ 。
