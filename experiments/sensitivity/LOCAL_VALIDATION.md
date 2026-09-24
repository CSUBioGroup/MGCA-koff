# 本地交付验证：2026-09-09

- 9 项规划/完整性/统计/汇总单元测试通过。
- 可选 PyTorch 模型测试通过：19 个设置（含两个等上限对照）均使用原模型类和原编码器类；初始化、有限 forward/backward、参数量一致性通过；参数量 13,392,395。
- 合成缓存特征的实际训练器恢复集成测试通过：第 7 epoch 中断后继续训练、第 12 epoch 早停 checkpoint 后中断恢复，均与不中断训练的最佳模型逐 tensor 一致；后者没有额外训练一轮。
- 上述 PyTorch 测试使用本机 CPU PyTorch，不代表服务器 CUDA/ESM2 全流程已执行。恢复集成测试中的输入、评估指标提供器和环境采集使用测试 fixture；模型、优化器、损失、RNG、checkpoint、训练日程和导出逻辑为实际副本。
- Python 3.8 语法兼容检查通过；两个 Bash 脚本通过 `bash -n`。
- 基于实际本地数据成功生成 510 个任务、17 个唯一配置、44 个 train/val 文件的计划；计划预览未放入代码压缩包。
- 使用 RDKit 对实际 22 套 train/validation 做只读检查：原始 pair 不重叠，drug-cold canonical drug 不重叠，protein-cold exact sequence 不重叠；10 套 warm 的 canonical 等价 pair 重叠保留为披露事项。没有打开 test 文件。
- 原 v10 模型 SHA256：e6efef46abec20d1b25305c69ca099e2a5f84f6906f7bcb3f9b1e6c1786d7f7b。
- 原 v10 训练器 SHA256：743fa1db439454984468128be23a34be21ed93abc12dae6a4a78b9258be2f8c5。二者未被修改。
- 包内原始来源与副本差异哈希见 source_provenance.json；全部交付文件哈希见 release_manifest.json。
- 尚未执行真实 ESM2/GPU 端到端预检查或 510 次敏感性训练；服务器运行时会执行相应检查，不提供虚构实验结果。
