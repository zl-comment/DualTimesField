# 第二批基础模型基线：阶段0来源、能力与公平性审计（2026-10-23）

分支 `research/foundation-baseline-expansion-20261023`，worktree
`/root/workspace/Code/DualTimesField-foundation-baselines`，基点
`19ba700d5988c07083ba7751a31ab853744e94af`。机器可读能力表：
[capability_table.json](../logs/forecasting/foundation_baselines_20261023/capability_table.json)；
权重 SHA256 清单：`weights/weight_identities.json`（worktree 内，不入 Git）。

## 一、共享协议与父运行复核（实测）

- 父运行 `/root/workspace/Code/DualTimesField/outputs/forecasting/aligned_baselines_20261012`
  只读复用；实测三地区各 split 起点数 61273/4321/4393/17521，与冻结样本声明一致。
- 实测 `flatten_inputs` 宽度 **1465**（72×2 + 72×10 + 24×10 + 1 + 24×11 + 24×1 + 24×3）；
  提示词块表中 `future_exogenous` 标注 [24,11]，与实测一致。
- 协议段（data/forecast_protocol/七个外生段/aligned_protocol/model/training）从第一批
  扩展配置逐字复制，`assert_shared_protocol` 对父配置仍然成立；第一批注册表、矩阵
  校验与冻结运行未被修改（有回归测试）。

## 二、逐模型审计结论

### A1 tabpfn_v3 —— Q 表，全量公共输入

- **实现身份**：`tabpfn` PyPI **9.2.0**（包版本≠模型代际）。默认构造器解析到 v3.5
  权重（`ModelVersion.V3_5`，constants.py:38-39 实测）；本适配器显式固定
  `Prior-Labs/tabpfn_3` 的 `tabpfn-v3-regressor-v3_default.ckpt`
  （revision `24a16a89d245…`，232,679,299 字节）。
- **本地权重访问**：官方包自带交互式许可下载门（`TabPFNLicenseError`，账号入口
  https://ux.priorlabs.ai，`TABPFN_TOKEN`）。本实现不使用该下载器、不传任何数据：
  权重从公开 HF 仓库一次下载到 worktree 共享缓存，适配器直接传本地路径。许可
  `tabpfn-3-license-v1.0`（研究允许）。无 token 存储；如机构政策要求官方流程，
  缺失项仅为该账号接受步骤与 token。
- **输入映射**：七块全量 1465 列（与树模型同一展平），24 个 horizon 独立回归任务。
- **原生分位**：`predict(output_type='quantiles', quantiles=[.05,.1,.5,.9,.95])`，
  由预测分布 CDF 精确给出，五级全原生。
- **上下文限制**：文档上限 1M 样本×2000 特征；本任务 61273×1465 在限内，**全量
  train 上下文可用，无子采样，无 ctxN 受限组**。
- **成本实测（L40）**：fit 7.6s/horizon；4096-query 批 22.2 ms/query，峰值 7.5GB；
  每格（17521×24 任务）约 2.6 小时，9 格约 23.5 GPU 小时。
- **seed**：控制集成预处理 RNG（random_state=seed+horizon）；同 seed 两次推理
  **逐位相同**（实测 max diff 0）；跨 seed 真实不同（max diff 24.3 实测）→ 三 seed
  为同权重不同集成抽样，三格独立推理。
- **预训练来源**：Prior Labs V3 合成+开放数据混合（arXiv:2605.13986）；真实数据集
  与 NEM 重叠**未审计**；权重 2026-04/05 发布，晚于 2023–2024 测试期。发布晚≠已证
  污染，资料不足≠证明无污染；候选报告中以 `pretraining_overlap_unaudited` 标注。

### A2 chronos2_zeroshot —— Q 表，受限输入

- **实现身份**：`chronos-forecasting` PyPI **2.3.2** 的 `Chronos2Pipeline`；权重
  `amazon/chronos-2`（revision `29ec3766d36d…`，455MB，Apache-2.0，120M 参数）。
  非 Chronos-T5/Bolt/synth 变体；冻结、无微调。
- **原生分位（关键核查）**：检查点 config.json 训练 21 级 [.01,.05,…,.99]，**五个
  公共级全部是训练级别**，`predict_quantiles` 直接选取、无插值。包内插值路径仅用于
  训练范围外级别（我们不请求）。`use_arcsinh=true` 为检查点原生变换。
- **输入映射**：价格历史逆标准化一次→原始 RRP 目标（输出即原价，一次逆变换）；
  需求(标准化单位)为 past 协变量；日历 10 特征 past+future。**无法合法映射**：
  origin_context（无静态槽位）、future_exogenous/ctf/quantile_exogenous（官方要求
  协变量同时有历史段，缓存中无、不得编造）→ `input_scope=restricted`，不与全列
  模型作等输入比较。
- **上下文限制**：模型默认 8192；显式 context_length=72、prediction_length=24、
  cross_learning=False（每起点独立组，无跨序列注意力）。
- **成本**：smoke 24 起点 9.1s；正式每格预计 <1 小时量级（GPU）。
- **seed**：无作用（冻结权重、无采样）→ 三 seed 复用验证子集后复制（reuse_of 记录）。
- **预训练来源**：Chronos 数据集 + GIFT-Eval pretrain 子集 + 合成（模型卡）；NEM
  重叠未审计；2025 发布晚于测试期。

### B1 moirai2_zeroshot —— P 表，受限输入

- **实现身份**：PyPI `uni2ts` **2.0.0 不含 Moirai-2 类**（实测 import 失败）；改为
  官方 GitHub 仓库固定 commit `cfd46d4510ed8896f263116f32928eede05b0a75`
  （与代码中既有 blocker 记录一致）。权重 `Salesforce/moirai-2.0-R-small`
  （revision `30f43ff08c84…`，43MB，11.4M 参数，cc-by-nc-4.0，仅研究用途）。
- **原生分位（关键核查）**：检查点 config.json `quantile_levels` 恰为
  **[0.1…0.9]**，输出为 QuantileForecastGenerator 的分位头，**无 q05/q95、无分布/
  采样路径**。按协议不外推、不复制端点、不补高斯尾：**进 P 表**（点=原生 q50），
  原生 q10/q90 另存辅助诊断，不进五分位评分。模型 ID 保留 `moirai2_zeroshot`
  （其确能接收协变量，非 price-only），受限范围在 registry/source.json/产物声明。
- **输入映射**：价格历史（逆标准化一次）past_target；需求 past_feat_dynamic_real
  （past-only 槽位）；日历合并 feat_dynamic_real（96×10，past+future 段）。origin_context
  与仅未来块无槽位 → restricted。
- **接口事实（实测）**：raw `forecast.predict` 会把所有数组裁剪/填充到 context_length，
  与 future 段协变量语义冲突（tensor 62≠82 实测报错）；官方示例路径为
  `create_predictor`（gluonts TFTInstanceSplitter 正确切分 past/future），本适配器
  采用后者。
- **上下文限制**：max_seq_len 512 token、patch 16；context 72。
- **seed**：无作用（dropout 构造器强制 0）。

### B2 timesfm25_xreg —— P 表，受限输入 + in-context XReg

- **实现身份**：`timesfm` PyPI **2.0.2**，内含 `TimesFM_2p5_200M_torch` 与
  `timesfm/utils/xreg_lib.py`，即官方 2.5 实现；PyPI 3.0.x 跟踪 TimesFM 3.0，**不用**
  （防版本漂移）。权重 `google/timesfm-2.5-200m-pytorch`（revision `1d952420fba8…`，
  882MB，0.2B 参数，Apache-2.0）。
- **原生分位（关键核查）**：通道恰为 `[mean, q10…q90]`（`self.q = len(quantiles)+1
  # 10`、decode 循环 index 1..9 实测）→ **无 q05/q95**：进 P 表，点=原生中位通道
  +XReg（库的点返回），原生 10 通道另存辅助。mean 通道存在但不用。
- **XReg 官方路径（关键核查）**：`forecast_with_covariates` 的
  `BatchedInContextXRegLinear` 在**每个 origin 自己的合法 72h 历史**上拟合最小二乘
  （ridge=0，逐序列目标归一化，无子采样），预测期协变量效应**等量加回所有分位通道**
  （位移）。这是标签拟合步骤（用该起点合法历史价格），**不是"无拟合"**，已按此登记；
  不触 train/selection/calibration 标签（registry fit_requires_train_labels=False）。
- **输入映射**：价格历史（逆标准化一次）；日历 96h 动态协变量（XReg 需历史+未来
  段）；origin_context 为静态数值协变量。**需求不可作动态协变量**：未来需求仅存在于
  不同标准化器的预报 vintage（predispatch），拼接=混合统计变量，记录后拒绝；仅未来
  块同 Chronos 拒绝 → restricted。
- **官方开关选择（固定配置，正式前冻结）**：`infer_is_positive=False`（NEM 有负价，
  默认开启会对 72h 窗非负的起点单侧截断原生输出——选择性隐藏截断）；`torch_compile
  =False`（离线 worker 无 JIT 依赖、断点续跑确定性）；`normalize_inputs=True`（官方
  逐序列归一化，已记录）；`fix_quantile_crossing=True`、`force_flip_invariance=True`、
  `use_continuous_quantile_head=True`、`return_backcast=True`（XReg 必需）为官方默认。
- **XReg extras**：`timesfm[xreg]==2.0.2` 需显式安装（否则 ImportError，实测）。

## 三、分组与已知限制

| 模型 | 表 | 输入组 | 上下文组 | 预训练重叠 |
|---|---|---|---|---|
| tabpfn_v3 | Q5 | full_public（1465 列） | full_train_context（61273） | unaudited |
| chronos2_zeroshot | Q5 | restricted（价格+需求past+日历） | origin_own_72h | unaudited |
| moirai2_zeroshot | P | restricted（同上） | origin_own_72h | unaudited |
| timesfm25_xreg | P | restricted_xreg（+gas 静态） | origin_own_72h_incontext_fit | unaudited |

四个检查点均发布于 2025–2026，晚于 2023–2024 测试期；各模型卡所述训练数据与 NEM
的重叠均未提供可审计证据。候选对照可列，但**不宣称严格无污染的新盲测**。

## 四、环境与可复现性

- 四个独立 venv（.venvs/{tabpfn_v3,chronos2,moirai2,timesfm25}，Python 3.11.17），
  各目录 requirements.lock.txt + ENVIRONMENT.md。宿主驱动 560.35.05（CUDA 12.6）
  无法运行 torch cu130 轮子：tabpfn/chronos/timesfm 用
  `--index-url …/whl/cu126`（2.14.1+cu126），uni2ts 按其约束用 2.4.1+cu121。
- 权重统一缓存于 worktree `weights/hf_home`（HF_HOME），正式 worker 以
  `HF_HUB_OFFLINE=1` 运行，不自动更新依赖或权重；无权重入 Git。
- GPU：每卡最多一个 worker、默认两张空闲卡（config gpu_ids [6,7]，占用阈值 4096MiB
  门控）；未终止任何他人进程。

## 五、阻断与待办

- 无硬阻断：四个模型权重均可本地合法访问、依赖齐备、真实权重 smoke 路径打通
  （见阶段2记录）。
- TabPFN 官方交互许可流程未走（不需要；如机构要求，见上文精确入口与缺失项）。
- 未做项（按提示词范围外）：TabPFN-v2 对照、TabPFN-TS、Chronos-2 微调、NBEATSx/TFT、
  Mitra、2025/2026 新时间轨道、168h 扩展。
