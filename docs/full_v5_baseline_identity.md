# V5 baseline identity — frozen artefact reuse, not a retraining reproduction

本文件区分 V5 所依赖的基线身份层次。**V5 是冻结产物复用并补算开发预测，不是本轮从头重训复现**；当前目录重构版本不能自动等同于原规模运行已重新验证。

## 承载层（本轮工作分支）

- 分支：`research/v5-calibrated-history-20261015`
- 起点 commit：`b03155f26d716511a37c00eaf7f0bd2e3a9d3b3e`（与 V4 归档时相同的树）
- V5 新增代码/配置/测试/文档的全部哈希在 prepare 时写入 `identity/identity.json` 并在其后每步复核。

## 原 LightGBM 实现身份（第一轮审计锁定）

| 项目 | 身份 |
|---|---|
| 原始实现 commit | `70352de817d422ca9789208d0deb418dcee9e448` |
| 原源码快照 SHA256（47 文件） | `6f67f928f9689fb48e9943a1f0ddc4db797a324d1fa8cbd4aaf5c1cc48c650dd` |
| 原矩阵配置 SHA256 | `59acc7cda472bb556e34475d2b77d91007ddaffde03371476bf6b5fd4871fb86` |
| 当前单模型配置 | `configs/lightgbm/aligned.yaml`（SHA256 `7afbcd26ce5c94f4c8418114501f0ed34ef90ef31944c848f2316617afac42ba`） |
| 原运行 | `outputs/forecasting/aligned_baselines_20261012`（jobs.json、data_cache、runs/lightgbm/seed*/） |

启动记录中的 `git_head` 是前一提交 `af546fa…`；**不能只拿该 HEAD 复现**，须用上表 `70352de` 快照。原运行的关键结构：每地区每 horizon 按五分位共同 pinball 选择共享轮数（非逐分位早停）、最多 2000 rounds、patience 50、lr 0.05、num_leaves 31、min_child_samples 50、subsample/colsample 0.8。

## 本轮实际参数锚点

- **配对锚点 seed 2026**；seeds 2026/2027/2028 的三 seed 成绩仅作背景参考。
- 模型目录：`outputs/forecasting/aligned_baselines_20261012/runs/lightgbm/seed2026/{region}/models/`（每地区 120 booster = 24 horizon × 5 分位）。prepare 时由 `FrozenLightGBMPredictor` 逐文件哈希校验并记录 provenance（manifest SHA、selected_rounds、包版本）。
- 2022H2 预测为**本轮补算**（同一冻结 booster + 冻结缓存，纯 CPU 推断），并与 `outputs/forecasting/lightgbm_event_diagnosis_20261013/predictions/seed2026/{region}/calibration.npz` 的既有独立补算交叉核对（差异写入 identity）。
- 2023–2024 test 预测直接读取原运行冻结产物 `runs/lightgbm/seed2026/{region}/test.npz`，仅 final-test 阶段可打开。

## 原始数据与缓存（绝对路径在仓库外）

- 原始小时 CSV SHA256：NSW1 `454ac6013d13f351c6c1edf5dde832b66a441639a9f35414c8de0d23a8c8c08f`；QLD1 `29bfc9d1c02fc78baac77d145d3313132174559f4a3acc05ba7b42f1317d400f`；TAS1 `260df3603a382c1173566b089913b82586cb28d09a0efa17e289b91029178045`。
- 缓存清单 `data_cache/{region}/hashes.json` SHA256：NSW1 `0c6fbd7b057e32e31871731a334b8e57b83632536add1890c4fd8526997fdbb1`；QLD1 `a33f55f883c47ad78f8ee013780947f0ca4dab13ee81e36ab6ce4eff5b54d5f5`；TAS1 `5e3109a52388a9b872ffd8e794fb37c8dfdfb4dd36fd011c62626db421ce80e4`。
- seed 2026 test 预测 SHA256：NSW1 `639568368494182adea5aea0dedf5f373840c7887ae3df5bb69786d8e951dcdc`；QLD1 `e0122a237e10fc9df00665d236cadb850b08f582e6527f9c1d38554eb4c7ee7c`；TAS1 `79c663eefb59edc9c908f97510ec86a92b0e731241b29f20b2e63a683b8792f5`。
- 其余十个外生原始文件的绝对路径与哈希在 `jobs.json:data_hashes`，部分指向兄弟目录 `DualTimesField-max-spare`。**单独复制本仓库源码不足以复现**；prepare 会逐一重算这些哈希。
- 发布时间映射：`available_at = delivery_start + 1h` 为数据构建时强制校验的档案代理；prepare 重验该不变量。档案时间戳是可用性代理，未经独立认证的真实首次发布时间。

## 依赖版本

jobs.json 环境：Python 3.11.0、NumPy 2.4.6、PyTorch 2.5.1+cu124、LightGBM 4.7.0、XGBoost 3.2.0、scikit-learn 1.9.1；本轮 venv 另核实 pandas 3.0.6、SciPy 1.17.1。LightGBM/XGBoost 不在 requirements.txt，版本以 jobs.json 与模型 manifest 为准。

## V4 对照产物身份

A_stack 只读复用 `outputs/forecasting/full_v4_20261014/runs/v4_full/seed2026/{region}/{calibration,test}_predictions.npz`（公共预测契约 schema，origins 与缓存网格逐元素一致在加载时断言；文件哈希写入 identity）。V4 的训练/选择历史见其自身归档；本轮不重跑、不修改。
