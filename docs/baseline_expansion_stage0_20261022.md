# 阶段0状态：公平基线扩展工作区核对（2026-10-22）

分支 `research/fair-baseline-expansion-20261022`，HEAD `33bc769`（提示词提交），父提交 `1e962ea`。
独立 worktree `/root/workspace/Code/DualTimesField-fair-baselines`；`/root/workspace/Code/DualTimesField`（Codex 活动 worktree，`research/v5-anchor-calibration-fix`）未被切换分支、修改或安装依赖。

## 公共协议字段（以冻结产物为准）

- `protocol_id: aligned-baselines-20261012-static-v1`；stage `full_fixed_configuration_rerun`。
- 七块输入（INPUT_KEYS 顺序）：history_values 72×2、history_calendar 72×10、future_calendar 24×10、origin_context 1、future_exogenous 24×11、ctf_exogenous 24×1、quantile_exogenous 24×3；flatten 宽度 1465（与冻结 LightGBM manifest 的 feature_count 一致）。
- 切分：train 61,273 / selection 4,321 / calibration 4,393 / test 17,521 个 origin（NSW1 实测；test 与原冻结样本数一致）。
- 目标标准化为 train 拟合 z-score（NSW1 mean=70.4597, std=164.8734）；预测逆变换回原始 RRP。

## 环境与资源

- 独立 venv：`.venv`（Python 3.11.0）。版本镜像 Codex 环境冻结版本：torch 2.5.1+cu124、numpy 2.4.6、pandas 3.0.6、scikit-learn 1.9.1、lightgbm 4.7.0、xgboost 3.2.0、PyYAML 6.0.3，另装 pytest/hydra-core/omegaconf/matplotlib/tqdm。未向 Codex `.venv` 写入任何内容。
- GPU 实测：6/7 空闲，0–2 各有 10–19 GiB 占用但利用率 0%，4/5 接近满载。正式调度按实际占用动态判断，不假设固定空闲。

## 冻结产物可用性（只读）

- 父运行 `outputs/forecasting/aligned_baselines_20261012`（符号链接，只读）：90/90 格 completed，含 `runs/{model}/seed{S}/{region}/` 的 models/、test.npz、metrics.json、status.json 与 data_cache（四 split 全量）。
- 扩展运行 `aligned_baselines_20261013_extension`（符号链接，只读）。
- 公共评分器复算：lightgbm seed2026 NSW1 MAE 44.3633237737048、xgboost 45.145121408791454，与冻结 metrics.json 逐位一致。
- **冻结基座复现门槛（P0-1/2/3 的前提）已验证**：
  - LightGBM：120 个 booster 文件/地区 + manifest `selected_rounds`，`num_iteration=selected_rounds[h]` 预测 test，与冻结 test.npz 未排序分位输出 max|diff| = 0.0（逐位一致）。
  - XGBoost：24 个 ubj/地区 + `iteration_range=(0, selected_rounds[h])`，同样逐位一致。
  - 因此校准段（2022H2，4,393 origins）的冻结树样本外预测可由 booster 忠实再生成，且每格运行时都会重放该 test 复现校验作为硬门槛。

## 每模型实现身份（第一批）

| 模型 | 身份 | 基座 | 关键来源 |
|---|---|---|---|
| lightgbm_cp | 本地适配 | 冻结 LightGBM（复用，不重训） | CP 公式沿用仓库内 `forecasting/lasso/model.py::absolute_residual_cp`（PostForecasts.jl 589f940，绝对残差 + type-7），与 J34（IEEE 11430173）建议的树+CP 对照同型 |
| lightgbm_qra | 本地适配 | 冻结 LightGBM 5 分位预测 | QRA（分位回归平均）逐 horizon 线性分位回归；J34 建议，非完整论文复现 |
| xgboost_cp | 本地适配 | 冻结 XGBoost | 与 lightgbm_cp 共用 `common/calibration.py`，不复制公式 |
| ddnn_jsu | 方法重实现 | 从零训练 | J34（lebeand/uq-epf）公开代码仅含 Normal 参数化 DDNN；JSU 参数化按论文公式本地实现（JSU 分位/ICDF 与 NLL），网络形态（Kaiming/ Xavier 初始化、可选 BN/dropout、Adam）参照其 ddnn_normal.py |
| nhits_qra | 官方库适配 | 从零训练 | NHITS 骨干（Nixtla neuralforecast，MIT）+ 固定种子 MC-dropout + 冻结骨干后 QRA（P08/P16）；数据 frame 仅作 adapter 表示，不使用其日期切分 |

## 需要扩展的最少接口（阶段1实施清单）

1. `forecasting/common/calibration.py`：`absolute_residual_cp` 自 `lasso/model.py` 提取（原导入与数值行为不变）+ 新增 QRA 拟合工具。
2. `forecasting/common/frozen_base.py`：冻结 LightGBM/XGBoost 基座加载、逐位复现校验、校准段预测再生成、booster 哈希记录。
3. `forecasting/common/baseline_registry.py`：第一批 5 模型的能力注册（输出表、是否需校准段、资源、原生训练、基座依赖、依赖可用性）。
4. `forecasting/aligned_experiment.py::validate_config`：single_model 白名单加入注册的扩展模型（唯一改动点，向后兼容；matrix/extension 语义不变，旧测试不破坏）。
5. `forecasting/aligned_expansion.py`：新增扩展实验入口（prepare/run/worker），父运行审计复用 `audit_parent`，data_cache 只读符号链接，worker 经注册表分派；报告复用 `build_aligned_report`，另交付跨 run 合并报告器。
6. 新矩阵配置 `configs/experiments/aemo_forecast_aligned_expansion_20261022.yaml` 与五个 `configs/<model>/aligned.yaml`（共享节与父矩阵逐项相等）；新增 `tests/test_expansion_model_layout.py`，不改旧 `test_model_layout.py` 的 MODELS 与旧断言。

已知限制：外生发布时间为档案代理；2023–2024 为回顾性测试；三 seed 非独立时间序列证据。全程不修改 Full/Anchor 实现与既有结果。
