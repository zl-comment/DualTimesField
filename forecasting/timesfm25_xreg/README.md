# timesfm25_xreg — TimesFM-2.5 + 官方 XReg 基线（第二批 2026-10-23）

冻结 google/timesfm-2.5-200m-pytorch（timesfm 2.0.2 官方 2.5 实现，非 3.0）+
官方 `forecast_with_covariates` XReg 外生路径。权威记录见 [source.json](source.json)。

- XReg 为逐 origin 在其合法 72h 价格历史上做最小二乘（in-context 拟合，非
  train split、非"无拟合"）；协变量效应等量加回所有原生通道。
- 原生通道为 [mean, q10–q90]，无 q05/q95 → **P 表**（点=原生中位通道+XReg），
  原生通道另存辅助诊断。
- 映射：日历（96 动态）+ gas（静态）；需求未来仅存在不同标准化的预报 vintage，
  拼接被拒；仅未来外生块不可映射。
- `infer_is_positive=False`（官方开关；NEM 存在负价，默认开启会造成选择性截断）。
- 环境：[ENVIRONMENT.md](ENVIRONMENT.md)、[requirements.lock.txt](requirements.lock.txt)。
