# 当前主表基线的论文与引用

覆盖当前主表的 15 个静态基线。每个 `model.py` 开头提供论文链接与普通参考文献格式；对应目录的 `CITATIONS.bib` 提供可复制 BibTeX，`source.json` 记录用途、核对日期和证据页面。

组合基线分别引用骨干、校准方法及直接参考的软件；相关比较论文单独标注。Linear 与 MLP 为本地通用对照，其背景文献不视为本地完整模型的原论文。

| 主表模型 ID | 代码内引用 | BibTeX | 引用内容 |
|---|---|---|---|
| `lightgbm` | [lightgbm/model.py](lightgbm/model.py) | [CITATIONS.bib](lightgbm/CITATIONS.bib) | LightGBM 原论文 |
| `lightgbm_cp` | [lightgbm_cp/model.py](lightgbm_cp/model.py) | [CITATIONS.bib](lightgbm_cp/CITATIONS.bib) | LightGBM 基座原论文；分割共形预测的方法背景；绝对残差 CP 公式的软件来源；相关电价比较论文（J34），非完整复现 |
| `xgboost` | [xgboost/model.py](xgboost/model.py) | [CITATIONS.bib](xgboost/CITATIONS.bib) | XGBoost 原论文 |
| `xgboost_cp` | [xgboost_cp/model.py](xgboost_cp/model.py) | [CITATIONS.bib](xgboost_cp/CITATIONS.bib) | XGBoost 基座原论文；分割共形预测的方法背景；绝对残差 CP 公式的软件来源；相关电价比较论文（J34），非完整复现 |
| `lightgbm_qra` | [lightgbm_qra/model.py](lightgbm_qra/model.py) | [CITATIONS.bib](lightgbm_qra/CITATIONS.bib) | LightGBM 基座原论文；QRA 原论文；相关电价比较论文（J34），非完整复现 |
| `patchtst` | [patchtst/model.py](patchtst/model.py) | [CITATIONS.bib](patchtst/CITATIONS.bib) | PatchTST 原论文；RevIN 组件原论文 |
| `gatets_adapter` | [gatets/model.py](gatets/model.py) | [CITATIONS.bib](gatets/CITATIONS.bib) | GateTS 原论文 |
| `timexer` | [timexer/model.py](timexer/model.py) | [CITATIONS.bib](timexer/CITATIONS.bib) | TimeXer 原论文 |
| `linear` | [linear/model.py](linear/model.py) | [CITATIONS.bib](linear/CITATIONS.bib) | Linear/DLinear 对照设计的相关论文 |
| `dlinear` | [dlinear/model.py](dlinear/model.py) | [CITATIONS.bib](dlinear/CITATIONS.bib) | DLinear 原论文 |
| `mlp` | [mlp/model.py](mlp/model.py) | [CITATIONS.bib](mlp/CITATIONS.bib) | 前馈网络反向传播的经典背景文献 |
| `pricefm` | [pricefm/model.py](pricefm/model.py) | [CITATIONS.bib](pricefm/CITATIONS.bib) | PriceFM 原始工作（预印本） |
| `ddnn_jsu` | [ddnn_jsu/model.py](ddnn_jsu/model.py) | [CITATIONS.bib](ddnn_jsu/CITATIONS.bib) | 电价 DDNN/Johnson SU 原论文；J34 比较论文及本地网络形态参考 |
| `lasso_arx72_cp` | [lasso/model.py](lasso/model.py) | [CITATIONS.bib](lasso/CITATIONS.bib) | Lasso 原论文；分割共形预测的方法背景；绝对残差 CP 公式的软件来源 |
| `nhits_qra` | [nhits_qra/model.py](nhits_qra/model.py) | [CITATIONS.bib](nhits_qra/CITATIONS.bib) | NHITS 骨干原论文；MC-dropout 方法论文；QRA 原论文；NHITS+QRA 电价比较论文（P08）；相关校准讨论（P16） |

合并后的文献库：[paper/manuscript/references.bib](../paper/manuscript/references.bib)。各目录引用复用该文献库的同名条目，避免重复导入；正文是否引用某篇论文仍由对应的 LaTeX `\cite{...}` 决定。

核对规则：优先采用正式期刊/会议出版页；预印本使用 arXiv 记录。TimeXer 使用正式会议页的九位作者；RevIN 使用作者项目页所列 ICLR 2022；QRA 使用正式卷期年份 2015。PriceFM 的首次提交年份与代码固定版本分别记录。

本次仅新增引用说明与文献记录，不改动模型计算、训练预算、数据处理、已有实验档案或主表分数。
