# ddnn_jsu

实现类型：**方法重实现**（依据论文公式；官方公开仓库无 JSU 源码，仅有 Normal 头）。

分布深度网络 + Johnson SU 参数头：MLP 输出每 horizon 的 (γ, δ, ξ, λ)（δ、λ 经 softplus+ε 保证严格为正），以精确 JSU NLL 训练于共享标准化价格空间；五个公共分位来自解析逆 CDF（支持负价），逆变换回原始 RRP。检查点按声明口径以选择期**原始价格公共五分位 QS** 选择（与原文 NLL 选择方式的差异已在 source.json 与运行 metadata 中声明）。

- 输入块映射：七块公共输入按 INPUT_KEYS 展平（与树模型同展平），无丢弃块。
- 网络形态参照官方 ddnn_normal.py：Kaiming 隐层/Xavier 输出、可选 BN/dropout（第一版 hidden [512,512]、无 BN、dropout 0）、Adam。
- 限制：位置参数 ξ 不声称等于中位数；点指标用真实分布中位数；未复现原文超参搜索。

配置：[configs/ddnn_jsu/aligned.yaml](../../configs/ddnn_jsu/aligned.yaml)。实验模型 ID：`ddnn_jsu`。来源与差异：[source.json](source.json)。
