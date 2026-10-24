# timexer

实现类型：**官方核心源码 + 本地适配**。

来源：[https://github.com/thuml/TimeXer](https://github.com/thuml/TimeXer)。

保留官方主干；加入未来已知变量 token，输出改为五分位数。历史路径有同权重数值比较测试。

配置：[configs/timexer/aligned.yaml](../../configs/timexer/aligned.yaml)。实验模型 ID：`timexer`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

原始代码与固定提交记录位于 [vendor/PROVENANCE.md](vendor/PROVENANCE.md)；`vendor` 中的原始源码快照保留校验值。

## 论文链接与引用

固定官方主干，增加已知未来变量 token 与五分位头；作者列表以 NeurIPS 正式出版页为准。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **TimeXer 原论文**：Wang, Yuxuan; Wu, Haixu; Dong, Jiaxiang; Qin, Guo; Zhang, Haoran; Liu, Yong; Qiu, Yunzhong; Wang, Jianmin; Long, Mingsheng (2024). TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables. Advances in Neural Information Processing Systems 37, 469–498. [论文链接](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) · BibTeX key：`wang2024timexer`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
