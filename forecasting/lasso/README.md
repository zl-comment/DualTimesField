# lasso

实现类型：**官方库调用 + 本地模型/CP 适配**。

来源：[https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)。

使用 sklearn Lasso；本地 ARX72+CP，不是原论文 LEAR。CP 公式参考 https://github.com/lipiecki/PostForecasts.jl，提交 589f9407d375e2272fb731873443f691c0060d85。

配置：[configs/lasso/aligned.yaml](../../configs/lasso/aligned.yaml)。实验模型 ID：`lasso_arx72_cp`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。
