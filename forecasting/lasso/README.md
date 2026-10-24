# lasso

实现类型：**官方库调用 + 本地模型/CP 适配**。

来源：[https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)。

使用 sklearn Lasso；本地 ARX72+CP，不是原论文 LEAR。CP 公式参考 https://github.com/lipiecki/PostForecasts.jl，提交 589f9407d375e2272fb731873443f691c0060d85。

配置：[configs/lasso/aligned.yaml](../../configs/lasso/aligned.yaml)。实验模型 ID：`lasso_arx72_cp`。

代码入口：[model.py](model.py)。共享数据、训练调度和评分仍由 `forecasting/aligned_*.py` 提供。目录整理不改变输入、损失、模型参数、选择规则或评分定义。

## 论文链接与引用

本地 ARX72+CP，使用 sklearn Lasso；不是原论文 LEAR。本地逐 horizon 对称绝对残差校准，使用 type-7 分位插值；Lei 等仅提供分割共形预测的方法背景，具体公式参考 PostForecasts.jl。插值规则与时间序列非交换性意味着这里不主张该文的有限样本覆盖保证。

可复制的引用格式：[BibTeX](CITATIONS.bib)。作者、年份、出处及核对来源见 [source.json](source.json)，模型代码开头也列出引用。

- **Lasso 原论文**：Tibshirani, Robert (1996). Regression Shrinkage and Selection Via the Lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267–288. [论文链接](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) · BibTeX key：`tibshirani1996lasso`。
- **分割共形预测的方法背景**：Lei, Jing; G'Sell, Max; Rinaldo, Alessandro; Tibshirani, Ryan J.; Wasserman, Larry (2018). Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523), 1094–1111. [论文链接](https://doi.org/10.1080/01621459.2017.1307116) · BibTeX key：`lei2018distributionfree`。
- **绝对残差 CP 公式的软件来源**：Lipiecki, Arkadiusz; Weron, Rafał (2025). PostForecasts.jl: A Julia package for probabilistic forecasting by postprocessing point predictions. SoftwareX, 31, 102200. [论文链接](https://doi.org/10.1016/j.softx.2025.102200) · BibTeX key：`lipiecki2025postforecasts`。

引用信息核对日期：2026-10-23。引用方法或组件不代表复现了原论文的完整实验。
