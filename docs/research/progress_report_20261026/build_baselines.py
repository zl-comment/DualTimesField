import json
from pathlib import Path

ROOT=Path('/root/workspace/Code/DualTimesField')
OUT=Path('/root/workspace/Code/research_progress_report_20261026')
models=[]
def add(id,title,refs,*paragraphs):
    models.append(dict(id=id,title=title,refs=refs.split(),paragraphs=list(paragraphs)))

add('linear','Linear：线性结构对照','zeng2023transformers',
'Zeng 等（2023）的 LTSF-Linear 研究指出，简单线性映射可成为长序列预测的有力基线。本研究的 Linear 是据此构建的本地通用对照，并没有对应的独立“Linear 原论文”。其核心为 q̂τ=Wτp历史+Aτz已知+bτ，以历史价格的线性时间映射加上已知外生信息的线性 adapter，直接产生 24 个时距的五分位预测。',
'本地实现沿用 DLinear 的参数布局，但关闭移动平均分解；公共训练使用分位损失，而不是照搬原论文的点预测实验。它用于检验复杂表示是否真正超越历史价格、日历及需求等线性可解释信号。对尖峰而言，线性模型能够外推，但无法独立表达稀缺状态下的阈值触发及高阶交互，且尾部事件稀少时会被常态样本主导。')
add('dlinear','DLinear：趋势与残差分解','zeng2023transformers',
'Zeng、Chen、Zhang 和 Xu（2023）提出的 DLinear 先用移动平均得到趋势 T=MA(p)，再令季节/残差项 S=p−T，分别线性预测并相加：ŷ=WT T+WS S。其价值是以很少参数将慢变水平和较快波动分开，从而检验 Transformer 的复杂度是否必要。',
'本地采用长度 25 的移动平均，端点重复填充，增加公共外生 adapter 和五分位输出头；属于参考方法重实现，尚未完成官方权重映射等价验证。该结构适合电价日内波动与水平漂移的廉价建模，但移动平均不包含经济意义的供需状态，历史尖峰可能只表现为残差。若未来极端价格由新出现的稀缺机制触发，线性分解本身不能保证峰值幅度与发生时点准确。')
add('mlp','MLP：全输入非线性对照','rumelhart1986learning',
'MLP 为本地通用前馈网络，背景引用 Rumelhart、Hinton 和 Williams（1986）的反向传播工作；该文解释如何通过链式法则把输出误差传回隐层以学习内部表示，并非本项目电价网络的完整原始设计。当前结构将七块公共输入展平为 z，经单隐层 GELU 后输出 24×5 个数，即 q̂=W₂ GELU(W₁z+b₁)+b₂。',
'本地默认隐层宽度为 64，全部公共信息都进入网络，没有双场模块、显式趋势分解或后置分布假设。这个对照有助于判断改进究竟来自额外输入、一般非线性还是所提出的结构。MLP 可学习价格与外生变量的非线性关系，但缺少时间结构归纳偏置和尖峰专属机制；对少见尾部状态的学习依赖样本覆盖，可能获得较好整体损失而仍漏报极端峰值。')
add('lasso_arx72_cp','Lasso ARX72 + CP：稀疏线性预测与残差区间','tibshirani1996lasso lei2018distributionfree lipiecki2025postforecasts',
'Tibshirani（1996）的 Lasso 通过最小化 (1/2n)‖y−Xβ‖²+λ‖β‖₁ 同时进行回归收缩与变量选择。本研究把 72 小时历史和公共外生特征用于逐时距线性回归，选择正则强度后，以样本外校准残差生成分位区间。这是本地 ARX72+CP，不应称作完整复现电价文献中的 LEAR。',
'共形预测的方法背景引用 Lei 等（2018）；本地具体公式来自 Lipiecki 与 Weron（2025）的 PostForecasts.jl：q̂τ=ŷ+sgn(τ−0.5)·Quantile|2τ−1|(|y校准−ŷ校准|)，逐时距以 type-7 插值估计残差分位。中位数始终是 Lasso 中心，区间关于中心对称。它可作为低成本、可解释的概率对照，但不能修正尖峰中心的系统性低估，也不能刻画上下尾不对称；由于时间重叠样本不满足交换性且使用插值分位，此实现不主张有限样本覆盖保证。')
add('lightgbm','LightGBM：梯度提升树分位回归','ke2017lightgbm',
'Ke 等（2017）提出 LightGBM，以梯度提升树逐步拟合残差/负梯度，预测可表示为 F(x)=Σmηfm(x)。原论文的核心效率机制包括按梯度大小抽样的 GOSS 与互斥特征捆绑 EFB；这些是库的算法背景，不等于本地实验显式启用了每一个机制。树的分裂可以表达需求、日历、历史价格等变量的阈值与交互。',
'本地调用官方 lightgbm 库，公共输入统一展平，逐预测时距分别训练五个分位 booster，并以该时距的公共五分位评分联合选择迭代轮数；不是原论文数据集实验复现。分位损失为 ρτ(u)=u[τ−1(u<0)]。它是识别非线性电价状态的重要强基线，但叶节点输出不易外推到训练样本未覆盖的峰值幅度，且没有专门约束跨时距轨迹或极端状态发生机制。')
add('xgboost','XGBoost：正则化树提升分位回归','chen2016xgboost',
'Chen 与 Guestrin（2016）提出 XGBoost，目标由经验损失与树复杂度正则项组成：L=Σiℓ(yi,F(xi))+ΣmΩ(fm)，以梯度/曲率信息指导逐步加树，并设计稀疏感知分裂、加权分位草图及工程优化。与简单线性模型相比，提升树能够表示分段非线性和变量交互。',
'本地使用官方 XGBRegressor，逐时距训练，采用 hist 树方法和 reg:quantileerror 直接预测五个分位，迭代前缀由共同验证指标选择。分位目标是当前库支持的任务配置，并非把 2016 年原论文视作五分位电价模型。该基线可以识别价格抬升的历史阈值条件，但在稀缺状态样本不足、突变机制变化和极端幅度外推上仍受限制；需要将整体评分与尖峰子集评分共同考察。')
add('lightgbm_cp','LightGBM + CP：冻结树中心的对称校准','ke2017lightgbm lei2018distributionfree lipiecki2025postforecasts lebedev2026uncertainty',
'该组合将 LightGBM 的非线性条件中心与绝对残差校准分开：先冻结原树模型，再在 2022H2 样本外残差上按时距估计区间宽度。Lei 等（2018）提供分割共形方法背景，PostForecasts.jl 提供本地采用的绝对残差/type-7 公式；Lebedev 等（2026，J34）是电价不确定性比较背景，不能被称作该组合的完整复现。',
'具体为 q̂τ=q̂base,0.5+sgn(τ−0.5)Q|2τ−1|(|y−q̂base,0.5|)。五分位重排后的基座 q50 在校准前后逐位相同，原始冻结预测复现校验也是运行门槛。本方法直接回答“复杂 Anchor 是否超越廉价区间修正”。它能纠正总体区间过窄，却不能改善中位预测和由此定义的 MAE，也不会根据当前稀缺程度动态改变同一时距宽度；尖峰条件覆盖与时间序列有限样本覆盖均不能由总体残差校准自动保证。')
add('xgboost_cp','XGBoost + CP：更换基座的校准归因对照','chen2016xgboost lei2018distributionfree lipiecki2025postforecasts lebedev2026uncertainty',
'XGBoost+CP 采用与 LightGBM+CP 完全一致的校准流程，唯一主要变化是冻结基座换成 XGBoost。原论文谱系分别来自 Chen 与 Guestrin（2016）的树提升、Lei 等（2018）的分割共形背景及 Lipiecki 与 Weron（2025）的软件公式；J34 提供在电价任务上比较不确定性方法的研究背景。',
'逐时距以 2022H2 的绝对误差经验分位形成对称五分位输出，测试阶段冻结宽度且 q50 等于冻结 XGBoost 的 q50，不新增输入也不重新训练基座。该设计能检验校准收益是否依赖某一种树算法。它与 LightGBM+CP 的共同局限是仅调整区间、无法消除中心偏差；若尖峰误差集中于少数供需失衡状态，固定残差宽度会面临常态过宽与尖峰仍不足的矛盾，不能把总体覆盖率改善直接解释为尖峰预测解决。')
add('lightgbm_qra','LightGBM + QRA：允许中心移动的线性后校准','ke2017lightgbm nowotarski2015qra lebedev2026uncertainty',
'Nowotarski 与 Weron（2015）提出 QRA，将多个点预测作为解释变量，通过分位回归形成组合概率预测：q̂τ=βτ,0+Σkβτ,k f̂k，系数以 pinball 损失估计。它同时具有预测平均和分位校准作用，是电价概率预测的重要低成本基线；J34 属于相关不确定性比较文献。',
'本地组合并不使用多个独立模型点预测，而用单个冻结 LightGBM 的排序后五分位及截距构成六维设计矩阵，在 2022H2 对每个时距、每个分位独立拟合无正则精确线性规划（alpha=0，HiGHS），测试期冻结系数。与 CP 不同，QRA 可以改变 q50，因此能修正中心偏差并改变区间不对称性。但它仍是有限预测特征的线性映射，校准期稀少尖峰、外推和制度变化可导致系数不稳定；其收益必须与复杂 Anchor 在相同校准信息预算下对比。')
add('patchtst','PatchTST：局部片段表示与 Transformer','nie2023time kim2022reversible',
'Nie 等（2023）的 PatchTST 将连续时间点切成 patch，再把片段映射为 token，使用共享权重的通道独立 Transformer 建模较长历史。片段化保留局部形态并减少注意力序列长度。RevIN（Kim 等，2022）对每个样本先归一化、输出后再恢复尺度，缓解训练与测试分布变化。',
'本地重写 RevIN、patch、注意力层与预测头：价格通道进入 Transformer，外生信息通过线性 adapter 加入，输出为五分位；未完成与官方权重映射的数值等价验证。主要路径可写为 q̂=RevIN⁻¹[Head(Transformer(Patch(RevIN(p))))]+A(z)。它适合学习价格的局部波形与周期关系，但外生交互仅由 adapter 表达，过去波形也未必包含下一次供给约束的信号；因此对孤立尖峰及机制转移的优势需要实测，不能由长序列基准成绩直接推断。')
add('gatets_adapter','GateTS adapter：注意力启发路由与专家混合','yemets2025gatets',
'Yemets、Lukashchuk 和 Izonin（2025）的 GateTS 面向单变量时序，利用注意力启发的路由机制选择专家，基本输出形式为 h=Σe∈TopK g_e(x)E_e(x)。其研究重点是在减少额外负载均衡训练负担的同时，让不同专家刻画不同序列形态；报告沿用对应固定代码的 2025 预印本引用。',
'本地保存官方固定提交骨干，将输出拓宽为 24×5，并额外使用线性已知输入 adapter，故主表 ID 明确为 gatets_adapter。实现会先计算全部专家再施加 top-k 输出掩码，不能直接照搬论文的稀疏计算速度收益。专家分工可能有助于区分平稳与高波动历史，但路由没有显式供需机制监督，外生量仅以加性方式进入，可能在罕见尖峰状态上路由不稳；是否形成有效“尖峰专家”需要额外诊断。')
add('timexer','TimeXer：内生片段与外生变量交叉注意力','wang2024timexer',
'Wang 等（2024）的 TimeXer 专门研究带外生信息的时间序列预测。内生价格被切成时间片段进行自注意力，外生序列按变量编码；可学习的全局内生 token 通过变量级交叉注意力吸收外生信息，再向价格片段传递。其核心注意力算子为 Attention(Q,K,V)=softmax(QKᵀ/√d)V。',
'本地保留官方骨干，新增已知未来变量 token 与五分位输出头；历史路径已进行同权重数值比较。历史需求、日历与未来已知日历/外生量由明确分开的合法输入路径接入，不使用未来价格标签。这使其成为“充分利用外生信息是否足够”的重要基线。它具有学习需求与价格非线性耦合的能力，但注意力相关性本身不等于可识别的市场因果机制，也不自动保证上尾校准和稀缺状态的可靠外推。')
add('pricefm','PriceFM：电价领域 MoE 与图结构迁移的局部实现','yu2025pricefm',
'Yu 等的 PriceFM（2025 首投，2026 修订）针对欧洲跨区域电价概率预测，将价格与负荷、风光预测等外生变量映射到共享专家表示，并利用输电拓扑形成图约束以聚合区域信息。其研究动机是将市场联系与电力网络知识纳入领域基础模型。',
'本地固定 2026 年 MoE 代码，从 TensorFlow 移植到 PyTorch，采用历史/未来两条线性投影、softmax 稠密专家加权和图池化；围绕 q50 逐级加减非负增量构造不交叉分位。关键差异是每地区单节点、从头训练，没有多区域预训练或空间迁移，官方 TensorFlow 等价验证尚未通过。因此该行测试的是 PriceFM 架构适配，而不是原文预训练基础模型能力。专家和层级分位有助于非线性和分位一致性，但单节点弱化了原文跨区网络优势，对尖峰的帮助不能借用原文结论。')
add('ddnn_jsu','DDNN-JSU：条件偏态厚尾分布建模','marcjasz2023distributional lebedev2026uncertainty',
'Marcjasz 等（2023）将分布神经网络用于电价预测，网络不只输出均值，而是输出条件分布参数；Johnson SU 能表示偏度、厚尾和负价。J34（Lebedev 等，2026）进一步比较统计与神经方法的不确定性，本地网络形态参考其配套 Normal DDNN，但 JSU 由本地按公式重实现。',
'本地令 W~N(μw,σw²)，Y=ξ+λsinh(W)，网络逐时距输出四参数，λ、σw 为正；训练使用精确负对数似然，分位为 qτ=ξ+λsinh(μw+σwΦ⁻¹(τ))。这说明 ξ 一般不等于中位数，点指标必须用真实 q50。七块公共输入全部展平，检查点按原价公共 QS 选择而非原文 NLL，且未复现大规模调参。',
'当前有效版本用 float64 解析逆 CDF，不裁剪 sinh 参数或预测，非有限值报错；早期 ±12 钳位运行只作历史审计。JSU 可表达上尾厚尾，比对称 CP 更灵活，但单一分布族可能无法表达多状态混合，尾部参数也可能放大极端预测；需分别检验概率评分、覆盖、尾部误差和数值稳定性。')
add('nhits_qra','NHITS + MC-dropout + QRA：多尺度骨干与后校准','challu2023nhits gal2016dropout nowotarski2015qra lettner2026performance lettner2026calibration',
'Challu 等（2023）的 NHITS 通过多速率池化、层级插值及残差回投，在不同频率尺度逐块重建历史并累加预测。Gal 与 Ghahramani（2016）的 MC-dropout 在预测时重复开启 dropout，把多次前向结果作为模型不确定性的近似样本；QRA（Nowotarski 与 Weron，2015）再对预测特征拟合条件分位。Lettner 等（2026，P08）把 NHITS+QRA 与基础模型比较，P16 则讨论电价概率校准挑战。',
'本地移植官方 NHITS 多尺度骨干，保留 MaxPool、插值、残差回投及 naive1 水平初始化，七块输入分别映射到历史、未来及静态外生槽位。先以 MAE 训练并选择点模型，再固定种子做 K=20 次 MC-dropout，仅 dropout 层处于训练模式；MC 样本经验分位作为 QRA 特征，在 2022H2 做无正则 LP 校准，测试期冻结。',
'这不是 P08/P16 全部实验流程的复现。多尺度结构适合不同时间尺度的波动，但池化和插值可能平滑短促尖峰；MC-dropout 的离散度也不等价于完整数据噪声。QRA 可以改善分位校准，却依赖有限采样与校准期尾部事件的代表性，因此该组合是较强的概率预测对照而非尖峰可靠性的自动保证。')
add('tabpfn_v3','TabPFN-3：冻结表格基础模型与上下文回归','priorlabs2026tabpfn3 hollmann2025tabpfn',
'Grinsztajn 等（2026）的《TabPFN-3: Technical Report》是当前 V3 检查点对应主引用；Hollmann 等（2025）的 Nature 论文属于 V2 谱系背景。TabPFN 在合成数据任务上预训练，使模型在一次上下文推断中利用带标签支持样本预测查询样本的条件分布，形式为 pθ(y查询|x查询,D支持)，不需要为每个任务更新权重。',
'本地显式固定 Prior-Labs/tabpfn_3 的 V3 regressor 检查点，避免包默认构造器解析到其他代际；七块公共输入展平为 1465 列，按 24 个时距建立独立回归任务。train 标签全部作为支持上下文，不使用 selection/calibration 标签，权重保持冻结；因此“无训练权重”不能写成“完全不接触任务标签”。其原生分位包含公共五级，纳入 Q 表。该模型能提供强表格非线性对照，但时间轨迹被拆成独立任务，极端市场状态的泛化仍取决于合成先验与支持集覆盖。')
add('chronos2_zeroshot','Chronos-2：带协变量的零样本时序基础模型','amazon2025chronos2',
'Ansari 等（2025）的《Chronos-2: From Univariate to Universal Forecasting》通过组注意力在相关序列、变量及协变量之间共享信息，使单变量、多变量与协变量预测都可在冻结模型中完成；合成多变量结构训练支持这种上下文学习能力。当前引用必须与 Chronos-2 对应，而非 Chronos-T5 或 Bolt。',
'本地固定 amazon/chronos-2，逐起点输入合法 72 小时价格历史、历史需求、过去/未来日历，预测未来 24 小时；cross_learning=False，未微调或后校准。检查点原生 21 个分位级别包含所需五级，进入 Q 表。但 origin_context 和三组仅未来外生块因官方接口缺少合法历史段而未接入，所以它属于受限输入组，不是与完整公共输入模型等信息比较。原生非线性变换与预训练可能帮助尺度适应，但未校准的上尾与不完整外生稀缺信号仍可能限制尖峰预测。')
add('moirai2_zeroshot','Moirai 2.0：解码器式多分位零样本模型','woo2025moirai2',
'Liu 等的《Moirai 2.0: When Less Is More for Time Series Forecasting》（2025 首投，2026 修订）采用 decoder-only Transformer、单一 patch 及分位损失，通过多 token 预测和递归多分位解码提高效率。它不同于早期 Moirai 的掩码编码器与混合分布输出，也不同于 Moirai-MoE。本报告保留本地 bib key 以便追溯，但纠正其错误作者与标题。',
'本地使用固定 uni2ts 提交和 Salesforce/moirai-2.0-R-small 权重，输入价格历史、历史需求和跨过去/未来的日历；缺失 origin_context 与仅未来外生块，属受限输入零样本对照。原生分位只有 0.1 至 0.9，故仅 q50 进入共同点预测 P 表，q10/q90 另作诊断，不外推 q05/q95，也不计算统一五分位 QS。该模型可展示通用先验的迁移能力，但不能用缺失公共尾部分位的输出证明极端区间质量，尖峰表现仍需共同点指标和事件分析。')
add('timesfm25_xreg','TimesFM-2.5 + XReg：残差基础模型与逐起点回归','das2024decoder',
'Das、Kong、Sen 和 Zhou（2024）的 TimesFM 工作以输入 patch 和 decoder-only 预训练实现通用时间序列预测；本地实际检查点为 Google 的 2.5 代 200M PyTorch 版本，版本能力需同时以官方模型卡说明，不能把 ICML 2024 原版等同于 2.5。',
'本地官方 XReg 路径在每个起点的合法 72 小时上下文上拟合回归，先扣除协变量效应，再由 TimesFM 预测残差，最后等量加回各原生通道：q̂τ=X未来β̂+q̂TimesFM,τ(残差历史)。为避免跨起点共享拟合，逐起点独立调用；这属于上下文拟合，并非“完全无拟合”。日历可合法接入，静态气价虽传入却被逐起点中心化消除，需求及仅未来外生块未接入。',
'模型只有 mean 与 q10–q90 原生通道，因此共同表仅使用原生中位数 P 指标。当前发布配置 ridge=0，维护候选不能提前替换主表；负价任务关闭正值推断截断。短上下文的冗余日历设计、回归外推及受限输入均可能影响尖峰，已有秩与算子诊断只提示敏感性，不能断言全部异常由数值问题造成。')

refs={}
for p in (ROOT/'forecasting').glob('*/source.json'):
    for r in json.loads(p.read_text()).get('references',[]):
        refs[r['bibtex_key']]={'key':r['bibtex_key'],'citation':r['citation'],'url':r['paper_url'],'verified_date':'2026-10-26','verification':'本地引用及来源记录；以下 verification_urls 标明本轮独立核查证据'}
def ref(key,citation,url):
    refs[key]={'key':key,'citation':citation,'url':url,'verified_date':'2026-10-26','verification':'本轮已打开一手出版页/arXiv记录核对'}
ref('priorlabs2026tabpfn3','Grinsztajn, Léo; Flöge, Klemens; Key, Oscar; et al. (2026). TabPFN-3: Technical Report. arXiv:2605.13986v2.','https://arxiv.org/abs/2605.13986')
ref('hollmann2025tabpfn','Hollmann, Noah; Müller, Samuel; Purucker, Lennart; Krishnakumar, Arjun; Körfer, Max; Hoo, Shi Bin; Schirrmeister, Robin Tibor; Hutter, Frank (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637, 319–326.','https://www.nature.com/articles/s41586-024-08328-6')
ref('amazon2025chronos2','Ansari, Abdul Fatir; Shchur, Oleksandr; Küken, Jaris; et al. (2025). Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821.','https://arxiv.org/abs/2510.15821')
ref('woo2025moirai2','Liu, Chenghao; Aksu, Taha; Liu, Juncheng; Liu, Xu; Yan, Hanshu; Pham, Quang; Savarese, Silvio; Sahoo, Doyen; Xiong, Caiming; Li, Junnan (2025; revised 2026). Moirai 2.0: When Less Is More for Time Series Forecasting. arXiv:2511.11698v3.','https://arxiv.org/abs/2511.11698')
ref('das2024decoder','Das, Abhimanyu; Kong, Weihao; Sen, Rajat; Zhou, Yichen (2024). A decoder-only foundation model for time-series forecasting. Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 10148–10167.','https://proceedings.mlr.press/v235/das24c.html')
verified={
 'ke2017lightgbm':'https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html',
 'chen2016xgboost':'https://arxiv.org/abs/1603.02754',
 'zeng2023transformers':'https://ojs.aaai.org/index.php/AAAI/article/view/26317',
 'nie2023time':'https://arxiv.org/abs/2211.14730',
 'yemets2025gatets':'https://arxiv.org/abs/2508.17515',
 'wang2024timexer':'https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html',
 'yu2025pricefm':'https://arxiv.org/abs/2508.04875',
 'challu2023nhits':'https://ojs.aaai.org/index.php/AAAI/article/view/25854',
 'gal2016dropout':'https://proceedings.mlr.press/v48/gal16.html',
 'lettner2026performance':'https://arxiv.org/abs/2604.14739',
 'lebedev2026uncertainty':'https://arxiv.org/abs/2509.19417',
 'marcjasz2023distributional':'https://api.crossref.org/works/10.1016/j.eneco.2023.106843',
 'lettner2026calibration':'https://api.crossref.org/works/10.1145/3765611.3815392',
 'lei2018distributionfree':'https://api.crossref.org/works/10.1080/01621459.2017.1307116',
 'rumelhart1986learning':'https://api.crossref.org/works/10.1038/323533a0',
 'tibshirani1996lasso':'https://api.crossref.org/works/10.1111/j.2517-6161.1996.tb02080.x',
}
for key,url in verified.items():
    refs[key]['verification_urls']=[url]
    refs[key]['verification']='本轮已核对官方出版/arXiv记录或出版商登记的 Crossref 元数据'

corrections=[
 {'key':'priorlabs2026tabpfn3','correction':'本地将作者写为 Prior Labs、标题写为 Machine Learning with Zero Hyperparameter Tuning；arXiv 实际题为 TabPFN-3: Technical Report，首作者 Léo Grinsztajn。报告纠正，repo未修改。','url':refs['priorlabs2026tabpfn3']['url']},
 {'key':'hollmann2025tabpfn','correction':'本地 V2 谱系引用的作者及卷页有误；按 Nature 正式页更正为 Hollmann 等8位作者，637:319–326；此文不是V3检查点主引用。','url':refs['hollmann2025tabpfn']['url']},
 {'key':'amazon2025chronos2','correction':'本地 Amazon 公司署名和 Agentic Era 标题不是所链接论文元数据；更正为 Ansari 等，Chronos-2: From Univariate to Universal Forecasting。','url':refs['amazon2025chronos2']['url']},
 {'key':'woo2025moirai2','correction':'本地 Woo 等作者和 Unified Training 标题均与 arXiv:2511.11698 不符；更正为 Chenghao Liu 等，Moirai 2.0: When Less Is More for Time Series Forecasting。仅保留旧key便于追溯。','url':refs['woo2025moirai2']['url']},
 {'key':'das2024decoder','correction':'本地ICML论文作者列有误；正式出版页为 Abhimanyu Das、Weihao Kong、Rajat Sen、Yichen Zhou。2.5版本另以官方模型卡为证。','url':refs['das2024decoder']['url']},
]
used=list(dict.fromkeys(k for m in models for k in m['refs']))
data={'models':models,'references':[refs[k] for k in used], 'corrections':corrections,'verified_date':'2026-10-26',
 'version_sources':[{'model':'timesfm25_xreg','url':'https://huggingface.co/google/timesfm-2.5-200m-pytorch','note':'本轮打开，确认2.5-200m模型卡与2025-10-02 fused-QKV更新；具体固定版本/本地能力边界依据source.json和运行实现。'}],
 'caveats':[
 '以上尖峰适用性与限制是基于论文机制和本地实现作出的分析，不是未经实验验证的性能结论。',
 '公共原论文、组件论文、组合方法来源和本地适配必须区分；19个模型不是19篇独立原始模型论文。',
 '本轮独立核查纠正5条基础模型/谱系元数据；其余部分出版页因访问受限，使用出版商登记Crossref或本地已核查记录。RevIN、QRA及PostForecasts条目本轮未成功打开正式出版正文，不声称已逐页复核。',
 'GateTS arXiv页已有2026 Neurocomputing关联DOI；报告为与固定代码及本地key一致仍引用2025预印本，不混称期刊版实验。',
 '四个冻结基础模型的预训练数据不能在本项目内全面审计；零样本描述仅指本地未用训练split更新权重/拟合任务，不构成预训练无重叠证明。TabPFN明确使用train支持标签，TimesFM XReg明确使用起点历史拟合。',
 '历史 Informer、iTransformer、Full/no_fields 不属于当前19基线清单；若另列历史结构对照须独立注明。']}
(OUT/'baselines.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
print('wrote',OUT/'baselines.json','models',len(models),'references',len(used),'chars',sum(len(p) for m in models for p in m['paragraphs']))
