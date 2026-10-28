G0-A结果：NSW1 predispatch_rrp 本地 as-of 预审
日期：2026-10-28 UTC

判定
ENGINEERING_PASS：定向测试34项真实退出码0；预算墙钟0.35秒（上限300秒）；
网络禁用与路径守卫无违规；schema/单位/目标时间/run-vintage/publication_basis全部通过。
STRICT_ASOF_BLOCKED：结构性恒成立。source_last_changed_unix是archive_proxy
（MMSDM月度归档LASTCHANGED，NPZ构建于2026-10-10抓取），不能证明历史first-seen，
无法排除带回填LASTCHANGED的事后修正；本G0-A未启动任何前瞻采集。
quick_veto TRIGGERED：选中12个origin中4个存在尾部ffill填充
（覆盖23/24、21/24、20/24、18/24），这些行的未覆盖小时是最后一个已覆盖小时的重复。
当前NPZ对"无填充as-of使用"构成快速否决；本NPZ单独永远给不出strict G0 PASS。

版本与顺序
基点708327098d1e50349726f76f55a10893ba11173d（research/agent-evidence-contract-g0-20261028，
验证时工作区干净、无本项目活动实验）。
独立分支research/predispatch-rrp-asof-g0-20261028。
冻结提交7255d5d：协议、配置、守卫实现与34项测试先行提交。
修复提交67b9bb9：冻结测试attempt1（退出码1，3失败）暴露测试夹具缺陷
（合成origin基值未对齐小时网格），仅修复夹具，模块/协议/配置语义未变。
修复提交81f5785：attempt2（退出码1，1失败）与audit attempt1暴露assemble_verdict
把覆盖填充误计入engineering门禁，与冻结协议第4节矛盾；按协议修复verdict门禁。
正式测试与审计均在81f5785上执行；run_manifest.json记录全部执行源码哈希。
两次失败尝试日志与audit attempt1判定原样保留于本目录（preflight_tests.attempt*.txt/.exit、
g0a_verdict.attempt1.json），真实退出码未用管道掩盖。

冻结范围执行情况
仅NSW1；仅data/aemo_exogenous_holdout/merged/nsw1_predispatch.npz的predispatch_rrp；
origin选择=等距下标规则（(k*(n-1)+5)//11，k=0..11），只消费origin数组，与覆盖无关。
12个选中origin跨2015-01-01 05:00至2026-09-01 03:00 AEST。
未读取actual.npy/target.npy或任何标签；无网络（运行时socket禁用）；无模型调用；
无训练/GPU；无跨vintage拼接、尾部延伸、补零（审计只读）；无G0-B/G0-C/G1/G2；
无最终测试选择；未推送；未启动72小时采集。

六项检查结果
schema：pass（n=102263，键/形状/dtype齐全，origin严格逐小时，单origin单run序号）。
单位：pass（有限且在[-1000,20000] AUD/MWh内；选中行min=-4.11、max=791.46；
  血缘=冻结builder 8ef49cdf…读取PREDISPATCHPRICE.RRP；仅为包络否决级，不构成单位证明）。
目标时间：pass（origin在小时网格；value[h]↔origin+(h+1)小时结束AEST，血缘绑定builder）。
run/vintage：pass（12个origin的source_run_unix均≤origin，run年龄0.46-0.48小时，
  序号各异即每次为新vintage；无stale、无未来泄漏）。
24小时覆盖：fail-as-finding（8个24/24；4个尾部ffill：23/24、21/24、20/24、18/24
  ——凌晨起报的run只发布到次日04:00，剩余小时重复最后已覆盖小时，与builder文档一致）。
publication_basis：pass-as-archive-proxy（source_last_changed_unix逐元素等于
  source_run_unix，即builder别名；一律标archive_proxy，不宣称historical first-seen）。

资源
单进程CPU-only，审计墙钟0.35秒（上限300秒）；pytest 2.85秒；
网络0字节；GPU 0；训练0；模型调用0；新包安装0。

可重放命令（仓库根目录）
  .venv/bin/python -m pytest tests/test_g0a_predispatch_rrp.py -q \
      > outputs/forecasting/predispatch_rrp_asof_g0_20261028/preflight_tests.txt 2>&1; \
      echo $? > outputs/forecasting/predispatch_rrp_asof_g0_20261028/preflight_tests.exit
  .venv/bin/python scripts/audit_g0a_predispatch_rrp.py record-tests
  .venv/bin/python scripts/audit_g0a_predispatch_rrp.py audit
  .venv/bin/python scripts/audit_g0a_predispatch_rrp.py archive

边界声明
本审计只证明结构完整性、血缘绑定与填充/vintage事实；不含任何预测、评分或性能证据。
"covered<24"行的数值不可用于无填充as-of特征；如需strict as-of证明，必须另行前瞻采集
（本G0-A明确未启动）。outputs/与data/在.gitignore中；本目录为提交归档。
