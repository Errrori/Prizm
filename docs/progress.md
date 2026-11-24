# progress.md — 当前进度

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：新会话的交接仪表盘，只描述"项目现在在哪"。
> - 写法：每次任务收尾**全文重写**，但必须保留下方固定章节结构，不许增删章节。
> - 篇幅：全文 ≤ 60 行。只写现状，不写历史（历史在 git log 里）。
> - 禁令：技术债细节不写在这里（只在"阻塞与风险"里引用 DEBT 编号）；
>   功能想法不写在这里（进 backlog.md）；禁止修改本格式契约。

---

## 当前阶段
MVP、150 题评测集、流式反馈、真实模型全量复跑与答案一致性提优均已完成；需求 §4 用户自带 Key 输入经评估废弃，不再实施

## 已完成
- 任务 0-1：项目骨架，pytest 可运行
- 任务 0-2a/0-2b：评测 JSONL 格式、指标计算、10 题最小评测集
- 任务 1-1：配置 + CSV connector + DuckDB 只读执行引擎
- 任务 1-2：LLM adapter（mock 默认 / litellm 切换）+ planner + orchestrator
- 任务 1-3/1-4：Chainlit 上传→提问→SQL/结果；按 provider 取 key、显式 api_base
- 任务 2-0/2-1/2-2：额度台账、P0 profiler、澄清门禁（异常场景召回 100%）
- 任务 3-1/3-2：P1 自修复（总尝试 ≤3）、评测执行入口
- 任务 4-1：多轮上下文，数据集只加载一次并注入历史摘要
- 任务 5-1/5-2：RAG 数据字典 + 50 条术语评测（引用溯源 50/50）
- 任务 6-1：P2 透明面板：Trace/TraceStep/ProfileReference、Chainlit 嵌套 Step + 侧栏面板
- 任务 7-1：Excel(.xlsx) / SQLite connector + 上传端多 sheet/多表选择
- 任务 8-1：结构化 chart spec + matplotlib PNG + LLM 结论（模板兜底）
- 任务 9-1：真实模型评测 70% 一次成功 / 90% 自修复后；SQLite 额度台账；README + Dockerfile
- 任务 10-1：评测集扩至 150 题，多轮会话串接，异常检出率指标接通
- 任务 11-1/11-2：流式过程反馈；真实模型 150 题复跑（一次 83.3%、自修复后 86.0%）
- 任务 12-1：失败归因与答案一致性提优——新增 `--ids/--diagnostics`；planner 约束月份 YYYY-MM、差值方向 A-B、单行单列、COUNT(*)、CTE 最终 SELECT；repair 按错误类型加提示；全量 150 题一次 98.0%（147/150）、自修复后 99.3%（149/150）、异常 100%、澄清 100%；修正 tra-043 标准答案 1744→949
- 任务 13-1：需求 §4 用户自带 Key 输入评估后废弃，需求/架构/README/债务台账已同步移除或标注取消（DEBT-027）

## 进行中
- 无

## 下一步
- 无（后续按新任务安排）

## 阻塞与风险
- 非阻塞：ord-021 偶发差值只返回单边值（1/150，DEBT-026）
- 非阻塞：DEBT-023 LLM 结论未纳入评测；DEBT-022 图表 50 点上限、无多序列
- 非阻塞：DEBT-021 Excel/SQLite 仅单表；DEBT-019 RAG 模板问法；DEBT-016/017 路由与问答简化
- 非阻塞：DEBT-015 多轮未缓存可查询结果；DEBT-013 比对仍为启发式（本轮未发现误判）
- 非阻塞：DEBT-012 只读校验非完整 AST；DEBT-009/010/011 澄清启发式与回退
- 非阻塞：DEBT-007/008 主键与脏数值启发式；DEBT-003 CSV 编码回退；DEBT-006 费用依赖 provider 字段

## 关键入口
- 需求：docs/requirements.md ｜ 架构：docs/architecture.md
- 技术债：docs/debts.md ｜ 待办：docs/backlog.md
- 测试：`uv run pytest`（本机 `.venv\Scripts\python.exe -m pytest`，当前 346 项）｜ 启动：`uv run chainlit run app/main.py`
- 容器：`docker build -t prizm .` 后 `docker run --rm -p 8000:8000 --env-file .env -v prizm-data:/data prizm`
- 评测：`uv run python -m eval.build_tasks`、`-m eval.run_eval --execute`、`-m eval.run_clarify`、`-m eval.run_rag`
- 失败归因：`python -m eval.run_eval --execute --ids "id1,id2" --diagnostics eval/outcomes/diag.jsonl`
