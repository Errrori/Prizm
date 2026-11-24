# progress.md — 当前进度

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：新会话的交接仪表盘，只描述"项目现在在哪"。
> - 写法：每次任务收尾**全文重写**，但必须保留下方固定章节结构，不许增删章节。
> - 篇幅：全文 ≤ 60 行。只写现状，不写历史（历史在 git log 里）。
> - 禁令：技术债细节不写在这里（只在"阻塞与风险"里引用 DEBT 编号）；
>   功能想法不写在这里（进 backlog.md）；禁止修改本格式契约。

---

## 当前阶段
MVP、150 题评测集、流式反馈与真实模型复跑已完成；剩余演示 GIF 与自修复成功率提优

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
- 任务 9-1：真实模型评测 70% 一次成功 / 90% 自修复后；SQLite 额度台账；litellm 不读 .env；README + Dockerfile（非 root/只读/CJK 字体）
- 任务 10-1：评测集扩至 150 题（orders 55 / monthly 48 / traffic 47，含脏数据变体），多轮会话串接，异常检出率指标接通
- 任务 11-1：流式过程反馈——LLM 适配器流式、结论逐 token 回调、编排步骤事件、Chainlit 线程+异步队列实时渲染，还清 DEBT-020
- 任务 11-2：真实模型（glm-4-flashx-250414）复跑 150 题——一次 83.3%（125/150，达标≥75%）、自修复后 86.0%（129/150，未达≥90%）、异常检出 100%（12/12）、澄清触发 100%（150/150）；2 题 3 次仍失败，19 题答案不一致

## 进行中
- 无

## 下一步
- 提升自修复后总成功率至 ≥90%（当前 86%）
- 录制演示 GIF（DEBT-025）

## 阻塞与风险
- 非阻塞：自修复后总成功率 86% 未达 90% 目标；19 题答案不一致含 DEBT-013 比对启发式因素
- 非阻塞：DEBT-025 演示 GIF 未录制
- 非阻塞：DEBT-023 LLM 结论未纳入评测，失败静默回退模板
- 非阻塞：DEBT-022 图表 50 点上限、无多序列；字体已在 Docker 内置
- 非阻塞：DEBT-021 Excel/SQLite 仅单表
- 非阻塞：DEBT-019 RAG 评测集为模板生成；DEBT-016 路由启发式；DEBT-017 问答未用 LLM 归纳；DEBT-018 50 条 QA 评测待建
- 非阻塞：DEBT-015 多轮记忆未缓存可查询结果
- 非阻塞：DEBT-013 评测答案比对为启发式；DEBT-012 只读校验非完整 AST
- 非阻塞：DEBT-009/010/011 澄清启发式与 LLM 判定回退；DEBT-007/008 主键与脏数值启发式
- 非阻塞：DEBT-003 CSV 编码回退；DEBT-006 费用依赖 provider 字段

## 关键入口
- 需求：docs/requirements.md ｜ 架构：docs/architecture.md
- 技术债：docs/debts.md ｜ 待办：docs/backlog.md
- 测试：`uv run pytest`（本机 `.venv\Scripts\python.exe -m pytest`，当前 328 项）｜ 启动：`uv run chainlit run app/main.py`
- 容器：`docker build -t prizm .` 后 `docker run --rm -p 8000:8000 --env-file .env -v prizm-data:/data prizm`
- 评测：`uv run python -m eval.build_tasks`、`-m eval.run_eval --execute`、`-m eval.run_clarify`、`-m eval.run_rag`
