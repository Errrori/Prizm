# progress.md — 当前进度

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：新会话的交接仪表盘，只描述"项目现在在哪"。
> - 写法：每次任务收尾**全文重写**，但必须保留下方固定章节结构，不许增删章节。
> - 篇幅：全文 ≤ 60 行。只写现状，不写历史（历史在 git log 里）。
> - 禁令：技术债细节不写在这里（只在"阻塞与风险"里引用 DEBT 编号）；
>   功能想法不写在这里（进 backlog.md）；禁止修改本格式契约。

---

## 当前阶段
阶段 P1：RAG 数据字典与 50 条术语评测已完成；下一步 P2 透明面板（真实模型评测仍需网络/key）

## 已完成
- 2026-11-20 任务 0-1：项目骨架，pytest 可运行，首次 commit
- 2026-11-20 任务 0-2a：评测任务 JSONL 格式 + 指标计算 + 14 条单测
- 2026-11-20 环境：uv + 核心/dev/xl 依赖安装完成，生成 uv.lock
- 2026-11-20 任务 0-2b：10 题最小评测集 + 订单脏数据版 + run_eval 骨架
- 2026-11-21 任务 1-1：配置 + CSV connector + DuckDB 只读执行引擎
- 2026-11-21 任务 1-2：LLM adapter（mock 默认 / litellm 切换）+ planner + orchestrator
- 2026-11-21 任务 1-3：Chainlit CSV 上传→提问→SQL/结果展示
- 2026-11-21 任务 1-4：LLM 配置加固：显式传 api_base、按 provider 取 key
- 2026-11-21 任务 2-0：还清 DEBT-002：usage/费用统计、每日 token 额度、会话次数上限
- 2026-11-21 任务 2-1：P0 profiler：缺失/完全重复/主键重复/数值脏格式检测，紧凑报告注入 planner prompt 并在 Chainlit 展示
- 2026-11-21 任务 2-2：P0 澄清门禁：异常未确认口径与模糊问题先反问，选项按钮确认后注入口径执行；轻量澄清评测 100% 召回
- 2026-11-21 任务 3-1：P1 自修复：总尝试 ≤ 3 次（含首次），错误格式化回传、修复反馈注入、成功/耗尽路径；还清 DEBT-001（DuckDB parser 语句白名单 + 词法关键字/函数校验）
- 2026-11-21 任务 3-2：P1-2 评测执行入口：`run_eval --execute` 逐题跑 Orchestrator、产出 TaskOutcome JSONL 并统计一次/自修复后成功率；支持 limit/max-attempts；mock 模式拒绝执行
- 2026-11-21 任务 4-1：P1 多轮上下文：`ConversationMemory` 保存最近轮次问题/SQL/结果摘要并注入 planner；Orchestrator 数据集只加载一次、每轮成功记录历史
- 2026-11-21 任务 5-1：P1 RAG 核心：生成数据字典 + .md/.txt 说明文档摄取 + BM25Plus 中英文检索 + 带来源/字段引用的字典问答；Orchestrator 术语问题直接走 RAG
- 2026-11-21 任务 5-2：RAG 术语评测：50 条问法，引用溯源 50/50 = 100.0%；`eval/run_rag.py` 可复跑；251 项 pytest 全绿

## 进行中
- 无

## 下一步
- 阶段 P2：透明面板（trace + Chainlit Step）
- 真实模型评测需网络/key 恢复后验证一次 ≥75%、自修复后 ≥90%

## 阻塞与风险
- 非阻塞：DEBT-019 RAG 评测集为列名+模板生成，需补真实同义问法
- 非阻塞：DEBT-016 RAG 路由为关键词启发式
- 非阻塞：DEBT-017 RAG 问答为检索片段拼接，未用 LLM 归纳
- 非阻塞：DEBT-018 RAG 50 条术语 QA 评测尚未建立
- 非阻塞：DEBT-014 评测多轮任务未串接上一轮上下文
- 非阻塞：DEBT-015 多轮记忆只缓存摘要，未缓存可查询的上一轮结果
- 非阻塞：DEBT-013 评测答案比对为启发式，需在正式验收前复核
- 非阻塞：DEBT-012 只读校验仍非完整 AST、危险函数依赖名单
- 非阻塞：DEBT-009 异常相关性为关键词启发式、DEBT-010 模糊问题未自动展示概览、DEBT-011 LLM 判定失败回退启发式
- 非阻塞：DEBT-007 主键按列名启发式、DEBT-008 脏数值检测为启发式（直接影响门禁召回）
- 非阻塞：DEBT-003 CSV 编码启发式回退、DEBT-004 litellm .env 加载未隔离
- 非阻塞：DEBT-005 额度台账仅进程内存、DEBT-006 费用依赖 provider 响应字段

## 关键入口
- 需求：docs/requirements.md ｜ 架构：docs/architecture.md
- 技术债：docs/debts.md ｜ 待办：docs/backlog.md
- 测试：`uv run pytest`（本机用 `.venv\Scripts\python.exe -m pytest`）｜ 启动：`uv run chainlit run app/main.py`
- 评测：`uv run python -m eval.run_eval`、`uv run python -m eval.run_clarify`、`uv run python -m eval.run_rag`
