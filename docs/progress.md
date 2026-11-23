# progress.md — 当前进度

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：新会话的交接仪表盘，只描述"项目现在在哪"。
> - 写法：每次任务收尾**全文重写**，但必须保留下方固定章节结构，不许增删章节。
> - 篇幅：全文 ≤ 60 行。只写现状，不写历史（历史在 git log 里）。
> - 禁令：技术债细节不写在这里（只在"阻塞与风险"里引用 DEBT 编号）；
>   功能想法不写在这里（进 backlog.md）；禁止修改本格式契约。

---

## 当前阶段
阶段 P2：透明面板（trace + Chainlit 嵌套 Step + 侧栏面板）已完成；下一步真实模型评测与交付物

## 已完成
- 任务 0-1：项目骨架，pytest 可运行
- 任务 0-2a/0-2b：评测 JSONL 格式、指标计算、10 题最小评测集
- 任务 1-1：配置 + CSV connector + DuckDB 只读执行引擎
- 任务 1-2：LLM adapter（mock 默认 / litellm 切换）+ planner + orchestrator
- 任务 1-3/1-4：Chainlit 上传→提问→SQL/结果；按 provider 取 key、显式 api_base
- 任务 2-0/2-1/2-2：额度台账、P0 profiler、澄清门禁（异常场景召回 100%）
- 任务 3-1/3-2：P1 自修复（总尝试 ≤3，还清 DEBT-001）、评测执行入口
- 任务 4-1：多轮上下文，数据集只加载一次并注入历史摘要
- 任务 5-1/5-2：RAG 数据字典 + 50 条术语评测（引用溯源 50/50）
- 2026-11-23 任务 6-1：P2 透明面板：Trace/TraceStep/ProfileReference、Orchestrator 记录各阶段、Chainlit 嵌套 Step + 侧栏面板；256 项 pytest 全绿

## 进行中
- 无

## 下一步
- 真实模型评测：网络/key 恢复后验证一次成功率 ≥75%、自修复后 ≥90%
- 交付物：README（架构图/演示/评测表）与 Dockerfile 复核

## 阻塞与风险
- 非阻塞：DEBT-020 透明面板为执行后一次性渲染，未提供流式过程反馈
- 非阻塞：DEBT-019 RAG 评测集为模板生成；DEBT-016 路由启发式；DEBT-017 问答未用 LLM 归纳；DEBT-018 50 条 QA 评测待建
- 非阻塞：DEBT-014 评测多轮未串上下文；DEBT-015 多轮记忆未缓存可查询结果
- 非阻塞：DEBT-013 评测答案比对为启发式；DEBT-012 只读校验非完整 AST
- 非阻塞：DEBT-009/010/011 澄清启发式与 LLM 判定回退；DEBT-007/008 主键与脏数值启发式
- 非阻塞：DEBT-003 CSV 编码回退；DEBT-004 litellm .env 未隔离；DEBT-005 额度台账仅内存；DEBT-006 费用依赖 provider 字段

## 关键入口
- 需求：docs/requirements.md ｜ 架构：docs/architecture.md
- 技术债：docs/debts.md ｜ 待办：docs/backlog.md
- 测试：`uv run pytest`（本机 `.venv\Scripts\python.exe -m pytest`）｜ 启动：`uv run chainlit run app/main.py`
- 评测：`uv run python -m eval.run_eval`、`-m eval.run_clarify`、`-m eval.run_rag`
