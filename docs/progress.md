# progress.md — 当前进度

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：新会话的交接仪表盘，只描述"项目现在在哪"。
> - 写法：每次任务收尾**全文重写**，但必须保留下方固定章节结构，不许增删章节。
> - 篇幅：全文 ≤ 60 行。只写现状，不写历史（历史在 git log 里）。
> - 禁令：技术债细节不写在这里（只在"阻塞与风险"里引用 DEBT 编号）；
>   功能想法不写在这里（进 backlog.md）；禁止修改本格式契约。

---

## 当前阶段
阶段 P0：数据体检 + 澄清门禁已完成；下一步 P1 自修复（开工前先复核 DEBT-001）

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
- 2026-11-21 任务 2-2：P0 澄清门禁：异常未确认口径与模糊问题先反问，选项按钮确认后注入口径执行；轻量澄清评测 100% 召回；153 项 pytest 全绿

## 进行中
- 无

## 下一步
- 阶段 P1：自修复（≤3 次，先复核 DEBT-001）
- 之后：多轮上下文、RAG 数据字典问答、透明面板

## 阻塞与风险
- 非阻塞：DEBT-009 异常相关性为关键词启发式、DEBT-010 模糊问题未自动展示概览、DEBT-011 LLM 判定失败回退启发式
- 非阻塞：DEBT-007 主键按列名启发式、DEBT-008 脏数值检测为启发式（直接影响门禁召回）
- 非阻塞：DEBT-005 额度台账仅进程内存、DEBT-006 费用依赖 provider 响应字段
- 非阻塞：DEBT-001/003/004 为已知简化，DEBT-001 在 P1 开工前必须复核

## 关键入口
- 需求：docs/requirements.md ｜ 架构：docs/architecture.md
- 技术债：docs/debts.md ｜ 待办：docs/backlog.md
- 测试：`uv run pytest`（本机用 `.venv\Scripts\python.exe -m pytest`）｜ 启动：`uv run chainlit run app/main.py`
- 评测：`uv run python -m eval.run_eval`、`uv run python -m eval.run_clarify`
