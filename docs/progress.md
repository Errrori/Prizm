# progress.md — 当前进度

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：新会话的交接仪表盘，只描述"项目现在在哪"。
> - 写法：每次任务收尾**全文重写**，但必须保留下方固定章节结构，不许增删章节。
> - 篇幅：全文 ≤ 60 行。只写现状，不写历史（历史在 git log 里）。
> - 禁令：技术债细节不写在这里（只在"阻塞与风险"里引用 DEBT 编号）；
>   功能想法不写在这里（进 backlog.md）；禁止修改本格式契约。

---

## 当前阶段
阶段 1 最薄端到端链路已完成（仅 CSV）；下一步：P0 数据体检 profiler

## 已完成
- 2026-11-20 任务 0-1：项目骨架，pytest 可运行，首次 commit
- 2026-11-20 任务 0-2a：评测任务 JSONL 格式 + 指标计算 + 14 条单测
- 2026-11-20 环境：uv + 核心/dev/xl 依赖安装完成，生成 uv.lock
- 2026-11-20 任务 0-2b：10 题最小评测集 + 订单脏数据版 + run_eval 骨架
- 2026-11-21 任务 1-1：配置 + CSV connector + DuckDB 只读执行引擎
- 2026-11-21 任务 1-2：LLM adapter（mock 默认 / litellm 切换）+ planner + orchestrator
- 2026-11-21 任务 1-3：Chainlit CSV 上传→提问→SQL/结果展示；69 项 pytest 全绿
- 2026-11-21 任务 1-4：LLM 配置加固：`PRIZM_LLM_API_BASE` 显式传 api_base、按 provider 取 key、缺 key 启动即报错；82 项 pytest 全绿

## 进行中
- 无

## 下一步
- 偿还 DEBT-002（触发条件已命中）：token/费用统计 + 每日额度与会话次数上限
- 阶段 P0：数据体检 profiler（缺失/重复/脏格式检测 + 紧凑体检报告）

## 阻塞与风险
- DEBT-002 触发条件已命中（`.env` 已有真实 key 与 base_url），列为下一任务
- 非阻塞：DEBT-001/003/004 为已知简化，触发条件均未命中

## 关键入口
- 需求：docs/requirements.md ｜ 架构：docs/architecture.md
- 技术债：docs/debts.md ｜ 待办：docs/backlog.md
- 测试：`uv run pytest`（本机用 `.venv\Scripts\python.exe -m pytest`）｜ 启动：`uv run chainlit run app/main.py` ｜ 评测：`uv run python -m eval.run_eval`
