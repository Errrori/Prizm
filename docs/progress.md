# progress.md — 当前进度

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：新会话的交接仪表盘，只描述"项目现在在哪"。
> - 写法：每次任务收尾**全文重写**，但必须保留下方固定章节结构，不许增删章节。
> - 篇幅：全文 ≤ 60 行。只写现状，不写历史（历史在 git log 里）。
> - 禁令：技术债细节不写在这里（只在"阻塞与风险"里引用 DEBT 编号）；
>   功能想法不写在这里（进 backlog.md）；禁止修改本格式契约。

---

## 当前阶段
阶段 0 已完成；下一阶段：最薄端到端链路

## 已完成
- 2026-11-20 任务 0-1：项目骨架，pytest 可运行，首次 commit
- 2026-11-20 任务 0-2a：评测任务 JSONL 格式 + 指标计算 + 14 条单测
- 2026-11-20 环境：uv + 核心/dev/xl 依赖安装完成，生成 uv.lock
- 2026-11-20 任务 0-2b：10 题最小评测集 + 订单脏数据版 + run_eval 骨架

## 进行中
- 无

## 下一步
- 阶段 1：最薄端到端链路（CSV/Excel 接入 → DuckDB 注册 → 最小问答 → 结果展示）

## 阻塞与风险
- 无（有则写：现象 + 影响 + 关联 DEBT 编号）

## 关键入口
- 需求：docs/requirements.md ｜ 架构：docs/architecture.md
- 技术债：docs/debts.md ｜ 待办：docs/backlog.md
- 测试：`uv run pytest` ｜ 启动：`uv run chainlit run app/main.py` ｜ 评测：`uv run python -m eval.run_eval`