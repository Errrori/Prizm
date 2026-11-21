# debts.md — 技术债台账

> **格式契约（AI 必读，禁止修改本节）**
> - 用途：登记所有简化实现、临时方案、已知缺陷，到触发条件时必须偿还。
> - 写法：**只许追加新条目和更新状态字段**，禁止删除、合并、改写历史条目。
> - 编号：DEBT-001 起递增，编号永不复用。
> - 每笔债必须填全下方模板所有字段；"触发条件"必须可判断
>   （某阶段开始时 / 某事件发生前），禁止写"以后""有空时"。
> - 登记新债时，必须在代码对应位置加 `# DEBT-XXX: 一句话说明` 注释。
> - 还清时：状态改"已还清"+ 填还清日期和 commit，条目保留不删。
> - 每个阶段开工前扫描本文件，触发条件已命中的债报告给我，列为该阶段第一批任务。
> - 禁止修改本格式契约。

---

## 条目模板（照此填写）

### DEBT-XXX 一句话标题
- 状态：未还 / 已还清
- 现状：具体做了什么简化
- 为什么简化：当时的取舍
- 触发条件：什么时候必须还（可判断的阶段或事件）
- 涉及文件：路径列表（代码中应有 # DEBT-XXX 注释）
- 登记日期：YYYY-MM-DD
- 还清记录：（还清时填：日期 + commit + 做了什么）

---

## 台账

### DEBT-001 SQL 只读校验基于关键字黑名单
- 状态：未还
- 现状：`ensure_read_only_sql` 用 SELECT/WITH 前缀 + 关键字/函数黑名单拦截，未做 SQL AST 解析。
- 为什么简化：阶段 1 只需跑通链路；DuckDB 已设 `enable_external_access=false`，黑名单作为第二道防线。
- 触发条件：接入自修复（repair）前必须复核。
- 涉及文件：src/prizm/engine/duckdb_engine.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-002 LLM 额度与调用统计未实现
- 状态：已还清
- 现状：`LiteLLMAdapter` 只做单次 completion，未统计 token/费用，未执行每日额度与会话次数上限。
- 为什么简化：阶段 1 默认走 mock，不产生真实调用。
- 触发条件：首次接入真实模型 key 联调前。
- 涉及文件：src/prizm/config.py, src/prizm/llm/adapter.py
- 登记日期：2026-11-21
- 还清记录：2026-11-21 任务 2-0：新增 utils/tokens.py（Usage/Completion/DailyUsageLedger/SessionTurnCounter），adapter 返回 usage 并按日额度拦截，Chainlit 增加会话次数上限与友好提示；commit 见本任务 git log。

### DEBT-003 CSV 编码采用启发式回退
- 状态：未还
- 现状：按 UTF-8(BOM) → GBK 顺序尝试，未做 charset 检测；非 GBK 字节可能被误解为乱码而不报错。
- 为什么简化：当前评测数据集为 UTF-8，先覆盖常见中文 CSV。
- 触发条件：接入真实用户上传文件前。
- 涉及文件：src/prizm/datasource/csv_connector.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-004 litellm 自身的 .env 加载未隔离
- 状态：未还
- 现状：`LiteLLMAdapter` 已显式传 `api_key` / `api_base`，但 `import litellm` 仍会自行执行 `load_dotenv()`（按进程 CWD 找 `.env`）并把其中的 key 注入 `os.environ`；未固定 `.env` 路径，也未隔离环境变量。
- 为什么简化：阶段 1 为单机开发运行，显式传参已覆盖主路径。
- 触发条件：容器化部署，或多环境（dev/prod 共用进程环境 / CWD 不固定）部署前。
- 涉及文件：src/prizm/llm/adapter.py, src/prizm/config.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-005 每日额度台账仅存进程内存
- 状态：未还
- 现状：`DailyUsageLedger` 只保存在当前进程内，进程重启或部署多副本时每日 token 用量会归零/不共享。
- 为什么简化：当前为单机 demo，单进程即可满足每日额度限制。
- 触发条件：在线 demo 部署，或多进程/多副本运行前。
- 涉及文件：src/prizm/utils/tokens.py, app/main.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-006 费用统计依赖 litellm 响应内部字段
- 状态：未还
- 现状：费用从 `response._hidden_params.response_cost` 读取，provider/版本不提供该字段时只累计 token、`cost_usd` 为 None。
- 为什么简化：不引入独立价格表，先复用 litellm 自带成本信息。
- 触发条件：需要在界面展示费用，或按费用而非 token 做额度控制前。
- 涉及文件：src/prizm/llm/adapter.py
- 登记日期：2026-11-21
- 还清记录：—
