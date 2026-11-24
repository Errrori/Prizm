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
- 状态：已还清
- 现状：`ensure_read_only_sql` 用 SELECT/WITH 前缀 + 关键字/函数黑名单拦截，未做 SQL AST 解析。
- 为什么简化：阶段 1 只需跑通链路；DuckDB 已设 `enable_external_access=false`，黑名单作为第二道防线。
- 触发条件：接入自修复（repair）前必须复核。
- 涉及文件：src/prizm/engine/duckdb_engine.py
- 登记日期：2026-11-21
- 还清记录：2026-11-21 任务 3-1：改为 DuckDB parser 语句类型白名单（仅单条 SELECT；拒绝 CTE 内 INSERT/UPDATE/DELETE/COPY），并对去字符串/注释后的词法骨架做关键字与危险函数拦截；外部访问仍关闭。commit 见本任务 git log。

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
- 状态：已还清
- 现状：`LiteLLMAdapter` 已显式传 `api_key` / `api_base`，但 `import litellm` 仍会自行执行 `load_dotenv()`（按进程 CWD 找 `.env`）并把其中的 key 注入 `os.environ`；未固定 `.env` 路径，也未隔离环境变量。
- 为什么简化：阶段 1 为单机开发运行，显式传参已覆盖主路径。
- 触发条件：容器化部署，或多环境（dev/prod 共用进程环境 / CWD 不固定）部署前。
- 涉及文件：src/prizm/llm/adapter.py, src/prizm/config.py
- 登记日期：2026-11-21
- 还清记录：2026-11-23 任务 9-1：import litellm 前设置 `LITELLM_MODE=PROD`（禁止自动 load_dotenv）与 `LITELLM_LOCAL_MODEL_COST_MAP=True`（不联网拉成本表）；新增子进程回归测试验证 CWD `.env` 不会泄漏进环境。commit 见本任务 git log。

### DEBT-005 每日额度台账仅存进程内存
- 状态：已还清
- 现状：`DailyUsageLedger` 只保存在当前进程内，进程重启或部署多副本时每日 token 用量会归零/不共享。
- 为什么简化：当前为单机 demo，单进程即可满足每日额度限制。
- 触发条件：在线 demo 部署，或多进程/多副本运行前。
- 涉及文件：src/prizm/utils/tokens.py, app/main.py
- 登记日期：2026-11-21
- 还清记录：2026-11-23 任务 9-1：新增 `SqliteUsageLedger`（按自然日 upsert，跨日清零），app 按 `PRIZM_USAGE_LEDGER_PATH` 使用，Docker 将 `/data` 挂为持久卷；重启与同机多进程共享额度。commit 见本任务 git log。

### DEBT-006 费用统计依赖 litellm 响应内部字段
- 状态：未还
- 现状：费用从 `response._hidden_params.response_cost` 读取，provider/版本不提供该字段时只累计 token、`cost_usd` 为 None。
- 为什么简化：不引入独立价格表，先复用 litellm 自带成本信息。
- 触发条件：需要在界面展示费用，或按费用而非 token 做额度控制前。
- 涉及文件：src/prizm/llm/adapter.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-007 主键按列名启发式识别
- 状态：未还
- 现状：未显式指定主键时，按列名 `id` 或 `*_id` 取第一个匹配列作为疑似主键，多候选、复合主键或业务主键命名不规范时会误判/漏判。
- 为什么简化：MVP 不做 UI 主键选择，先用常见命名覆盖订单类数据。
- 触发条件：接入多表/复合主键，或评测出现主键误判前。
- 涉及文件：src/prizm/profiling/checks.py, src/prizm/profiling/profiler.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-008 脏数值格式检测为启发式
- 状态：未还
- 现状：只识别千分位、货币符、常见中文单位与百分号，未覆盖欧式分隔符、括号负数、更多单位或自然语言数值。
- 为什么简化：当前评测脏数据只埋了千分位，先覆盖需求列出的常见形态。
- 触发条件：评测出现脏格式漏检，或接入真实用户数据前。
- 涉及文件：src/prizm/profiling/checks.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-009 异常与问题的相关性判断为关键词启发式
- 状态：未还
- 现状：`ClarifyGate` 通过问题是否命中受异常影响的列名，或包含聚合/清洗类关键词来判断异常是否相关，未结合 SQL 语义或实际影响的统计口径。
- 为什么简化：澄清发生在生成 SQL 之前，先用可单测、确定性的启发式覆盖当前评测集。
- 触发条件：评测出现澄清漏报/误报，或接入多列/复合口径前必须复核。
- 涉及文件：src/prizm/agent/clarify.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-010 模糊问题仅反问关注点、未自动展示概览
- 状态：未还
- 现状：问题模糊时只返回“关注方向”选项，未按需求示例自动展示数据概览后再询问关注点。
- 为什么简化：文件上传后已展示过一次体检报告，本轮先保证门禁拦截与口径确认。
- 触发条件：模糊问题进入真实用户验收前，或澄清评测出现该场景误报/漏报时。
- 涉及文件：src/prizm/agent/clarify.py, app/main.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-011 LLM 模糊判定失败时回退关键词启发式
- 状态：未还
- 现状：`LLMAmbiguityJudge` 抛错或输出解析失败时，门禁回退到关键词启发式，可能漏判或误判模糊问题。
- 为什么简化：避免模型抖动或临时故障阻断用户提问。
- 触发条件：接入真实模型后发现模糊判定漏报时。
- 涉及文件：src/prizm/agent/clarify.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-012 只读校验仍非完整 AST，危险函数依赖名单
- 状态：未还
- 现状：`ensure_read_only_sql` 用 DuckDB parser 判定语句类型，再对去字符串/注释后的词法骨架做关键字与危险表函数名单校验；CTE 内写操作、动态 SQL/文件函数已拦截，但无法覆盖未来 DuckDB 新函数或复杂语法绕过，且未加引号的保留字标识符可能被误杀。
- 为什么简化：当前不引入额外 SQL AST 依赖，先保证自修复前的执行面可控。
- 触发条件：升级 DuckDB 版本、评测出现只读校验绕过或误杀、接入不可信用户数据前。
- 涉及文件：src/prizm/engine/duckdb_engine.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-013 评测答案比对为启发式
- 状态：未还
- 现状：`matches_expected` 对标量答案要求单行单列、记录列表按顺序逐行比较期望字段子集，数值用近似相等、字符串去首尾空白、日期兼容常见格式；未处理同义表达、单位换算、多列等价结果或业务口径差异。
- 为什么简化：当前最小评测集答案结构简单，先用确定性规则跑通执行入口；标准答案本身不能为了指标好看而改。
- 触发条件：正式用该入口做发布验收，或评测集扩展到 50 题后出现误判/漏判时。
- 涉及文件：eval/run_eval.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-014 评测多轮任务未串接上一轮上下文
- 状态：已还清
- 现状：`eval/run_eval.py --execute` 逐题独立创建 Orchestrator，`multi_turn` 任务没有带上上一题的问题/SQL/结果，只按单题执行。
- 为什么简化：当前 `tasks.jsonl` 没有定义任务间的会话关系，先复用单题执行入口跑通指标。
- 触发条件：正式验收多轮能力，或评测集补充多轮任务序列前。
- 涉及文件：eval/run_eval.py, eval/tasks.jsonl
- 登记日期：2026-11-21
- 还清记录：2026-11-23 任务 10-1：`EvalTask` 增加 `session` 字段，`load_tasks` 校验同会话任务连续且同数据集；`execute_tasks` 按会话复用同一 Orchestrator，使追问注入上一轮 SQL/结果摘要；新增单测验证第二轮 prompt 含上一轮 SQL。commit 见本任务 git log。

### DEBT-015 多轮记忆只缓存摘要，未缓存可查询的上一轮结果
- 状态：未还
- 现状：`ConversationMemory` 保存问题、SQL、列、行数和前几行 preview；追问时靠上一轮 SQL 重新计算，未把上一轮结果 DataFrame 注册成可查询视图。
- 为什么简化：SQL-first 下让模型改写上一轮 SQL 已能覆盖当前追问；避免在会话内长期持有大结果集。
- 触发条件：评测出现必须基于上一轮结果集合继续过滤/聚合的追问时。
- 涉及文件：src/prizm/agent/memory.py, src/prizm/agent/orchestrator.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-016 数据字典问答路由为关键词启发式
- 状态：未还
- 现状：`is_dictionary_question` 用“含义/定义/是什么/字段说明/数据字典”等关键词判断是否走 RAG，并用统计类关键词排除分析问题；同义表达、隐含术语问题或分析/术语混合问题可能误判。
- 为什么简化：当前先让 RAG 可被触发，避免每次提问都依赖额外 LLM 路由调用。
- 触发条件：真实用户出现明显误路由，或 RAG 评测显示术语问题漏检时。
- 涉及文件：src/prizm/rag/retriever.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-017 RAG 问答为检索片段拼接
- 状态：未还
- 现状：`DictionaryQA` 直接拼接 BM25 命中的字典片段并标注来源，未用 LLM 把多个片段归纳成自然语言答案。
- 为什么简化：优先保证“有出处、不臆造”，避免 RAG 生成阶段引入幻觉和额外成本。
- 触发条件：真实用户反馈答案可读性不足，或需要跨多个字典条目综合回答时。
- 涉及文件：src/prizm/rag/retriever.py
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-018 RAG 50 条术语 QA 评测尚未建立
- 状态：已还清
- 现状：当前只有 RAG 单元测试，尚未建立需求要求的 50 条术语 QA 集，也未统计引用溯源准确率。
- 为什么简化：先完成数据字典生成、BM25 检索、引用返回与 UI 路由，评测集另行设计以避免用模板答案自证。
- 触发条件：RAG 模块进入验收，或需要对外报告引用准确率前。
- 涉及文件：src/prizm/rag/__init__.py, eval/
- 登记日期：2026-11-21
- 还清记录：2026-11-24 任务 12-1 复核：`eval/rag_tasks.jsonl` 已含 50 条术语问法，`eval/run_rag.py` 输出引用溯源 50/50，README 已报告；本条此前漏更新状态。commit 见本任务 git log。

### DEBT-019 RAG 评测集为模板生成
- 状态：未还
- 现状：`eval/rag_tasks.jsonl` 的 50 条术语问法由列名 + 固定问法模板生成，覆盖路由与引用溯源，但不等于真实用户的同义、口语化或跨术语提问。
- 为什么简化：先用可复现的 50 条集跑通引用准确率指标，避免用模型自答生成标准答案。
- 触发条件：对外报告 RAG 指标前，或接入真实用户后出现模板未覆盖的问法时。
- 涉及文件：eval/run_rag.py, eval/rag_tasks.jsonl
- 登记日期：2026-11-21
- 还清记录：—

### DEBT-020 透明面板步骤为执行后一次性渲染
- 状态：已还清
- 现状：`Orchestrator.ask/resolve` 先同步跑完整轮分析，`app/main.py` 再按 `Trace` 渲染嵌套 Step 与侧栏面板；模型调用期间没有 token 级或步骤级实时反馈。
- 为什么简化：当前默认 mock 且整轮同步执行，先保证 trace 可测试、面板内容完整，避免在 UI 层引入流式状态机。
- 触发条件：接入真实模型进行联调/演示前，或用户反馈等待期缺少过程反馈时。
- 涉及文件：app/main.py, src/prizm/agent/orchestrator.py, src/prizm/agent/composer.py, src/prizm/llm/adapter.py
- 登记日期：2026-11-23
- 还清记录：2026-11-24 任务 11-1：`Orchestrator.ask/resolve` 新增可选 `progress` 回调，逐步发 `step` 事件、结论逐块发 `conclusion_token`；`LLMAdapter` 增可选 `stream`（Mock 切片 / litellm `include_usage`），`ConclusionComposer` 支持 `on_token` 并在失败时回退模板；`app/main.py` 用 `asyncio.to_thread` 跑同步编排、`asyncio.Queue` 桥接实时渲染 Step 与结论 token，结束后按权威文本校正。未传回调时行为不变。commit 见本任务 git log。

### DEBT-021 Excel/SQLite 数据源仅支持单表整表读取
- 状态：未还
- 现状：Excel 仅支持 `.xlsx` 且一次只读一个 sheet（默认表头，不处理合并单元格/多级表头）；SQLite 用 stdlib 只读连接一次读入一张用户表，不支持视图、附加库或多表关联；两者都整表读入内存。
- 为什么简化：Orchestrator 当前只持有单个 DataFrame，先验证 Connector 抽象与常见上传场景，避免引入多关系执行和 `.xls` 可选依赖。
- 触发条件：接入真实用户 Excel/SQLite 后出现多 sheet 汇总、多表关联、视图读取或大表内存不足。
- 涉及文件：src/prizm/datasource/excel_connector.py, src/prizm/datasource/sqlite_connector.py, app/main.py
- 登记日期：2026-11-23
- 还清记录：—

### DEBT-022 图表中文字体与点数上限为简化实现
- 状态：未还
- 现状：matplotlib 中文字体依赖系统已装字体（Windows 候选雅黑/黑体，容器可能缺字显示方框）；每张图最多画前 50 个数据点且只在 trace 记录截断，不提示用户；饼图要求 y 非负，未支持多序列/分组/面积图。
- 为什么简化：先把 P0 的图表链路跑通并用结构化 spec 控制安全面，避免引入字体文件打包、绘图 DSL 与分页交互。
- 触发条件：Docker/在线 demo 部署前，或用户反馈图表乱码、只显示部分数据造成误读、需要多序列图表时。
- 涉及文件：src/prizm/viz/charts.py, app/main.py
- 登记日期：2026-11-23
- 还清记录：—

### DEBT-023 LLM 结论质量未纳入评测且失败静默回退
- 状态：未还
- 现状：`ConclusionComposer` 让 LLM 基于前 10 行样本写 1-3 句结论，模型报错/额度不足/空输出时静默回退模板；样本可能不代表全量结果；评测入口（`eval/run_eval.py`）未挂载 composer，指标不校验结论准确性与可读性。
- 为什么简化：先保证结果链路不被额外模型调用拖垮，避免把结论生成引入首轮成功率指标；模板兜底保证始终有输出。
- 触发条件：需要对外报告结论质量、或真实模型验收要求结论可信度时。
- 涉及文件：src/prizm/agent/composer.py, src/prizm/agent/orchestrator.py, eval/run_eval.py
- 登记日期：2026-11-23
- 还清记录：—

### DEBT-024 评测集仅 10 题，未达需求约 50 题规模
- 状态：已还清
- 现状：`eval/tasks.jsonl` 只有 10 题，覆盖 cleaning / aggregation / trend / attribution / multi_turn；数据集只有 `orders.csv` 与 `orders_dirty.csv`，未达到需求约定的约 50 题 × 3 个公开风格数据集。
- 为什么简化：先跑通真实模型执行入口与指标；标准答案需人工设计，避免用模板答案自证。
- 触发条件：对外正式验收、或宣称完成需求第 7 节全部评测指标前。
- 涉及文件：eval/tasks.jsonl, eval/datasets, docs/requirements.md
- 登记日期：2026-11-23
- 还清记录：2026-11-23 任务 10-1：新增月度销量/渠道流量基础与脏数据集，评测集扩为 150 题（orders 55 / monthly 48 / traffic 47），五类任务与 12 条澄清场景全覆盖；标准答案由 `eval/build_tasks.py` 独立 pandas 计算生成并由单测比对；`anomaly_expectations.json` 接通异常检出率。真实模型复跑待额度恢复。commit 见本任务 git log。

### DEBT-025 README 演示 GIF 未录制
- 状态：已取消
- 现状：需求方决定不做演示 GIF，本条不再偿还；README 保留架构图、文字流程与评测表。
- 为什么简化：优先完成可运行链路、容器与评测，交付阶段亦确认不需要 UI 录屏。
- 触发条件：无（已取消，不再偿还）。
- 涉及文件：README.md
- 登记日期：2026-11-23
- 还清记录：—

### DEBT-026 差值/对比类问题偶发只返回单边值
- 状态：未还
- 现状：全量 150 题中 ord-021（“3 月比 1 月多多少”）首次只返回 3 月销售额，未做差值；SQL 可执行故未触发自修复，最终 149/150。
- 为什么简化：依赖 planner prompt 约束输出形状，未对“比…多多少”增加确定性 SQL 结构校验或结果形状校验。
- 触发条件：下次正式验收前，或同类差值题再次失败达到 2 题。
- 涉及文件：src/prizm/agent/planner.py, eval/tasks.jsonl
- 登记日期：2026-11-24
- 还清记录：—

### DEBT-027 用户自带 Key 会话输入未实现
- 状态：未还
- 现状：需求 §4 要求界面支持用户可选填入自己的 key（仅会话内存、不落盘、不打日志）；当前仅在额度超限时提示“改用你自己的 API key”，没有输入入口，key 仍只从环境变量读取。
- 为什么简化：MVP 联调优先使用部署方 key 与环境变量配置。
- 触发条件：在线 demo 对外提供前，或用户无法使用部署方 key 时。
- 涉及文件：app/main.py, src/prizm/config.py, src/prizm/llm/adapter.py
- 登记日期：2026-11-24
- 还清记录：—

