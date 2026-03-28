## 1. 架构目标 (Objective)
实现多 Agent 系统的本地自然语言转换引擎。
引入基于 Ollama 的本地小语言模型（SLM, `qwen3.5:4b`）作为意图识别的**前置路由层（Fast-Path）**。采用“本地主导，云端兜底（Edge-Cloud Collaborative）”的双引擎架构，以降低云端 API 成本并实现常用命令的毫秒级离线响应。



## 2. 依赖与环境约束 (Dependencies & Environment)
* **本地模型服务**: 假设 Ollama 服务已在本地运行，监听地址为 `http://localhost:11434/v1`。
* **目标模型**: `qwen3.5:4b`。
* **兼容性**: Ollama 原生兼容 OpenAI API 规范。必须复用现有的 `openai.AsyncOpenAI` 客户端类，仅需修改 `base_url` 和 `api_key`（任意非空字符串即可）。
* **无额外库**: 严禁引入 langchain 等重型框架，保持代码轻量。

## 3. 核心模块设计 (Core Modules to Implement/Modify)

### 3.1 客户端初始化模块 (Client Initialization)
* **要求**: 在统一个 LLM 客户端管理模块中，除了现有的云端大模型 Client，需新增一个实例化为 `local_slm_client` 的全局对象。
* **配置**: 指向 Ollama 的本地端口，配置较短的 Timeout（例如 5-10 秒），因为本地模型如果不响应应迅速降级。

### 3.2 双引擎路由控制器 (Dual-Engine Router)
* **目标文件**: 主控路由逻辑所在的文件（例如 `orchestrator.py` 或 `main_controller.py`）。
* **执行流 (Execution Flow)**:
    1.  **阶段一（本地尝试）**: 接收用户的自然语言输入，结合当前环境上下文，拼装 System Prompt。首先向 `local_slm_client` 发起非流式（非 Streaming）请求，强制开启 JSON 模式（`response_format={"type": "json_object"}`）。
    2.  **阶段二（置信度校验）**: 解析本地模型返回的 JSON。提取 `confidence` 和 `intent` 字段。
    3.  **阶段三（决策门 - Decision Gate）**:
        * **放行 (Pass)**: 如果 `confidence >= 0.8` 且 `intent` 属于明确的本地操作（如 `shell_agent` 或 `direct_answer`），则**接受**本地模型的决策，直接向后游 Agent 分发任务。
        * **拦截与降级 (Fallback)**: 如果满足以下任一条件，必须放弃本地结果，原样将 Prompt 转发给云端云端大模型 Client（Cloud LLM）：
            * 本地模型解析 JSON 失败（抛出 DecodeError）。
            * 本地模型请求超时或服务未启动。
            * 本地模型自我评估 `confidence < 0.8`。
            * 判定 `intent` 为 `tool_agent`（因为调用复杂 MCP 工具通常需要极高的逻辑推理能力，4B 模型可能胜任度不足，建议直接交由云端处理）。

### 3.3 本地模型专属 Prompt 优化 (SLM Prompt Engineering)
* **要求**: 4B 级别的小模型上下文窗口和注意力机制弱于千亿参数的云端模型。请为本地引擎设计一个**精简版**的 System Prompt (`get_local_system_prompt`)。
* **裁剪策略**:
    * 保留核心的 CoT 逻辑链限制（必须按顺序输出 `reasoning` -> `confidence` -> `intent`）。
    * **限制环境上下文体积**: 目录概览（files）限制在最多 10 项，且剔除隐藏文件，防止过度消耗局部注意力。
    * **简化输出 Schema**: 如果意图是 `shell_agent`，只需其输出 `task_description`，无需让其进行深度的 `risk_level` 评估（风险评估交由后游的 Shell Agent 利用正则和自身机制处理）。

## 4. 错误处理与用户体验 (Error Handling & UX)
* **静默降级**: 当本地 Ollama 未启动或崩溃时，系统不应在前端（Textual TUI）抛出红色报错，而应静默捕获 `ConnectionError`，并在终端日志区打印一条中性提示（例如：“*本地计算节点未就绪，已切换至云端大脑...*”），然后平滑过渡到云端请求。
* **耗时监控**: 记录本地模型和云端模型的路由耗时，可在 TUI 的状态栏（StatusBar）或 Debug 面板中展示，以直观体现双引擎带来的延迟优化。

