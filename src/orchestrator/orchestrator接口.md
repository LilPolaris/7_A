已整合完成。现在 `src/orchestrator/` 可以作为**可输入、可输出的后端总控 agent** 使用。

## 你现在应该用哪个入口
推荐前端直接对接：

```python
from src.orchestrator import build_controller
```

或：

```python
from src.orchestrator import OrchestratorAgent
```

---

## 1. 前端对接主入口

### 方案 A：直接拿 controller
```python
controller = build_controller(shell_executor=execute_shell_stream)
```

它返回一个可直接给前端的异步函数：

```python
async def controller(user_input: str, ui) -> dict:
    ...
```

这和你前端现在的 `command_handler(user_input, ui)` 形式一致。

---

### 方案 B：实例化总控 agent
```python
agent = OrchestratorAgent(shell_executor=execute_shell_stream)
result = await agent.handle_input(user_input, ui)
```

---

## 2. 输入接口

### 输入参数
```python
user_input: str
```

含义：
- 用户在前端输入框提交的一整行文本

支持两类输入：

#### 1）直接命令
```text
/ls
/python3 demo.py
```

#### 2）自然语言
```text
帮我看看当前目录有哪些 Python 文件
解释一下多 Agent 系统
删除那个文件
```

---

## 3. 输出接口

前端只需要提供一个 `ui` 对象，满足这两个方法：

```python
class OrchestratorOutput(Protocol):
    def output_system(self, text: str, style: str = "") -> None: ...
    def output_llm(
        self,
        content: str,
        markdown: bool | None = None,
        language: str | None = None,
    ) -> None: ...
```

### 含义
#### `output_system(...)`
用于输出：
- 路由信息
- 系统提示
- Shell 命令流式输出
- 错误信息

#### `output_llm(...)`
用于输出：
- direct_answer 的最终回答
- clarification 的追问内容

你的 `AgentCLI` 已经天然满足这两个接口，所以可以直接对接。

---

## 4. 行为规则

### A. 如果输入以 `/` 开头
总控 agent 会把它视为**直接命令**：

- 若配置了 `shell_executor`：
  - 直接执行
  - 流式输出走 `ui.output_system(...)`

- 若没配置：
  - 只返回一个 pending 结果
  - 不执行

返回值示例：

```python
{
    "status": "shell_command_executed",
    "intent": "shell_command",
    "command": "ls"
}
```

---

### B. 如果输入是自然语言
总控 agent 会：

1. 调用 `intent_classifier`
2. 得到结构化结果
3. 自动输出部分内容到前端
4. 返回结构化路由结果

#### 1）`direct_answer`
会自动调用：

```python
ui.output_llm(reply)
```

返回值示例：

```python
{
    "reasoning": "...",
    "confidence": 0.96,
    "intent": "direct_answer",
    "risk_level": "low",
    "reply": "多 Agent 系统是..."
}
```

#### 2）`clarification`
会自动调用：

```python
ui.output_llm(question_and_options, markdown=True)
```

返回值示例：

```python
{
    "reasoning": "...",
    "confidence": 0.4,
    "intent": "clarification",
    "risk_level": "low",
    "question": "你指的是哪个文件？",
    "options": ["test.py", "src/test.py", "其他"]
}
```

#### 3）`shell_agent` / `tool_agent`
会自动输出路由摘要到：

```python
ui.output_system(...)
```

并返回结构化任务，供你后续接真正的下游 agent：

```python
{
    "reasoning": "...",
    "confidence": 0.91,
    "intent": "shell_agent",
    "risk_level": "low",
    "task_description": "列出当前目录下的 Python 文件",
    "context_passed": ["*.py", "./"]
}
```

---

## 5. 现在的推荐前端对接方式

如果你要接到现在的 TUI，最推荐这样写：

```python
from src.orchestrator import build_controller
from src.tui.cmd_processor import execute_shell_stream
from src.tui.application import AgentCLI

def main():
    controller = build_controller(shell_executor=execute_shell_stream)
    app = AgentCLI(command_handler=controller)
    app.run()
```

这样：
- `/ls` 之类直接命令会执行
- 自然语言会先走总控分类
- `direct_answer / clarification` 会直接显示到日志区
- `shell_agent / tool_agent` 会先输出路由结果，后面你可以再接真正的执行 agent

---

## 6. 当前 `src/orchestrator/` 的职责分层

### `llm_client.py`
通用 LLM 调用层：
- `get_llm_client()`
- `get_api_mode()`
- `generate_text_response()`
- `stream_text_response()`

### `intent_classifier.py`
意图分类层：
- `get_advanced_context()`
- `get_system_prompt()`
- `parse_llm_json()`
- `validate_intent_result()`
- `classify_intent()`
- `handle_intent()`

### `orchestrator.py`
前端对接层 / 总控层：
- `OrchestratorOutput`
- `execute_command_async()`
- `OrchestratorAgent`
- `build_controller()`
- `main_controller()`

---

## 7. 一个重要提醒
现在总控 agent **已经能对接前端**，但：

- `shell_agent`
- `tool_agent`

目前只是**路由并返回结构化任务**，还没有自动接到真正的下游执行 agent。

也就是说，当前已经完成的是：

> **总控判断 + 前端输出 + 接口统一**

下一步如果你愿意，我可以继续帮你补：

1. `shell_agent` 下游执行器接口  
2. `tool_agent` 下游执行器接口  
3. 把总控返回结果自动串到下游 agent 上，实现真正完整的多 agent 流程