import subprocess
from pathlib import Path

import ollama


MODEL = "minimax-m2.7:cloud"
WORKDIR = Path.cwd()
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# 检查执行路径，只能在 WORKDIR 目录下执行
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# bash tool
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# read tool，用 read_text 读取 limit 行以内的内容
def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


# write tool，用 write_text 写入内容
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# edit tool，对 read 和 write 的结合，
# 读取当前文件内容，找出 old_text，替换成 new_text，再写回去
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 告诉 LLM，有这些 tool 可以使用（Ollama 格式）
TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read file contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}
    }},
]


# 当 LLM 返回要求 function call 的时候，调用对应的函数执行。
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        # 把对话历史发给模型，获取回复
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
        )
        message = response["message"]

        # 把 resp.reasoning 部分打印出来
        if message.get("thinking"):
            print(f"\033[34m[Thinking]\033[0m {message['thinking']}")

        # 把 resp.content 放入模型回复
        assistant_message = {"role": "assistant", "content": message.get("content", "")}

        # 如果 resp.tool_calls，也放入模型回复
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        # 把模型回复追加到对话历史
        messages.append(assistant_message)

        # 如果模型不需要调用工具，结束循环
        # 一般来说，就是模型已经把请求解决了，或者需要进一步用户输入
        if not tool_calls:
            return

        # 如果需要调用工具，逐个调用，并把结果追加到对话历史
        for tool_call in tool_calls:
            # 从 tool_call 里解析出工具名称和参数
            func = tool_call.get("function", {})
            tool_name = func.get("name")
            arguments = func.get("arguments", {})

            # 获取 tool handler，执行工具调用，获取结果
            handler = TOOL_HANDLERS.get(tool_name)
            if handler is None:
                output = f"Error: Unknown tool {tool_name}"
            else:
                print(f"\033[33m[Tool call]\033[0m {tool_name} with arguments {arguments}")
                output = handler(**arguments)
                print(output[:200])

            # 把工具调用的结果追加到对话历史
            messages.append({
                "role": "tool",
                "content": output,
            })


if __name__ == "__main__":
    history = []
    while True:
        # 获取输入，验证是否 exit
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip().lower() == "clear":
            history = []
            print("\033[36mHistory cleared.\033[0m")
            continue

        # 把用户输入到加对话历史
        history.append({"role": "user", "content": query})

        # 调用 agent loop
        agent_loop(history)

        # 如果最后一条消息是模型回复，就打印出来
        # 这条回复一般是模型对用户的最终回复，或者是需要用户进一步输入的提示
        last_message = history[-1]
        if last_message["role"] == "assistant" and last_message["content"]:
            print(last_message["content"])
