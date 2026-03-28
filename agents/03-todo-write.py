import os
import subprocess
from pathlib import Path

import ollama


MODEL = "minimax-m2.7:cloud"

WORKDIR = Path.cwd()
SYSTEM = f"""You are a coding agent at {WORKDIR}. You MUST use the todo tool to plan and track your progress for any task.
Mark in_progress before starting, completed when done. Prefer tools over prose. Act, don't explain."""


class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


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


# 当 LLM 返回要求 function call 的时候，调用对应的函数执行。
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
}


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
    {"type": "function", "function": {
        "name": "todo",
        "description": "Update todo list. Track progress on multi-step tasks.",
        "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}
    }},
]


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    rounds_since_todo = 0
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

        used_todo = False
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
                print(f"\033[33m[Tool calling]\033[0m {tool_name}")
                output = handler(**arguments)
                print(output[:200])

            # 把工具调用的结果追加到对话历史
            messages.append({
                "role": "tool",
                "content": output,
            })

            # 如果有使用 todo，记录下来
            if tool_name == "todo":
                used_todo = True

        # 如果该轮用了 todo，重置计数；否则计数加一
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        # 如果连续 3 轮没用 todo，就在下一轮开头提醒模型更新 todo
        # 这个提醒，是模拟用户的行为，所有 role 为 user
        if rounds_since_todo >= 3:
            messages.append({"role": "user", "content": "<reminder>Update your todos.</reminder>"})

if __name__ == "__main__":
    history = []
    while True:
        # 获取输入，验证是否 exit
        try:
            query = input("\033[36ms03 >> \033[0m")
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
