import subprocess

import ollama


MODEL = "qwen3.6:35b-a3b-coding-nvfp4"
SYSTEM = f"You are a coding agent. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run_bash(command: str) -> str:
    # 过滤掉一些明显危险的命令，防止误操作
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    # 执行命令，捕获输出，如果没有输出，返回 (no output)
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        # 把对话历史发给模型，获取回复
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            think=True
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

            # 执行工具调用，获取结果
            if tool_name != "bash":
                output = f"Error: Unknown tool {tool_name}"
            else:
                command = arguments["command"]
                print(f"\033[33m[Tool call]\033[0m bash with arguments {command}")
                output = run_bash(command)
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
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip().lower() == "clear":
            history = []
            print("\033[36mHistory cleared.\033[0m")
            continue

        # 把用户输入加到对话历史
        history.append({"role": "user", "content": query})

        # 调用 agent loop
        agent_loop(history)

        # 如果最后一条消息是模型回复，就打印出来
        # 这条回复一般是模型对用户的最终回复，或者是需要用户进一步输入的提示
        last_message = history[-1]
        if last_message["role"] == "assistant" and last_message["content"]:
            print(last_message["content"])
