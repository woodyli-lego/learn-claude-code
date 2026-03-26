import subprocess

import ollama


MODEL = "minimax-m2.7:cloud"
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
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
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
        )
        message = response["message"]

        # 如果模型有 thinking，打印出来
        if message.get("thinking"):
            print(f"\033[34m[Thinking]\033[0m {message['thinking']}")

        # 获取工具调用
        tool_calls = message.get("tool_calls", [])

        # 把模型回复加到对话历史
        assistant_message = {"role": "assistant", "content": message.get("content", "")}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        # 如果模型不需要调用工具，结束循环
        if not tool_calls:
            return

        # 如果需要调用工具，逐个调用，并把结果追加到对话历史
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name")
            arguments = func.get("arguments", {})

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

        # 把用户输入到加对话历史
        history.append({"role": "user", "content": query})

        # 调用 agent loop
        agent_loop(history)

        # 如果最后一条消息是模型回复，就打印出来
        last_message = history[-1]
        if last_message["role"] == "assistant" and last_message["content"]:
            print(last_message["content"])