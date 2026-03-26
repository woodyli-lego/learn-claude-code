import os
import json
import subprocess

from openai import OpenAI


OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "minimax-m2.7:cloud"


client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

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
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        # 把对话历史发给模型，获取回复
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            max_tokens=8000,
        )
        message = response.choices[0].message

        # 把模型回复的内容（如果有的话）加到对话历史
        assistant_message = {"role": "assistant", "content": message.content or ""}

        # 如果模型有 reasoning，把 reasoning 打印出来（但不追加到对话历史，因为 reasoning 不是模型回复的一部分）
        if message.reasoning:
            print(f"\033[34m[Reasoning]\033[0m {message.reasoning}")

        # 如果模型调用了工具，也把工具调用加到对话历史
        if message.tool_calls:
            assistant_message["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]
        messages.append(assistant_message)

        # 如果模型不需要调用工具，结束循环（此时 assistant_message.content 会有内容，会被打印为 agent 的回复）
        if not message.tool_calls:
            return

        # 如果需要调用工具，那么逐个调用，并把结果追加到对话历史
        for tool_call in message.tool_calls:
            # 由于我们只定义了 bash 工具，这里需要做 validation
            if tool_call.function.name != "bash":
                output = f"Error: Unknown tool {tool_call.function.name}"
            else:
                # 解析工具调用的参数，执行命令，获取结果并输出
                arguments = json.loads(tool_call.function.arguments)
                command = arguments["command"]
                print(f"\033[33m[Tool call] {tool_call.function.name}\033[0m with arguments {command}")
                output = run_bash(command)
                print(output[:200])
            # 把工具调用的结果追加到对话历史，作为模型下一轮输入的一部分
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
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
        # 一般来说，执行到这里，说明 agent 已经完成任务，可以输入下一个任务。
        # 或者 agent 有疑问，需要用户输入更多信息。
        last_message = history[-1]
        if last_message["role"] == "assistant" and last_message["content"]:
            print(last_message["content"])
