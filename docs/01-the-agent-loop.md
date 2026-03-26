# s01: The Agent Loop (智能体循环)

`[ s01 ] s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12`

> *"One loop & Bash is all you need"* -- 一个工具 + 一个循环 = 一个智能体。
>
> **Harness 层**: 循环 -- 模型与真实世界的第一道连接。

## 问题

语言模型能推理代码, 但碰不到真实世界 -- 不能读文件、跑测试、看报错。没有循环, 每次工具调用你都得手动把结果粘回去。你自己就是那个循环。

## 解决方案

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> |  Tool   |
| prompt |      |       |      | execute |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                    (loop until there are no tool calls)
```

一个退出条件控制整个流程。循环持续运行, 直到模型不再调用工具。

## 工作原理

1. 用户 prompt 作为第一条消息。

```python
messages.append({"role": "user", "content": query})
```

2. 将消息和工具定义一起发给 LLM。这里用的是 OpenAI 兼容的 Chat Completions 接口。

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": SYSTEM}, *messages],
    tools=TOOLS,
    max_tokens=8000,
)
```

3. 取出本轮助手消息。检查 `tool_calls` -- 如果模型没有调用工具, 结束。

```python
message = response.choices[0].message
messages.append({"role": "assistant", "content": message.content or ""})
if not message.tool_calls:
    return
```

4. 执行每个工具调用, 把结果作为 `tool` 消息追加。回到第 2 步。

```python
for tool_call in message.tool_calls:
    arguments = json.loads(tool_call.function.arguments)
    output = run_bash(arguments["command"])
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": output,
    })
```

组装为一个完整函数:

```python
def agent_loop(query):
    messages = [{"role": "user", "content": query}]
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            max_tokens=8000,
        )
        message = response.choices[0].message
        messages.append({"role": "assistant", "content": message.content or ""})

        if not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            output = run_bash(arguments["command"])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })
```

不到 30 行, 这就是整个智能体。后面 11 个章节都在这个循环上叠加机制 -- 循环本身始终不变。

## 变更内容

| 组件          | 之前       | 之后                           |
|---------------|------------|--------------------------------|
| Agent loop    | (无)       | `while True` + `tool_calls`    |
| Tools         | (无)       | `bash` (单一工具)              |
| Messages      | (无)       | 累积式消息列表                 |
| Control flow  | (无)       | `if not message.tool_calls`     |

## 试一试

```sh
cd learn-claude-code
python agents/s01_agent_loop.py
```

试试这些 prompt (英文 prompt 对 LLM 效果更好, 也可以用中文):

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`
4. `Create a directory called test_output and write 3 files in it`

## 笔记

输入 `create file python.py that prints "Hello World!" 后，agent loop 的工作流程如下：

1. main 把 SYSTEM 定义 (You are a coding agent...) 和用户消息 (creat file ...) 放入消息列表，交给 agent_loop，把消息列表发给 LLM。
2. LLM 的 reasoning 是 "I will use bash to create the file"，要求使用 bash，执行命令 `cat > python.py << EOF\nprint('Hello World')\nEOF`。
3. agent_loop 调用 run_bash 执行命令，创建文件，输出为 `(no content)`。
4. agent_loop 把工具调用和结果追加到消息列表，再次发给 LLM。
5. LLM 的 reasoning 是 "Let me verify it was created correctly"，要求使用 bash，执行命令 `cat python.py`。
6. agent_loop 调用 run_bash 执行命令，输出为 `print('Hello World')`。
7. agent_loop 把工具调用和结果追加到消息列表，再次发给 LLM。
8. LLM 的 reasoning 是 "Let me verify it runs correctly"，要求使用 bash，执行命令 `python python.py`。
9. agent_loop 调用 run_bash 执行命令，输出为 `Hello, World!`。
10. agent_loop 把工具调用和结果追加到消息列表，再次发给 LLM。
11. LLM 这次直接回复了 content（没有 reasoning，也没有工具调用）："Done! File is created successfully"。
12. agent_loop 把助手消息追加到消息列表，结束循环，回到 main。
13. main 打印消息列表的最后一条消息的 content，即 LLM 的回复 "Done! File is created successfully"。
14. main 进入 loop，等待用户输入新的 prompt。

## 小结

- 有两个 loop
  - agent loop 的作用是：当模型需要调用工具时，执行工具，把结果发还给模型继续任务，直到模型不再调用工具。
  - main loop 的作用是：当 agent loop 结束时，等待用户新的输入（可能是上一个任务需要用户更多输入才能继续，也可能是上一个任务结束了，等待新的任务）。
