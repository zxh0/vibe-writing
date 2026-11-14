"""第一章：最小的Agent，一个循环加一个bash工具。

用法：在要干活的目录里运行
    python3 agent.py "任务描述"
需要三个环境变量：OPENAI_BASE_URL、OPENAI_API_KEY、MODEL。
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODEL", "")

SYSTEM = """You are a coding agent working in a terminal.
To run a shell command, put it in a bash tag, one per reply:

<bash>ls -la</bash>

The next message will show you the command's output.
Each command runs in a fresh shell, so `cd` does not persist.
When the task is done, reply with a short summary and no bash tag."""


def llm(messages):
    """把整段对话发给模型，返回它的回复。细节在第二章。"""
    body = {"model": MODEL, "messages": messages}
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def bash(command):
    """执行一条命令，把退出码、stdout和stderr拼成一段文本。"""
    try:
        r = subprocess.run(command, shell=True, capture_output=True,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "error: command timed out after 60 seconds"
    return f"exit code: {r.returncode}\n{r.stdout}{r.stderr}"


def agent(task):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    while True:
        reply = llm(messages)
        print(f"\n[assistant]\n{reply}")
        messages.append({"role": "assistant", "content": reply})
        m = re.search(r"<bash>(.*?)</bash>", reply, re.DOTALL)
        if m is None:  # 模型不再要求执行命令，任务结束
            return reply
        output = bash(m.group(1))
        print(f"\n[bash]\n{output}")
        messages.append({"role": "user", "content": output})


if __name__ == "__main__":
    agent(" ".join(sys.argv[1:]))
