"""第二章：打开llm这个黑盒子，看清每一轮的请求、响应、用量和花费。

在第一章的基础上加了三样东西：
1. TRACE=1时，把每一轮的请求和响应原样打印出来；
2. 检查停止原因，回复被max_tokens截断时不当成任务完成；
3. 每一轮打印token用量，最后算出一共花了多少钱。

用法：在要干活的目录里运行
    python3 agent.py "任务描述"
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODEL", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
TRACE = os.environ.get("TRACE") == "1"

# 每百万token的价格，按所用模型的官方价格表填写
PRICE_HIT = float(os.environ.get("PRICE_HIT", "0"))    # 输入，命中缓存
PRICE_MISS = float(os.environ.get("PRICE_MISS", "0"))  # 输入，未命中缓存
PRICE_OUT = float(os.environ.get("PRICE_OUT", "0"))    # 输出

SYSTEM = """You are a coding agent working in a terminal.
To run a shell command, put it in a bash tag, one per reply:

<bash>ls -la</bash>

The next message will show you the command's output.
Each command runs in a fresh shell, so `cd` does not persist.
When the task is done, reply with a short summary and no bash tag."""

TRUNCATED = ("Your last reply was cut off by the max_tokens limit. "
             "Please send it again, shorter.")


def show(title, data):
    print(f"\n>>> {title}\n" + json.dumps(data, indent=2, ensure_ascii=False))


def llm(messages):
    """把整段对话发给模型，返回完整的响应JSON。"""
    body = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS}
    if TRACE:
        show("request", body)
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    if TRACE:
        show("response", data)
    return data


def cached_tokens(usage):
    """输入里命中缓存的token数，各家的字段名不一样。"""
    if "prompt_cache_hit_tokens" in usage:  # DeepSeek
        return usage["prompt_cache_hit_tokens"]
    details = usage.get("prompt_tokens_details") or {}  # OpenAI
    return details.get("cached_tokens") or 0


class Meter:
    """累计每一轮的token用量，算出一共花了多少钱。"""

    def __init__(self):
        self.turns = 0
        self.hit = self.miss = self.out = 0

    def add(self, usage):
        hit = cached_tokens(usage)
        self.turns += 1
        self.hit += hit
        self.miss += usage["prompt_tokens"] - hit
        self.out += usage["completion_tokens"]
        print(f"\n[usage] turn {self.turns}: "
              f"input {usage['prompt_tokens']} (cached {hit}), "
              f"output {usage['completion_tokens']}")

    def cost(self):
        return (self.hit * PRICE_HIT + self.miss * PRICE_MISS
                + self.out * PRICE_OUT) / 1e6

    def report(self):
        print(f"\n[total] {self.turns} turns, "
              f"input {self.hit + self.miss} (cached {self.hit}), "
              f"output {self.out}, cost {self.cost():.4f}")


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
    meter = Meter()
    while True:
        data = llm(messages)
        choice = data["choices"][0]
        reply = choice["message"]["content"] or ""
        meter.add(data["usage"])
        print(f"\n[assistant]\n{reply}")
        messages.append({"role": "assistant", "content": reply})
        if choice["finish_reason"] == "length":  # 被截断，不能当成说完了
            messages.append({"role": "user", "content": TRUNCATED})
            continue
        m = re.search(r"<bash>(.*?)</bash>", reply, re.DOTALL)
        if m is None:  # 模型不再要求执行命令，任务结束
            meter.report()
            return reply
        output = bash(m.group(1))
        print(f"\n[bash]\n{output}")
        messages.append({"role": "user", "content": output})


if __name__ == "__main__":
    agent(" ".join(sys.argv[1:]))
