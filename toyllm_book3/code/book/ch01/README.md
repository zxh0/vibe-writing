# 第一章代码

* `agent.py`：最小的Agent，一个`while`循环加一个`bash`工具，只用标准库。
* `demo/`：埋了一个bug的小项目，`median()`在列表长度为偶数时取错了下标。

运行前设置三个环境变量，任何兼容OpenAI Chat Completions协议的服务都可以。本书用的是DeepSeek：

```bash
export OPENAI_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY=...
export MODEL=...
```

模型名可以这样查：

```bash
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

Agent会真的执行模型给出的命令，所以先把`demo/`复制到别处，在副本里跑：

```bash
cp -r demo /tmp/demo && cd /tmp/demo
python3 test_stats.py
python3 /path/to/agent.py "Run python3 test_stats.py and fix the failing test."
```
