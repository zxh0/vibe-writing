# 第二章代码

* `agent.py`：在第一章的基础上，加了打印原始请求和响应的开关、停止原因检查、每轮的token用量和总花费。只用标准库。
* `chat_template.py`：用Qwen3的分词器重演对话模板，看消息列表怎样变成一段文本、再切成token。依赖`transformers`，用`uv run`运行。
* `demo/`：和第一章一样的小项目。

`agent.py`的配置都从环境变量读：

| 变量 | 含义 | 默认值 |
|------|------|--------|
| `OPENAI_BASE_URL` | 服务地址 | `https://api.deepseek.com` |
| `OPENAI_API_KEY` | API密钥 | 无 |
| `MODEL` | 模型名，可以用`/models`接口查 | 无 |
| `MAX_TOKENS` | 每轮最多生成多少个token | 4096 |
| `TRACE` | 设成1时打印每一轮的请求和响应正文 | 不打印 |
| `PRICE_HIT`、`PRICE_MISS`、`PRICE_OUT` | 每百万token的价格：输入命中缓存、输入未命中缓存、输出 | 0 |

运行方法和第一章一样，在`demo/`的副本里跑：

```bash
cp -r demo /tmp/demo2 && cd /tmp/demo2
TASK="Run python3 test_stats.py and fix the failing test."
TRACE=1 python3 /path/to/agent.py "$TASK"
```

想看截断的效果，把`MAX_TOKENS`设成20左右。
