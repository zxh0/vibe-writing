# /// script
# requires-python = ">=3.9"
# dependencies = ["transformers"]
# ///
"""第二章：看对话模板怎样把消息列表拼成一段文本，再切成token。

云端API看不到这一步，所以用本地的Qwen3分词器演示。只需要分词器文件，
不需要模型权重。用法：
    uv run chat_template.py                    # 从Hugging Face下载分词器
    uv run chat_template.py path/to/Qwen3-0.6B # 用本地目录
"""
import sys

from transformers import AutoTokenizer

path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)

messages = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the failing test."},
    {"role": "assistant", "content": "<bash>python3 test_stats.py</bash>"},
    {"role": "user", "content": "exit code: 1\nAssertionError"},
]

text = tok.apply_chat_template(messages, tokenize=False,
                               add_generation_prompt=True)
print(text)

ids = tok(text).input_ids
print(f"{len(ids)} tokens")
for i in ids[:12]:
    print(f"{i:>7}  {tok.convert_ids_to_tokens(i)!r}")
