"""第七章 · 对话模板：把一段对话拼成一串词元，再标出哪些词元要算loss

格式是ChatML，和SmolLM2、Qwen3的对话版一样。每一轮是：

    <|im_start|>角色\\n内容<|im_end|>\\n

角色是system、user、assistant之一。两个特殊词元是第三章训练分词器时
预留的，编号1和2，预训练时一次也没出现过。

拼的时候，同时给每个词元记一个掩码：1表示这个词元要模型学着说出来，
0表示它只是上下文。只有助手的回答（连同结尾的<|im_end|>）是1：
模型要学的是怎么回答，以及答完了要停下；问题怎么问，不用它学。

运行（拼一段两轮的对话，逐个词元列出掩码）：
    cd code
    uv run python book/ch07/chat_template.py
"""

from tokenizers import Tokenizer

from config import DATA

EOT, IM_START, IM_END = 0, 1, 2     # <|endoftext|>、<|im_start|>、<|im_end|>
TOKENIZER = DATA / "tokenizer" / "tokenizer.json"


class ChatTemplate:
    def __init__(self, tok):
        self.tok = tok
        assert tok.token_to_id("<|im_start|>") == IM_START
        assert tok.token_to_id("<|im_end|>") == IM_END
        # 每一轮开头的“<|im_start|>角色\n”是固定的，先分好词。
        self.header = {role: tok.encode(f"<|im_start|>{role}\n").ids
                       for role in ("system", "user", "assistant")}
        self.newline = tok.encode("\n").ids

    def render(self, messages, bodies=None):
        """一段对话 → (ids, mask)，两个一样长的列表。

        每轮分三块分词再接起来：开头、内容、结尾。内容单独分词，
        回答的边界才不会和前后的字符粘成一个词元。bodies是事先分好词的
        内容，批量处理时用，省得一条一条地分。"""
        if bodies is None:
            bodies = [self.tok.encode(m["content"]).ids for m in messages]
        ids, mask = [], []
        for m, body in zip(messages, bodies):
            learn = int(m["role"] == "assistant")
            ids += self.header[m["role"]] + body + [IM_END] + self.newline
            mask += ([0] * len(self.header[m["role"]]) + [learn] * len(body)
                     + [learn] + [0] * len(self.newline))
        return ids, mask

    def prompt(self, messages):
        """推理时用：拼好前面的对话，再接上助手那一轮的开头，等模型开口。"""
        ids, _ = self.render(messages)
        return ids + self.header["assistant"]


def main():
    tok = Tokenizer.from_file(str(TOKENIZER))
    chat = ChatTemplate(tok)
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And of Italy?"},
        {"role": "assistant", "content": "Rome."},
    ]
    ids, mask = chat.render(messages)
    text = tok.decode(ids, skip_special_tokens=False)
    print(text, end="")
    print(f"\n共{len(ids)}个词元，其中{sum(mask)}个算loss\n")
    print("  掩码  编号  词元")
    for i, m in zip(ids, mask):
        print(f"  {m:>4}{i:>6}  {tok.id_to_token(i)!r}")


if __name__ == "__main__":
    main()
