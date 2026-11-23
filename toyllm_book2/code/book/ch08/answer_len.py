"""第八章 · DPO之后，回答变长了吗

偏好数据里，好回答比坏回答长的占了57%。DPO会不会顺便学会“写长一点”？
取验证集里的前几十个问题，每个模型用贪婪解码各答一遍，比三样：

    平均长度    回答有多少个词元，没停下的就是上限
    自己停下    在上限之前写出<|im_end|>的比例
    重复        回答里的四元组（连续四个词元）有多少是前面出现过的，
                原地打转越厉害，这个数越大

问题只取不太长的（不超过200个词元），免得问题本身就占掉大半个窗口。

运行（在CPU上，几个模型各要几分钟）：
    cd code
    uv run python book/ch08/answer_len.py runs/sft/model runs/dpo/model
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer

from chat_template import EOT, IM_END, ChatTemplate
from config import DATA
from export import load

VAL = DATA / "ultrafeedback" / "test_prefs-00000-of-00001.parquet"


def questions(chat, n, max_len=200):
    """验证集里的前n个问题，按对话模板拼好不超过max_len个词元的。"""
    out = []
    for p in pq.read_table(VAL, columns=["prompt"])[0].to_pylist():
        messages = [{"role": "user", "content": p}]
        if len(chat.prompt(messages)) <= max_len:
            out.append(messages)
            if len(out) == n:
                break
    return out


def answer(model, chat, messages, max_new):
    """和chat.py的reply一样贪婪地往下写，但返回词元和“有没有自己停下”。"""
    ids = torch.tensor([chat.prompt(messages)])
    start = ids.shape[1]
    for _ in range(max_new):
        ids = model.generate(ids, 1, temperature=0.0)
        if ids[0, -1].item() in (IM_END, EOT):
            return ids[0, start:-1].tolist(), True
    return ids[0, start:].tolist(), False


def repeated(ids, n=4):
    """回答里的n元组，有多少是前面已经出现过的。"""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", type=Path, nargs="+", help="导出的模型目录")
    ap.add_argument("--n", type=int, default=50, help="问几个问题")
    ap.add_argument("--max-new", type=int, default=300)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    tok = Tokenizer.from_file(str(args.models[0] / "tokenizer.json"))
    chat = ChatTemplate(tok)
    qs = questions(chat, args.n)
    print(f"验证集的前{len(qs)}个问题，贪婪解码，每个最多{args.max_new}个词元\n")
    print("模型              平均长度   自己停下   重复")
    for path in args.models:
        model = load(path).eval()
        lens, stops, reps = [], 0, []
        for messages in qs:
            ids, stopped = answer(model, chat, messages, args.max_new)
            lens.append(len(ids))
            stops += stopped
            reps.append(repeated(ids))
        name = path.parent.name if path.name == "model" else path.name
        print(f"{name:<16}{sum(lens) / len(lens):>10.0f}"
              f"{stops / len(qs):>11.0%}{sum(reps) / len(reps):>7.0%}",
              flush=True)


if __name__ == "__main__":
    main()
