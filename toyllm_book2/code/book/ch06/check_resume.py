"""第六章 · 断了接着训，和不断的比一比

两份log.jsonl逐步对照：loss、梯度范数、验证loss，看是不是完全相同。
在同一台机器的CPU上，应该逐位相同；在GPU上，有些内核（比如词嵌入的反向）
累加的顺序不固定，只能做到非常接近。

运行（先不间断地训一遍，再训一遍，中途断开，接着训完）：
    cd code
    S=(--device cpu --batch 4 --seq 128 --accum 2 --steps 60 --warmup 6
       --eval-every 20 --eval-batches 4 --sample-every 30 --ckpt-every 20)
    uv run python book/ch06/train.py --name full "${S[@]}"
    uv run python book/ch06/train.py --name cut "${S[@]}" --kill-at 50
    uv run python book/ch06/train.py --name cut "${S[@]}" --resume
    uv run python book/ch06/check_resume.py full cut
"""

import argparse
import json
from pathlib import Path

from config import CODE

KEYS = {"loss": "训练loss", "grad_norm": "梯度范数", "val_loss": "验证loss",
        "samples": "生成样例"}


def read(path):
    return {r["step"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--runs", type=Path, default=CODE / "runs")
    args = ap.parse_args()
    a = read(args.runs / args.a / "log.jsonl")
    b = read(args.runs / args.b / "log.jsonl")
    assert a.keys() == b.keys(), f"步数不一样：{len(a)}行和{len(b)}行"
    worst = {k: 0.0 for k in KEYS}
    for step in sorted(a):
        for k in KEYS:
            if k not in a[step] and k not in b[step]:
                continue
            x, y = a[step].get(k), b[step].get(k)
            if k == "samples":
                worst[k] = max(worst[k], float(x != y))
            else:
                worst[k] = max(worst[k], abs(x - y) / abs(y) if x != y else 0.0)
    print(f"共{len(a)}步")
    for k, err in worst.items():
        if err == 0:
            verdict = "完全相同"
        elif k == "samples":
            verdict = "有不同"
        else:
            verdict = f"最大相对误差 {err:.1e}"
        name = KEYS[k]
        pad = 12 - sum(2 if ord(c) > 0x2e80 else 1 for c in name)   # 汉字占两列
        print(f"  {name}{' ' * pad}{verdict}")


if __name__ == "__main__":
    main()
