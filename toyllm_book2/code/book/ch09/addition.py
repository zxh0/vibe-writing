"""第九章 · 加法题：出题、标准回答、判分

题目是“What is 345 + 678?”，标准回答是“345 + 678 = 1023”。
两个加数的位数可以不同，位数决定难度。

判分只看两件事：回答自己停下了（写出了<|im_end|>），而且回答里
最后一个数等于正确答案。对了给1，错了给0，没有部分分。这样的奖励
用不着奖励模型，一行代码就能算，叫可验证的奖励。

运行（出几道题，给几个回答判分）：
    cd code
    uv run python book/ch09/addition.py
"""

import random
import re


def make(rng, da, db):
    """出一道题：两个加数分别是da位、db位。1位数包括0。"""
    def number(d):
        return rng.randint(0 if d == 1 else 10 ** (d - 1), 10 ** d - 1)
    return number(da), number(db)


def question(a, b):
    return f"What is {a} + {b}?"


def solution(a, b):
    """标准回答，冷启动SFT用。"""
    return f"{a} + {b} = {a + b}"


def reward(text, a, b, stopped):
    """1：自己停下，而且最后一个数是a + b；否则0。"""
    numbers = re.findall(r"\d+", text)
    return float(stopped and bool(numbers) and int(numbers[-1]) == a + b)


def main():
    rng = random.Random(0)
    for da, db in [(1, 1), (2, 1), (2, 2), (3, 3), (4, 4)]:
        a, b = make(rng, da, db)
        print(f"{da}位 + {db}位  {question(a, b):<26}{solution(a, b)}")
    a, b = 345, 678
    print()
    for text, stopped in [("345 + 678 = 1023", True),
                          ("The answer is 1023.", True),
                          ("345 + 678 = 1013", True),
                          ("345 + 678 = 1023", False),
                          ("To add these two numbers, we", False)]:
        print(f"  奖励 {reward(text, a, b, stopped):.0f}  "
              f"{'停下' if stopped else '没停'}  {text}")


if __name__ == "__main__":
    main()
