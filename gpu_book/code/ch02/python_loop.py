# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""第2章：用纯Python三重循环计算矩阵乘法，测量GFLOPS，并与NumPy的结果对比。"""
import random
import time

import numpy as np


def matmul(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


random.seed(0)
for n in [64, 128, 256]:
    A = [[random.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    B = [[random.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    t0 = time.perf_counter()
    C = matmul(A, B)
    t = time.perf_counter() - t0
    err = np.abs(np.array(C) - np.array(A) @ np.array(B)).max()
    print(f"n={n:<4} {t:8.3f} 秒  {2 * n**3 / t / 1e9:.4f} GFLOPS  与NumPy的最大差异 {err:.1e}")
