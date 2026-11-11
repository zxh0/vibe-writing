# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""第2章：测量NumPy（FP32）矩阵乘法的GFLOPS。

线程数由环境变量控制，必须在导入NumPy之前设置。macOS上的NumPy使用苹果的Accelerate库，
用VECLIB_MAXIMUM_THREADS控制；Linux上的NumPy通常使用OpenBLAS，用OPENBLAS_NUM_THREADS控制。例如：
    VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run numpy_bench.py
"""
import os
import time

import numpy as np

cfg = np.show_config(mode="dicts")
blas = cfg["Build Dependencies"]["blas"]
print(f"NumPy {np.__version__}，BLAS：{blas['name']} {blas.get('version', '')}，"
      f"线程数：{os.environ.get('VECLIB_MAXIMUM_THREADS') or os.environ.get('OPENBLAS_NUM_THREADS') or '默认'}")

rng = np.random.default_rng(0)
for n in [512, 1024, 2048, 4096]:
    A = rng.standard_normal((n, n), dtype=np.float32)
    B = rng.standard_normal((n, n), dtype=np.float32)
    A @ B  # 预热
    best = float("inf")
    for _ in range(5):  # 取五次中最快的一次
        t0 = time.perf_counter()
        A @ B
        best = min(best, time.perf_counter() - t0)
    print(f"n={n:<5} {best:8.3f} 秒  {2 * n**3 / best / 1e9:8.1f} GFLOPS")
