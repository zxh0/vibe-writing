# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""用三种视角计算同一个矩阵乘法，并与NumPy的结果对比。"""
import numpy as np


def matmul_dot(A, B):
    """点积视角：C[i][j]是A的第i行与B的第j列的点积。"""
    M, K, N = len(A), len(B), len(B[0])
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matmul_outer(A, B):
    """外积视角：C是K个外积之和，第k个外积由A的第k列和B的第k行构成。"""
    M, K, N = len(A), len(B), len(B[0])
    C = [[0.0] * N for _ in range(M)]
    for k in range(K):
        for i in range(M):
            for j in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matmul_block(A, B, bs):
    """分块视角：把矩阵切成bs×bs的小块，小块之间按矩阵乘法的规则相乘、相加。"""
    M, K, N = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(0, M, bs):
        for j in range(0, N, bs):
            for k in range(0, K, bs):
                C[i:i+bs, j:j+bs] += A[i:i+bs, k:k+bs] @ B[k:k+bs, j:j+bs]
    return C


A = [[1, 2, 3],
     [4, 5, 6]]
B = [[7, 8],
     [9, 10],
     [11, 12]]
print("点积视角:", matmul_dot(A, B))
print("外积视角:", matmul_outer(A, B))
print("NumPy   :", (np.array(A) @ np.array(B)).tolist())

rng = np.random.default_rng(0)
A = rng.standard_normal((8, 8)).astype(np.float32)
B = rng.standard_normal((8, 8)).astype(np.float32)
C = matmul_block(A, B, bs=4)
print("分块视角与NumPy的最大差异:", np.abs(C - A @ B).max())
