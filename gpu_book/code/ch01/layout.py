# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""观察矩阵在内存中的布局：行主序、列主序、stride和转置。"""
import numpy as np

A = np.arange(6, dtype=np.float32).reshape(2, 3)
print("A =\n", A)
print("行主序在内存中的顺序:", A.ravel(order="K").tolist())
print("shape:", A.shape, " strides:", A.strides)

F = np.asfortranarray(A)
print("列主序在内存中的顺序:", F.ravel(order="K").tolist())
print("shape:", F.shape, " strides:", F.strides)

T = A.T
print("A.T =\n", T)
print("A.T shape:", T.shape, " strides:", T.strides)
print("A.T和A共享内存:", np.shares_memory(A, T))
