import numpy as np


# 手写实现，用于逐步演示余弦相似度的计算过程。
# NumPy 有现成实现：vec_a @ vec_b / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
def cos_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = sum(x * x for x in vec_a) ** 0.5
    norm_b = sum(y * y for y in vec_b) ** 0.5
    return dot / (norm_a * norm_b)


vec_cat = np.array([0.9, 0.1, 0.8])
vec_dog = np.array([0.8, 0.2, 0.9])
vec_one = np.array([0.1, 0.9, 0.0])
print(round(cos_sim(vec_cat, vec_dog), 3))  # 0.99
print(round(cos_sim(vec_cat, vec_one), 3))  # 0.165

vec_oh_cat = np.array([1, 0, 0])
vec_oh_dog = np.array([0, 1, 0])
print(cos_sim(vec_oh_cat, vec_oh_dog))      # 0.0


vocab = ["cat", "dog", "one", "two"]
token_to_id = {token: i for i, token in enumerate(vocab)}

# 嵌入矩阵，形状是 d×V（3×4），每一列是一个词元的词嵌入
mat_e = np.array([[0.9, 0.8, 0.1, 0.2],
                  [0.1, 0.2, 0.9, 0.8],
                  [0.8, 0.9, 0.0, 0.1]])


# 查表：词元 ID 是几，就取第几列
def embed(token: str) -> np.ndarray:
    return mat_e[:, token_to_id[token]]


print(embed("dog"))              # [0.8 0.2 0.9]

vec_one_hot = np.eye(len(vocab))[token_to_id["dog"]]
print(vec_one_hot)               # [0. 1. 0. 0.]
print(mat_e @ vec_one_hot)       # [0.8 0.2 0.9]
