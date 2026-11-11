import os
import re
import urllib.request
from gensim.models import Word2Vec

DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = "tiny_shakespeare.txt"


# Split into word and punctuation tokens; newlines become <nl>.
def tokenize(text):
    raw = re.findall(r"\n|[A-Za-z']+|[^A-Za-z'\s]", text)
    return ["<nl>" if t == "\n" else t.lower() for t in raw]


if not os.path.exists(DATA_FILE):
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
text = open(DATA_FILE, encoding="utf-8").read()

# 一行台词算一个“句子”，Word2Vec 只在句子内部找上下文
sentences = [tokenize(line) for line in text.split("\n") if line.strip()]

# 训练 Word2Vec（固定随机种子、单线程，结果才能复现）
model = Word2Vec(
    sentences=sentences,
    vector_size=50,  # 词嵌入维度
    window=5,        # 上下文窗口：前后各看 5 个词
    min_count=5,     # 出现不到 5 次的词不要
    epochs=30,
    seed=42,
    workers=1,
)

print(model.wv["king"].shape)
for word in ["father", "night", "romeo"]:
    neighbors = [(w, round(s, 3)) for w, s in model.wv.most_similar(word, topn=5)]
    print(word, neighbors)

# 类比：husband - he + she ≈ ?
analogy = model.wv.most_similar(positive=["husband", "she"], negative=["he"], topn=3)
print([(w, round(s, 3)) for w, s in analogy])
