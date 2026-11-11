"""第二章 · 切出验证集

按文档切，不按token切；一开始就切好，以后不再改。

怎么做到“以后不再改”：不随机抽，而是看每篇文档的`id`。FineWeb-Edu给每篇文档
一个全局唯一的id，对它算MD5，取模1000，落在[0, 2)的就进验证集，大约千分之二。
同一篇文档，无论什么时候、在哪台机器上跑，算出来都一样，所以不用把训练集另存一份：
第三章分词时，用同一个`is_val`判断一遍就行。

验证集的正文另存一份JSON Lines，方便翻看，也方便后面的章节直接读。

运行：
    cd code
    uv run python book/ch02/split.py
"""

import hashlib
import json

import pyarrow.parquet as pq

from download import FINEWEB_DIR, shard_paths

VAL = FINEWEB_DIR / "val.jsonl"
VAL_PER_MILLE = 2


def is_val(doc_id):
    h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
    return h % 1000 < VAL_PER_MILLE


def main():
    n_train = n_val = tok_train = tok_val = 0
    with open(VAL, "w", encoding="utf-8") as out:
        for path in shard_paths():
            table = pq.read_table(path, columns=["id", "text", "token_count"])
            for doc_id, text, n in zip(*(table.column(c).to_pylist()
                                         for c in ("id", "text", "token_count"))):
                if is_val(doc_id):
                    n_val += 1
                    tok_val += n
                    out.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
                else:
                    n_train += 1
                    tok_train += n

    print(f"分片         {len(shard_paths())} 个")
    print(f"训练集       {n_train:,} 篇，{tok_train:,} token")
    print(f"验证集       {n_val:,} 篇，{tok_val:,} token（{n_val / (n_train + n_val):.2%}）")
    print("token数是GPT-2分词器数的，换成第三章的分词器会不一样。")
    print(f"\n写到 {VAL.relative_to(FINEWEB_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
