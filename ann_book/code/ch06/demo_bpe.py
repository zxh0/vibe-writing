import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.n_vocab)
# 50257

text = "Uneasy lies the head that wears a crown."
ids = tokenizer.encode(text)
print(ids)
# [52, 710, 4107, 7363, 262, 1182, 326, 17326, 257, 12389, 13]
print([tokenizer.decode([i]) for i in ids])
# ['U', 'ne', 'asy', ' lies', ' the', ' head', ' that', ' wears', ' a', ' crown', '.']
