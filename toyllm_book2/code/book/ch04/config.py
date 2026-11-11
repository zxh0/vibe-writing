"""第四章 · 模型配置

一个dataclass，记下模型的形状。字段名照抄SmolLM2的`config.json`，
这样两边能互相转换：SmolLM2的配置能读进来（核对用），我们的配置
存出去，第一本的引擎和`transformers`都认。

默认值就是本书的模型，约26.7M参数。
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# code/book/ch04/config.py → 上三层就是 code/
CODE = Path(__file__).resolve().parent.parent.parent
MODELS = CODE / "models"
DATA = CODE / "data"


@dataclass
class Config:
    vocab_size: int = 8192             # 第三章训练的分词器
    hidden_size: int = 512             # 每个词元用多少个数表示，即d
    num_hidden_layers: int = 8
    num_attention_heads: int = 8       # 每个头64维
    num_key_value_heads: int = 2       # GQA：每4个q头共用一组k、v
    intermediate_size: int = 1408      # SwiGLU中间那一层的宽度
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True   # 输出层和词嵌入共用一个矩阵
    initializer_range: float = 0.02    # 初始化的标准差
    bos_token_id: int = 0
    eos_token_id: int = 0              # <|endoftext|>

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def load(cls, model_dir):
        """读`config.json`，只取这里认识的字段，其余的不管。"""
        raw = json.loads((Path(model_dir) / "config.json").read_text())
        assert raw["model_type"] == "llama" and not raw.get("rope_interleaved", False)
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})

    def save(self, model_dir):
        """写`config.json`，补上`transformers`要的几个固定字段。"""
        raw = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_act": "silu",
            "attention_bias": False,
            "mlp_bias": False,
            "rope_interleaved": False,
            "rope_scaling": None,
            "torch_dtype": "float32",
            **asdict(self),
        }
        path = Path(model_dir) / "config.json"
        path.write_text(json.dumps(raw, indent=2) + "\n")
