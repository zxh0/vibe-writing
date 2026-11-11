#!/bin/sh
# 下载 SmolLM2-135M，放到 code/models/SmolLM2-135M/ 下面。
# 第四章拿它和我们自己的模型对答案：同一份权重，loss 和梯度要和 transformers 对上。
# 用到四个文件，一共约 271 MB，最大的是 model.safetensors，269 MB。
# 这个目录不进入版本控制，所以拿到代码之后要先跑一遍。
#
#     cd code
#     ./download_model.sh
#
# 已经下过的文件会跳过，中途断了再跑一遍会接着下。
# 连不上 HuggingFace 的话，换个镜像站：
#
#     HF_ENDPOINT=https://hf-mirror.com ./download_model.sh
set -eu

REPO=HuggingFaceTB/SmolLM2-135M
ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
DIR=$(cd "$(dirname "$0")" && pwd)/models/${REPO##*/}

echo "仓库   $REPO"
echo "来源   $ENDPOINT"
echo "目录   $DIR"
echo

mkdir -p "$DIR"

for f in config.json generation_config.json tokenizer.json model.safetensors; do
    if [ -f "$DIR/$f" ]; then
        # 花括号不能省：紧跟中文的 $f 会被 shell 当成变量名的一部分
        echo "跳过 ${f}（已存在）"
        continue
    fi
    echo "下载 $f"
    # -L 跟随跳转，权重本体存在 CDN 上；-C - 断点续传；--fail 让 HTTP 错误变成
    # 非零退出码，而不是把一页 HTML 错误信息存成权重文件。
    if ! curl -L -C - --fail --progress-bar \
         -o "$DIR/$f.part" "$ENDPOINT/$REPO/resolve/main/$f"; then
        echo
        echo "下载失败。可以换镜像站：HF_ENDPOINT=https://hf-mirror.com ./download_model.sh"
        exit 1
    fi
    # 下完了才改成正式名字。中途断掉不会留下一个看着是好的、其实缺一截的文件。
    mv "$DIR/$f.part" "$DIR/$f"
done

echo
ls -l "$DIR"
echo
echo "下一步：uv run python book/ch04/check_smollm2.py"
