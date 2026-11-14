# LLM API：一次请求

第一章的`llm`函数是个黑盒子：消息列表进去，一段文本出来。这一章把它打开，不用SDK，看看每一轮发出去的是什么，收回来的是什么，又花了多少钱。打开以后，我们会顺手修好第一章留下的一个bug，还会发现Agent和普通聊天程序在花钱这件事上有一个很大的不同。

本章的代码在第一章的基础上改，放在`code/book/ch02/`目录下。模型用的是DeepSeek V4 Flash，通过DeepSeek提供的OpenAI兼容接口调用。



## 打开黑盒子

要看清一次请求，最直接的办法就是把它原样打印出来。我们给`llm`函数加一个开关：设置了环境变量`TRACE=1`，就把每一轮的请求正文和响应正文都打印出来。另外，函数的返回值也从“回复的文本”改成了“完整的响应JSON”，因为这一章要用到响应里的其他字段：

```python
def show(title, data):
    print(f"\n>>> {title}\n" + json.dumps(data, indent=2, ensure_ascii=False))


def llm(messages):
    """把整段对话发给模型，返回完整的响应JSON。"""
    body = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS}
    if TRACE:
        show("request", body)
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + API_KEY},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    if TRACE:
        show("response", data)
    return data
```

和第一章相比，请求正文里多了一个`max_tokens`，后面会专门讲它。下面先看请求，再看响应。

### 请求长什么样

一次请求由三部分组成：地址、请求头和正文。地址是服务的根地址加上`/chat/completions`，用`POST`方法。DeepSeek的根地址是`https://api.deepseek.com`，换成别家的服务，只要改环境变量`OPENAI_BASE_URL`，代码不用动。

请求头只有两个。`Content-Type`说明正文是JSON；`Authorization`里放的是API密钥，格式是`Bearer`加一个空格再加密钥。密钥等于钱，谁拿到了谁就能用我们的账户调用模型，所以它只从环境变量里读，既不写进代码，也不打印出来。这也是`show`只打印正文、不打印请求头的原因。

正文是一个JSON对象。第一轮请求的正文如下（系统提示词太长，这里省略了后半段）：

```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {
      "role": "system",
      "content": "You are a coding agent working in a terminal.\n..."
    },
    {
      "role": "user",
      "content": "Run python3 test_stats.py and fix the failing test."
    }
  ],
  "max_tokens": 4096
}
```

（待核实：`model`字段的值，以DeepSeek的`/models`接口返回的为准。）

正文里最重要的就是`messages`，也就是第一章那个消息列表，原封不动地发了过去。这里没有会话编号，也没有“上次聊到哪了”之类的东西。服务器拿到的就是这么一个列表，它不知道、也不关心这是第几轮。

### 响应长什么样

响应也是一个JSON对象。按照Chat Completions协议，它的结构大致是下面这样，具体的值这里先用省略号代替：

```json
{
  "id": "...",
  "object": "chat.completion",
  "created": ...,
  "model": "...",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": ...,
    "completion_tokens": ...,
    "total_tokens": ...
  }
}
```

（待补：用DeepSeek实际跑一次，把`TRACE=1`打印出的第一轮响应原样贴在这里，替换上面的示意。DeepSeek在`usage`里还会多给几个字段，见下一节。）

在`id`、`created`、`model`这几个字段里，`id`是这次请求的编号，出了问题找客服时用得上；`created`是时间戳；`model`是实际处理请求的模型名。Agent真正要用的，是`choices`和`usage`。



## 响应里有什么

响应里除了模型说的话，还有两样Agent离不开的东西：模型为什么停下来，这一轮用了多少token。三样东西分别看。

### 生成的内容

模型说的话在`choices[0].message.content`里。`choices`是一个列表，因为协议允许一次请求生成好几个候选回复（参数`n`），但Agent从来只要一个，所以总是取第0个。

返回的`content`不一定有值。第四章用正式的工具调用时，模型有时只给出工具调用，不说话，`content`就是`null`。所以第二章的代码里写成了`choice["message"]["content"] or ""`，没有内容时当成空字符串处理。另外，推理模型在给出回复之前会先“想”一段，这段思考内容放在哪个字段、要不要发回去，各家的做法不一样，第五章再讲。

### 停止原因

模型为什么停下来，写在`finish_reason`里。常见的取值有下面几种：

| 取值 | 含义 |
|------|------|
| `stop` | 模型自己说完了，生成了表示结束的特殊token，或者碰到了停止词 |
| `length` | 说到一半，到了`max_tokens`的上限，被服务器截断了 |
| `tool_calls` | 模型要调用工具，第四章会用到 |
| `content_filter` | 内容被服务方的安全过滤拦下了 |

第一章的代码完全没看这个字段。它默认模型每次都是自己说完的，这个假设在大多数时候成立，但不总是成立，下一节就会看到它造成的bug。

### 用量

这一轮用了多少token，写在`usage`里。`prompt_tokens`是输入的token数，也就是整个消息列表变成token以后有多长；`completion_tokens`是输出的token数，也就是模型这次生成了多少个token；`total_tokens`是两者之和。

另外，输入的token里有一部分可能命中了服务器的缓存，命中缓存的部分计费更便宜。这个数各家都会报告，但字段名不一样：DeepSeek直接在`usage`里给出`prompt_cache_hit_tokens`和`prompt_cache_miss_tokens`，OpenAI则放在`prompt_tokens_details.cached_tokens`里。我们写一个小函数，把两种写法统一起来：

```python
def cached_tokens(usage):
    """输入里命中缓存的token数，各家的字段名不一样。"""
    if "prompt_cache_hit_tokens" in usage:  # DeepSeek
        return usage["prompt_cache_hit_tokens"]
    details = usage.get("prompt_tokens_details") or {}  # OpenAI
    return details.get("cached_tokens") or 0
```

缓存是怎么回事、为什么能便宜，留到第十三章再讲。这一章只需要知道，它会影响我们算钱。



## 第一章的一个bug

有了停止原因，就能看出第一章的循环有一个bug。假设模型的回复很长，说到一半碰到了`max_tokens`的上限，被截断在这里：

```text
Let me run the tests first.
<bash>python3 test_st
```

这条回复里只有开头的`<bash>`，没有结尾的`</bash>`，第一章的正则表达式匹配不到。于是循环认为模型不再要求执行命令，任务完成了，高高兴兴地退出。用户看到的是一个什么都没做就结束了的Agent，而且没有任何报错。

这个bug平时不容易碰到，因为一条命令通常很短，远远到不了上限。但在两种情况下它会冒出来：一是模型在给出命令之前写了一大段分析，二是模型想用一条命令写出一个很长的文件。在第六章之前，我们的Agent改文件全靠`cat`加重定向，第二种情况其实并不少见。

修起来很简单：先看`finish_reason`，是`length`的话，就不能当成说完了。我们把被截断的回复照样追加到消息列表里，再告诉模型它的回复被截断了，让它重发一个短一点的：

```python
TRUNCATED = ("Your last reply was cut off by the max_tokens limit. "
             "Please send it again, shorter.")
```

```python
        if choice["finish_reason"] == "length":  # 被截断，不能当成说完了
            messages.append({"role": "user", "content": TRUNCATED})
            continue
```

另一种办法是调大`max_tokens`，但上限总是有的，只是碰到得晚一些。真正的Agent还会限制每轮输出的长度，防止模型一次写出太长的东西，这里先用最简单的办法。

要亲眼看到这个bug，可以把`MAX_TOKENS`设得很小，比如20，分别跑一下第一章和第二章的代码：

```bash
TASK="Run python3 test_stats.py and fix the failing test."
MAX_TOKENS=20 python3 agent.py "$TASK"
```

第一章的代码不认`MAX_TOKENS`这个环境变量，所以要对比的话，得临时在第一章的请求正文里加上`"max_tokens": 20`。

（待补：两边实际运行的输出。第一章的代码预期在第一轮就退出，第二章的代码预期会反复收到截断提示。）



## 每轮重发

模型没有记忆，第一章已经讲过：每次请求，都要把整段对话从头发过去。第一章只是把它当成一个事实，这一节来看它的代价。

### 输入token一轮比一轮多

第二章的代码每一轮都会打印用量。用第一章那个修bug的任务跑一遍，就能看到每一轮的输入token数：

（待补：DeepSeek实际运行时每一轮的`[usage]`输出，以及最后的`[total]`一行。）

不管模型具体走了几轮，输入token的变化规律都一样：第一轮最少，只有系统提示词和任务；此后每一轮，都比上一轮多出一条模型的回复和一条命令的输出。输出token则没有这个规律，每轮多少取决于模型这一轮说了多少话。

### 平方增长

这个规律可以算一算。设系统提示词和任务一共有 $s$ 个token，之后每一轮新增的回复和命令输出平均有 $k$ 个token。那么第 $i$ 轮的输入是 $s+(i-1)k$ 个token，一个任务跑了 $n$ 轮，输入token的总数就是：

$$
\sum_{i=1}^{n} \big(s+(i-1)k\big) = ns + \frac{n(n-1)}{2}k
$$

第二项里有 $n^2$ ，所以当轮数多起来以后，输入token的总数大致随轮数的平方增长。代入一组假设的数字感受一下：设 $s=300$ ， $k=500$ 。跑4轮，总输入是 $1200+3000=4200$ 个token；跑40轮，总输入是 $12000+390000=402000$ 个token。轮数多了10倍，输入token多了将近100倍。

这就是Agent和普通聊天程序在花钱上最大的不同。聊天程序其实也是每轮重发整段对话，但一次聊天往往只有几轮，每轮也只有几句话，平方增长还没来得及显现。Agent就不一样了：一个稍复杂的任务动辄几十轮，每一轮的命令输出又可能很长，比如一次测试的完整报错、一个几百行的文件。所以对Agent来说，输入token才是大头，输出token反而是小头。

好在这里有一个特点可以利用：每一轮的请求，开头都和上一轮一模一样，只是在末尾多了两条消息。服务器的缓存正是冲着这个特点来的，命中缓存的输入token要便宜得多。第十三章会细讲。



## 怎么算钱

模型服务按token计费，一般按每百万token报价，而且分三种价格：输入里命中缓存的、输入里没命中缓存的，以及输出。三者通常差别很大，命中缓存的输入最便宜，输出最贵。

一个任务花了多少钱，就是把每一轮的三种token数加起来，再分别乘以对应的价格：

$$
\text{花费} = \frac{h \cdot p_{\mathrm{hit}} + m \cdot p_{\mathrm{miss}} + o \cdot p_{\mathrm{out}}}{10^6}
$$

其中 $h$ 、 $m$ 、 $o$ 分别是整个任务里命中缓存的输入、没命中缓存的输入和输出的token数， $p_{\mathrm{hit}}$ 、 $p_{\mathrm{miss}}$ 、 $p_{\mathrm{out}}$ 是对应的每百万token的价格。

价格会变，不同模型也不一样，所以代码里不写死，而是从环境变量里读：

```python
# 每百万token的价格，按所用模型的官方价格表填写
PRICE_HIT = float(os.environ.get("PRICE_HIT", "0"))    # 输入，命中缓存
PRICE_MISS = float(os.environ.get("PRICE_MISS", "0"))  # 输入，未命中缓存
PRICE_OUT = float(os.environ.get("PRICE_OUT", "0"))    # 输出
```

（待核实：DeepSeek V4 Flash的三种价格，以DeepSeek官网的价格页为准，写进正文时注明查询日期。）

累计用量和计算花费的工作交给一个小类`Meter`。每一轮调用一次`add`，任务结束时调用`report`打印总数：

```python
class Meter:
    """累计每一轮的token用量，算出一共花了多少钱。"""

    def __init__(self):
        self.turns = 0
        self.hit = self.miss = self.out = 0

    def add(self, usage):
        hit = cached_tokens(usage)
        self.turns += 1
        self.hit += hit
        self.miss += usage["prompt_tokens"] - hit
        self.out += usage["completion_tokens"]
        print(f"\n[usage] turn {self.turns}: "
              f"input {usage['prompt_tokens']} (cached {hit}), "
              f"output {usage['completion_tokens']}")

    def cost(self):
        return (self.hit * PRICE_HIT + self.miss * PRICE_MISS
                + self.out * PRICE_OUT) / 1e6

    def report(self):
        print(f"\n[total] {self.turns} turns, "
              f"input {self.hit + self.miss} (cached {self.hit}), "
              f"output {self.out}, cost {self.cost():.4f}")
```

（待补：修bug那个任务实际花了多少钱。）



## 从消息到token

前面一直在说“消息列表变成token”，可模型只认识一个token序列，消息列表又是怎么变成token序列的呢？答案是**对话模板**（chat template）。服务器收到消息列表以后，先用对话模板把它拼成一段文本，再用分词器把这段文本切成token，最后才交给模型。

这一步发生在服务器上，云端API是看不到的。好在开源模型的对话模板是公开的，我们可以用本地的分词器把这一步重演一遍。下面的脚本用的是Qwen3的分词器，它只需要分词器的几个文件，不需要模型权重：

```python
path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)

messages = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the failing test."},
    {"role": "assistant", "content": "<bash>python3 test_stats.py</bash>"},
    {"role": "user", "content": "exit code: 1\nAssertionError"},
]

text = tok.apply_chat_template(messages, tokenize=False,
                               add_generation_prompt=True)
print(text)

ids = tok(text).input_ids
print(f"{len(ids)} tokens")
for i in ids[:12]:
    print(f"{i:>7}  {tok.convert_ids_to_tokens(i)!r}")
```

这里的消息列表模仿了第一章Agent的第二轮请求：系统提示词、任务、模型要执行的命令、命令的输出。为了让输出短一点，内容都做了简化。运行结果如下：

```text
<|im_start|>system
You are a coding agent.<|im_end|>
<|im_start|>user
Fix the failing test.<|im_end|>
<|im_start|>assistant
<bash>python3 test_stats.py</bash><|im_end|>
<|im_start|>user
exit code: 1
AssertionError<|im_end|>
<|im_start|>assistant

53 tokens
 151644  '<|im_start|>'
   8948  'system'
    198  'Ċ'
   2610  'You'
    525  'Ġare'
    264  'Ġa'
  10822  'Ġcoding'
   8315  'Ġagent'
     13  '.'
 151645  '<|im_end|>'
    198  'Ċ'
 151644  '<|im_start|>'
```

可以看到，对话模板做的事情其实很朴素：每条消息前面加上`<|im_start|>`和角色名，后面加上`<|im_end|>`，然后首尾相接。`<|im_start|>`和`<|im_end|>`是两个特殊token，各自只占一个ID（151644和151645），不会被切碎。`Ċ`是换行符，`Ġ`表示前面有一个空格，这是字节级BPE分词器的写法，第一本的分词一章讲过。

最值得注意的是最后一行：模板在末尾多加了一个`<|im_start|>assistant`和换行，这是参数`add_generation_prompt=True`的作用。它相当于替模型起了个头：“下面轮到assistant说话了”。模型从这里开始一个token一个token地往下预测，直到它自己预测出`<|im_end|>`，服务器就知道它说完了，这时的`finish_reason`就是`stop`。如果它一直没有预测出`<|im_end|>`，直到数量达到`max_tokens`，那就是`length`。

还有一点：我们约定的`<bash>`标签，在模型眼里就是几个普通的token，和别的文字没有任何区别。模型之所以会按约定输出它，全靠系统提示词里的那几行说明。第四章的正式工具调用，会在这个模板里留下不一样的痕迹。

不同的模型，对话模板长得不一样。DeepSeek用的是自己的模板，云端API看不到，但服务器做的是同样的事情：把消息列表拼成一段文本，切成token，交给模型。`usage`里的`prompt_tokens`，数的就是这段文本切出来的token。所以消息列表的token数，总是比各条消息内容的token数之和多一点，多出来的是模板加上的这些特殊token和角色名。上面的例子里，四条很短的消息，一共是53个token。



## 请求参数

请求正文里能放的参数不少，对Agent最要紧的是长度上限，其余的简单交代一下就行。

### 长度上限

也就是`max_tokens`：模型这一次最多生成多少个token。它只管输出，不管输入。我们的代码默认设成4096，可以用环境变量`MAX_TOKENS`修改。

设得太小，回复容易被截断，就会碰到前面那个bug；设得太大，单次请求的花费上限就高，模型一旦开始胡言乱语，要说很久才会停。另外要注意，`max_tokens`只是一个上限，不是目标，模型不会因为上限大就多说。OpenAI的新接口里，这个参数改名叫`max_completion_tokens`了，第五章比较各家协议时再说。

### 采样参数

温度、top-p、停止词，各自对应第一本采样一章的哪一步：`temperature`是在softmax之前把logits除以一个温度，温度越低，输出越确定；`top_p`是只在概率累计到 $p$ 的那几个候选token里抽样；`stop`是一组停止词，模型一旦生成了其中某个字符串就立刻停下，此时的`finish_reason`也是`stop`。

我们的代码一个都没设，全用服务器的默认值。这是有意的：对Coding Agent来说，服务方给出的默认值通常就是调过的，没有明确的理由，不去动它。而且有的推理模型不接受这些采样参数，传了会被忽略，甚至报错。（待核实：DeepSeek V4 Flash对这几个参数的处理。）

停止词倒是有一个有趣的用法。如果把`stop`设成`["</bash>"]`，模型写完一条命令就会立刻停下，不会接着往下编“命令的输出大概是这样”。不过停止词本身不会出现在回复里，解析时要记得这一点。



## 上下文窗口

模型能处理的token数是有上限的，叫**上下文窗口**。一次请求的输入加上输出，不能超过这个上限。超过了，服务器会直接拒绝这次请求，返回一个错误。（待核实：DeepSeek V4 Flash的上下文窗口大小。）

对Agent来说，这不是一个遥远的上限。前面算过，消息列表每轮都在变长；一次失败的测试、一个大文件的内容，就可能占去几千甚至几万个token。任务一长，消息列表迟早会撑满上下文窗口。在撑满之前，还有两个问题会先冒出来：一是越来越贵，这一章已经算过；二是越来越慢，因为模型在生成第一个token之前，要先把整个输入处理一遍，输入越长，等得越久，第三章会实测这个延迟。

到了那一步怎么办？是把前面的对话总结一下，还是丢掉一些旧的命令输出？这是第十三章上下文工程要回答的问题。



## 代码实现

第二章的代码在`code/book/ch02/`目录下，`agent.py`是在第一章的基础上改的。前面已经逐段讲过改动的部分，这里汇总一下：

* `llm`函数返回完整的响应JSON，并且在设置了`TRACE=1`时打印请求和响应的正文
* 请求正文里加上了`max_tokens`
* 循环里先检查`finish_reason`，被截断时提醒模型重发
* 新增`cached_tokens`函数和`Meter`类，每轮打印用量，结束时打印总花费

所有的配置都从环境变量读，开头一段是这样的：

```python
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODEL", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
TRACE = os.environ.get("TRACE") == "1"
```

循环本身变化不大：每一轮先把用量记到`Meter`里，再检查停止原因，最后才去找命令；任务结束时打印总花费。完整的样子如下：

```python
def agent(task):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    meter = Meter()
    while True:
        data = llm(messages)
        choice = data["choices"][0]
        reply = choice["message"]["content"] or ""
        meter.add(data["usage"])
        print(f"\n[assistant]\n{reply}")
        messages.append({"role": "assistant", "content": reply})
        if choice["finish_reason"] == "length":  # 被截断，不能当成说完了
            messages.append({"role": "user", "content": TRUNCATED})
            continue
        m = re.search(r"<bash>(.*?)</bash>", reply, re.DOTALL)
        if m is None:  # 模型不再要求执行命令，任务结束
            meter.report()
            return reply
        output = bash(m.group(1))
        print(f"\n[bash]\n{output}")
        messages.append({"role": "user", "content": output})
```

除此之外，目录里还有一个`chat_template.py`，就是前面重演对话模板的那个脚本。它依赖`transformers`，用`uv run chat_template.py`运行，第一次会自动安装依赖、下载分词器。`agent.py`仍然只用标准库。



## 代码效果

和第一章一样，先把`demo/`复制一份，在副本里运行。这次打开`TRACE`，并且填上价格：

```bash
cp -r demo /tmp/demo2 && cd /tmp/demo2
export PRICE_HIT=... PRICE_MISS=... PRICE_OUT=...
TASK="Run python3 test_stats.py and fix the failing test."
TRACE=1 python3 /path/to/agent.py "$TASK"
```

（待补：DeepSeek实际运行的输出。`TRACE`的输出很长，正文里只摘第一轮完整的请求和响应，以及每一轮的`[usage]`行和最后的`[total]`行。）

（待补：根据实际输出，对照“每轮重发”一节的规律讲一讲：输入token是不是一轮比一轮多，命中缓存的有多少，总花费是多少。）



## 本章小结

这一章打开了`llm`这个黑盒子。一次请求就是一个HTTP的`POST`：请求头里放密钥，正文里放模型名、整个消息列表和`max_tokens`；响应里有模型的回复、停止原因和用量。服务器在背后用对话模板把消息列表拼成一段文本，切成token，再交给模型。

对Agent来说，这一章有三个结论。第一，不能只看回复的文本，还要看停止原因，被截断的回复不能当成任务完成。第二，模型没有记忆，每一轮都要重发整段对话，一个任务的输入token总数大致随轮数的平方增长，输入才是花钱的大头。第三，消息列表迟早会撑满上下文窗口。

到目前为止，我们的Agent每一轮都要等模型把整段回复生成完，屏幕上才会出现文字。下一章换成流式输出，让文字一个token一个token地出现。
