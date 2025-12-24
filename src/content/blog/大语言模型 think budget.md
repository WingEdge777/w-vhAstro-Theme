---
title: 大语言模型 think budget
categories: AI infer
tags:
  - LLM serving
  - sglang
id: "1234567"
date: 2025-10-26 17:13:27
cover: "/assets/images/banner/7b1491d13dfb97a4.webp"
---
:::note
我不知道有多少人在使用思考模型，但笔者个人对思考模型的推理性能（性能吞吐）是极度不满意的，因此笔者一直避免使用思考模型，但人在江湖身不由己，有时候不得不使用思考模型。
:::

从 OpenAI 开始提出推理模型开始，思考模型已经逐渐成为了学界和业界的热点，国内开源模型两巨头 qwen 和 deepseek，都有思考模式和非思考模式。尽管深度思考模型在推理任务上表现出色，但它们在推理过程中需要产生大量的思考信息，然后才输出最终结果。这导致了思考模型在推理任务上需要消耗大量的计算资源和时间。

本文尝试从限制 think token 长度的角度降低推理模型的推理时延。其实千问官网已经提供了带 think budget 的 demo，阿里云平台提供的 api 接口也有 thinkbudget 功能，但在当前开源 serving 框架里却依然没有一个开箱即用的实现，因此笔者分享个人的实现方法。

首先说一下 think budget 的实现，其原理非常简单，就是在模型推理时，当输出的 `token 长度` 超过 think budget 时，就强行输出 `think 停止 token id`，从而达到限制思考长度的目的。qwen 的 [官方文档](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html#thinking-budget) 对此也有详细的说明。

更具体的，笔者在此分别给出两种实现方法：

## 两种方法

### Custom Logits Processor

第一种是从 server 端考虑，我们可以添加自定义的 logits processor，具体实现和 qwen 的官方文档所说类似，需要在具体的 serving 框架中实现，其实当前 sglang 也有待 merge 的 [PR](https://github.com/sgl-project/sglang/pull/6208/files)， 但不知为何一直没有被合并。该实现看似修改代码较多，但原理依然如前文所述并不复杂。笔者还找到网上有一个公开的 transformers 库 使用的 logits processor 实现如下：

``` python
#| filename: thinking_budget_processor.py
#| language: python

from transformers.generate import LogitsProcessor

class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """
    A processor where after a maximum number of tokens are generated,
    a </think> token is added at the end to stop the thinking generation,
    and then it will continue to generate the response.
    """
    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.think_end_token = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float('-inf')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.tokens_generated += 1
        if self.max_thinking_tokens == 0 and not self.stopped_thinking and self.tokens_generated > 0:
            scores[:] = self.neg_inf
            scores[0][self.nl_token] = 0
            scores[0][self.think_end_token] = 0
            self.stopped_thinking = True
            return scores

        if self.max_thinking_tokens is not None and not self.stopped_thinking:
            if (self.tokens_generated / self.max_thinking_tokens) > .95:
                scores[0][self.nl_token] = scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                scores[0][self.think_end_token] = (
                    scores[0][self.think_end_token] * (1 + (self.tokens_generated / self.max_thinking_tokens))
                )

            if self.tokens_generated >= (self.max_thinking_tokens - 1):
                if self.tokens_generated == self.max_thinking_tokens-1:
                    scores[:] = self.neg_inf
                    scores[0][self.nl_token] = 0
                else:
                    scores[:] = self.neg_inf
                    scores[0][self.think_end_token] = 0
                    self.stopped_thinking = True

        return scores
```

以上代码转载自：<https://muellerzr.github.io/til/end_thinking.html>

已有朱玉在前，笔者也无需多说了。

### Double-Query Think Budget

第二种笔者称之为 double-query，无需修改框架或自定义 logits processor, 通过两次调用的方式，来实现 think budget。

我们将对模型的调用封装为两次调用：第一次调用时，设置 max_tokens 为 think budget 大小，收到输出结果后，先判断已输出的内容中是否已经包含了 eos 或 think 停止 token id，如果包含，则直接返回；否则，我们将进行第二次调用时，设置 max_tokens 为剩余的 token 数量，之后收到的输出结果，就是模型最终要输出的内容了。

此处以调用 sglang qwen 模型 server 代码 为例：

```python
import asyncio
import aiohttp
import json

def create_bench_client_session():
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )

api_url = "http://10.60.68.98:30000/generate"

async def run(think_budget = 512, max_new_tokens=4096):
    query = "选购手机该看哪些参数？详细介绍一下"
    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    payload = {
            "text": prompt,
            "sampling_params": {
                "top_k": 20,
                "top_p": 0.8,
                "max_new_tokens": think_budget,
                "ignore_eos": False,
            },
            "stream": False,
            "logprob_start_len": -1,
        }
    
    async with create_bench_client_session() as session:
        async with session.post(
            url=api_url, json=payload
        ) as res:
            data = await res.text()
            # print(data)
    res = json.loads(data)
    text = res["text"]
    if res["meta_info"]["finish_reason"]["type"] != "stop":
        if "</think>" in res["text"]:
            payload["text"] = prompt + res["text"]
        else:
            payload["text"] = prompt + res["text"] + "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

        payload["sampling_params"]["max_new_tokens"] = max_new_tokens - think_budget

        async with create_bench_client_session() as session:
            async with session.post(
                url=api_url, json=payload
            ) as res:
                data = await res.text()
                # print(data)
                text = text + "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n" + json.loads(data)["text"]
    print(text)
    return text

if __name__ == "__main__":
    asyncio.run(run())
```

单一 server 或结合 cache-aware 调度的 server 都可以利用前缀 kv cache, 因此开销不大。如此可以简单有效地限制最大思考长度，保障 SLA。
