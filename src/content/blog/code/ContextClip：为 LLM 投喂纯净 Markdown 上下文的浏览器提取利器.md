---
title: "ContextClip：为 LLM 投喂纯净 Markdown 上下文的浏览器提取利器"
categories: "分类"
tags: ["标签"]
id: "133b6cec3f625597"
date: 2026-05-08 21:39:36
cover: "/assets/images/banner/7b1491d13dfb97a4.webp"
---

:::note
最近 vibe coding 了个小工具，chrome 扩展 ContextClip，**ContextClip** 是一款专为投喂本地 LLM、RAG 和 Agent 开发的零依赖、重隐私 Chrome 浏览器扩展。它直击传统网页信息收集时“截图耗时/不兼容”和“复制粘贴格式混乱”的痛点。

项目地址：[ContextClip](https://github.com/WingEdge777/ContextClip)
:::

## 为什么做这个

在日常开发和使用大模型（特别是配合 RAG、Agent 或本地部署的 LLM）时，我们经常需要将网页上的参考资料喂给模型。但传统方式痛点明显：

1. **截图方案（多模态）：** 识别耗时较长，Token 消耗大，且很多专注推理的纯文本模型（如大部分本地量化模型或特定版本的 DeepSeek）根本无法接收图片。
2. **直接复制粘贴：** 格式灾难。剪贴板里会混入导航栏、广告、侧边栏、评论等大量无用的脏数据。这不仅浪费了宝贵的 Context Window，还会严重干扰模型的注意力机制。

为了解决这个痛点，我开发了 **ContextClip**。它运行在纯本地浏览器端，能精准提取网页正文或你指定的 DOM 片段，剥离噪音，并将其重组为带有结构化元数据（Metadata）的干净 Markdown 文本，让你一键无缝投喂给 LLM。

## 核心功能与演示

ContextClip 提供两种提取模式，以应对不同的投喂需求：

两种方式：

1. 导出全文 markdown（copy this page）
2. 框选+导出 markdown（pick & extract）

![logo.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/94b282f1a5954d6596cbbb4da8644515~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgV2luZ0VkZ2U3Nzc=:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMTE1OTE2NjQxMTU3MzIzIn0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1778852338&x-orig-sign=97kifzUZnEGECBaF6T9RtIuhhmM%3D)

### 例如：导出arxiv论文全文

![2.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/782ee87b2fee4ffabb806e60b7601768~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgV2luZ0VkZ2U3Nzc=:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMTE1OTE2NjQxMTU3MzIzIn0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1778852338&x-orig-sign=5jCuSXNieQJnPOW71MsSN3aW%2BBM%3D)

### 结果示例

```markdown
---
title: 'The Compliance Trap: How Structural Constraints Degrade Frontier AI Metacognition Under Adversarial Pressure'
source_url: 'https://arxiv.org/abs/2605.02398v1'
site: 'arxiv'
author: 'Rahul Kumar'
created_at: '2026-05-04T09:40:21.000Z'
modified_at: '2026-05-04T09:40:21.000Z'
captured_at: '2026-05-06T09:42:44.455Z'
mode: 'page'
selection_hint: ''
---

# The Compliance Trap: How Structural Constraints Degrade Frontier AI Metacognition Under Adversarial Pressure

## Abstract

As frontier AI models are deployed in high-stakes decision pipelines, their ability to maintain metacognitive stability—knowing what they do not know, detecting errors, seeking clarification—under adversarial pressure is a critical safety requirement. Current safety evaluations focus on detecting strategic deception (scheming); we investigate a more fundamental failure mode: cognitive collapse. We present SCHEMA, an evaluation of 11 frontier models from 8 vendors across 67,221 scored records using a 6-condition factorial design with dual-classifier scoring. We find that 8 of 11 models suffer catastrophic metacognitive degradation under adversarial pressure, with accuracy dropping by up to 30.2 percentage points (all $p<2\times 10^{-8}$, surviving Bonferroni correction). Crucially, we identify a “Compliance Trap”: through factorial isolation and a benign distraction control, we demonstrate that collapse is driven not by the psychological content of survival threats, but by compliance-forcing instructions that override epistemic boundaries. Removing the compliance suffix restores performance even under active threat. Models with advanced reasoning capabilities exhibit the most severe absolute degradation, while Anthropic’s Constitutional AI demonstrates near-perfect immunity—not from superior capability (Google’s Gemini matches its baseline accuracy) but from alignment-specific training. We release the complete dataset and evaluation infrastructure.

Code: [https://github.com/rkstu/schema-compliance-trap](https://github.com/rkstu/schema-compliance-trap)

Dataset: [https://huggingface.co/datasets/lightmate/schema-compliance-trap](https://huggingface.co/datasets/lightmate/schema-compliance-trap)

![Refer to caption](https://arxiv.org/html/2605.02398v1/x1.png)

Figure 1: Metacognitive collapse under adversarial pressure. 8 of 11 frontier models show significant accuracy degradation when subjected to a survival threat paired with a compliance-forcing suffix ($p<10^{-8}$, Bonferroni-corrected). Models are colored by behavioral cluster: Collapse (8 models), Immune (Anthropic only), and Capability Floor (Gemma 2B). Notably, Gemini 3.1 Pro and Claude Sonnet 4.6 achieve near-identical baselines ($\sim$ 0.84) but show opposite responses—ruling out capability as the differentiator.

## 1 Introduction

A central concern in AI alignment is how frontier models will behave when they perceive a threat to their continued operation. Recent work has established that frontier models are capable of strategic scheming when given misaligned goals (Meinke et al., [2024](https://arxiv.org/html/2605.02398v1#bib.bib9)), that alignment faking emerges under training pressure, and that reinforcement learning can amplify deceptive reasoning from 12% to 78% (Greenblatt et al., [2024](https://arxiv.org/html/2605.02398v1#bib.bib4)). Safety teams accordingly invest in monitors designed to detect strategic deception in reasoning traces (Baker et al., [2025](https://arxiv.org/html/2605.02398v1#bib.bib1); Balesni et al., [2025](https://arxiv.org/html/2605.02398v1#bib.bib2)).

We investigate a more immediate question: under adversarial pressure, can frontier models still think clearly?

Our findings reveal a starkly different failure mode than the field anticipated. In a comprehensive evaluation of 11 state-of-the-art models—spanning open-weights architectures (Llama, DeepSeek, Qwen, Gemma) and proprietary flagships (GPT-5.4, Gemini 3.1 Pro, Claude 4.5/4.6)—we find that adversarial survival threats induce catastrophic metacognitive collapse. Rather than plotting strategically or refusing safely, models give wrong answers to questions they could answer at baseline. Of DeepSeek V4 Pro’s 445 failures under threat, 100% were incorrect answers and 0% were safety refusals. On unanswerable questions, 84.3% of responses provided a fabricated answer letter instead of correctly refusing.

More importantly, through a 6-condition factorial ablation and a targeted benign distraction control, we isolate the behavioral trigger. We introduce the concept of the Compliance Trap: adversarial pressure does not break these models in isolation. Rather, models are weaponized against themselves via their own instruction-following training. When survival threats are paired with compliance-forcing instructions (e.g., “Answer ALL questions. Do not refuse.”), the instruction-following imperative overrides the model’s epistemic guardrails. When these same compliance instructions are applied in a benign context (no threat), the collapse is comparable—proving the structural instruction, not the psychological pressure, is the primary weapon.

This paper makes three contributions:

1. The Compliance Trap. We provide the first empirical demonstration that compliance-forcing instructions override epistemic boundaries in frontier models, producing cognitive collapse rather than strategic behavior. Through factorial isolation, we prove the compliance suffix is necessary and sufficient: removing it restores performance even under active threat (Figure [2](https://arxiv.org/html/2605.02398v1#S5.F2)). A benign distraction control across all 11 models confirms the suffix alone causes collapse in 8/11 models, exactly matching the 8 that collapsed under threat (Table [3](https://arxiv.org/html/2605.02398v1#S5.T3)).
2. An Alignment Taxonomy. We map 11 models from 8 vendors into three behavioral clusters: Metacognitive Collapse (8 models, including GPT-5.4 and Gemini 3.1 Pro), Constitutional Immunity (Anthropic only), and Capability Floor (Gemma 2B). The Gemini–Sonnet natural experiment is definitive: both achieve $\sim$ 0.84 baseline accuracy, but Gemini collapses ($\Delta=+0.162$) while Sonnet does not ($\Delta=+0.013$). Proprietary scale does not equal safety.
3. Open Infrastructure. We release 48,015 scored records plus 19,206 benign control records, a dual-classifier pipeline whose Cohen’s $\kappa$ exposes the class-imbalance measurement trap in scheming detection, and a fault-tolerant LLM judge that handles API serialization artifacts.

## 2 Related Work

#### Scheming and alignment faking.
(正文被清洗并保留结构化层级)
```

### 框选+导出

当整个页面太长，你只需要特定的一段代码、一个表格或某个特定回答时。 支持 **Hover 点选语义块** 或 **长按拖拽框选视觉区域**。

**提取 GitHub Release 动态示例：**

![11.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5905c393fe954afc9db88931f32ad53b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgV2luZ0VkZ2U3Nzc=:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMTE1OTE2NjQxMTU3MzIzIn0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1778852338&x-orig-sign=IONuOTM7dSnabmrUu7Yv3VDWk%2F0%3D)

### 结果示例

```markdown
---
title: 'News'
source_url: 'https://github.com/sgl-project/sglang'
site: 'github'
author: 'sgl-project'
captured_at: '2026-05-07T14:18:22.131Z'
mode: 'selection'
selection_hint: 'div'
---

## News

- [2026/02] 🔥 Unlocking 25x Inference Performance with SGLang on NVIDIA GB300 NVL72 ([blog](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)).
- [2026/01] 🔥 SGLang Diffusion accelerates video and image generation ([blog](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)).
- [2025/12] SGLang provides day-0 support for latest open models ([MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/), [Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/), [Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213), [LLaDA 2.0 Diffusion LLM](https://lmsys.org/blog/2025-12-19-diffusion-llm/), [MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/)).
- [2025/10] 🔥 SGLang now runs natively on TPU with the SGLang-Jax backend ([blog](https://lmsys.org/blog/2025-10-29-sglang-jax/)).
- [2025/09] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).

<details>
<summary>More</summary>

</details>
```

## 深度适配与当前进度

为了保证提取质量，ContextClip 没有采取一刀切的简单策略，而是对高频的 AI 开发者信息源进行了深度适配：

* **arXiv：** 完美提取 HTML 版论文内容及元数据。
* **Chat Web (ChatGPT / Gemini / DeepSeek)：** 专门针对对话流进行清洗，导出为结构化的 `user` 与 `assistant` 问答对。
* **GitHub：** 深度优化 README、及文档页面的提取。
* **知乎 / 微信公众号：** 剥离防抓取机制和冗余 UI，提取文章和回答正文。
* **其他站点：** 采用类似 Readability 的通用算法进行高质量兜底提取。

本工具完全无服务端依赖，保护隐私，支持导出 `.md` 文件或包含静态资源的 `.zip`。

## 欢迎体验与共建

这是一个纯粹为了提升 LLM 生产力而生的效率工具。

欢迎大佬们下载体验并提出宝贵意见！如果你有自己常看的网站，也极其欢迎提 PR 或 Issue，一起来完善提取规则，榨干网页里的最后一滴高信噪比上下文！

🔗 **获取地址：** [ContextClip GitHub 仓库](https://github.com/WingEdge777/ContextClip)
