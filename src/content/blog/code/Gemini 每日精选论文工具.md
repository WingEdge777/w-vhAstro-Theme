---
title: "白嫖 github action + Gemini 自动化每日精选论文"
categories: "code"
tags: ["code"]
id: "56dd1be2094125b9"
date: 2026-04-10 23:18:27
cover: "/assets/images/banner/7b1491d13dfb97a4.webp"
---

:::note
AI coding 现在越来越火热，不可避免有点焦虑。我也 vibe coding 了一个 Gemini 每日精选论文工具（白嫖 github action 和 google AI studio 大模型 token），加紧学习效率。欢迎 star/fork 使用
:::

## 背景

随着 AI 技术的飞速发展，每天都有大量的论文发表，作为工程师或者研究人员，想要保持对最新技术的敏感度是非常重要的。尤其是对于像我这样从事系统和推理方向研究的工程师来说，自己也想保持跟进某些领域的最新进度和前沿技术。

但刷着知乎等技术板块也不是个事，有时候还是要看学术界的最新技术。

## Gemini 精选论文

因此我写了个小工具，在 github action 中，每天自动从 arxiv 捞出论文，然后白嫖 google AI studio 的模型（不得不说，这个羊毛薅的挺值的）进行打分精选，分类整合，同时还能提取摘要快速判断论文核心内容。

项目地址：<https://github.com/WingEdge777/daily-papers>

demo 如下：
![](https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/daily_papers_0.png)

![](https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/daily_papers_1.png)

## 使用配置

### 关注领域和关键字

可以自行更改研究领域关键词和 arxiv 分类

```yaml
# 研究领域关键词（用于 LLM 分类）
keywords:
  - "Large Language Models"
  - "Natural Language Processing"
  - "Vision Language Models"
  - "Diffusion Models"
  - "Multimodal"
  - "Image Generation"
  - "Video Generation"
  - "Agent"
  - "Distributed Computing"
  - "Operating Systems"
  - "Information Retrieval"
  - "Computer Vision"
  - "Machine Learning"

# ArXiv API 配置
arxiv:
  max_results: 200
  base_url: "http://export.arxiv.org/api/query"
  categories:
    - "cs.CV"
    - "cs.CL"
    - "cs.AI"
    - "cs.LG"
    - "cs.MM"
    - "cs.DC"
    - "cs.OS"
    - "cs.IR"
    - "cs.MA"
```

### LLM 和 输出论文数量

`Gemini 3.1 Flash Lite` 每天 500 次调用 (最大抓500篇文章)，足够使用了。如果想用其他模型，也可以自行更改配置

精选每个领域最多 5 篇论文，可以根据需要调整

``` yaml
# LLM 配置
llm:
  min_score: 70
  max_papers_per_keyword: 5
  rate_limit_interval: 4.1
  
  google:
    api_key: "${GOOGLE_AI_API_KEY}"
    base_url: "https://generativelanguage.googleapis.com/v1beta"
    model: "auto"
    fallback_model: "gemma-4-31b-it"
    priority_models:
      - "gemini-3.1-flash-lite-preview"
      - "gemini-3-flash-preview"
      - "gemini-2.5-flash-lite"
      - "gemini-2.5-flash"
      - "gemma-4-31b-it"
      - "gemma-4-26b-a4b-it"
      - "gemma-3-27b-it"
      - "gemma-3-12b-it"
    temperature: 0.3
    max_output_tokens: 2048
    timeout: 60
    max_retries: 3
    retry_delay_429: 10
    retry_delay_503: 10
    retry_delay_timeout: 5
```

## 结束

项目地址：<https://github.com/WingEdge777/daily-papers>

欢迎朋友们 star 或 fork 定制化使用，如有使用不便之处也欢迎提 PR 修改
