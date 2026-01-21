---
title: "手把手 CUDA 编程实践"
categories: "随笔"
tags: ["vitamin-cuda","cuda","c++"]
id: "3c6cbaa1ee1bfb8c"
date: 2026-01-21 19:43:44
cover: "/assets/images/banner/8cef6fb3c78dc3ad.webp"
---

:::note
开个坑，记录一下自己 CUDA 编程的实践 kernel 实现，从易到难，由基础开发到应用优化
:::

## vitmin-cuda

好久没写过 CUDA C++ 代码了，最近重新拾起来

起因是看到了 [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 这个项目，感觉非常不错，可以作为学习 CUDA 编程的参考

同样的，本人正在开坑一个项目：[vitamin-cuda](https://github.com/wingedge777/vitamin-cuda)，主要也是 kernel 开发实践，并与 torch native 实现做对比，在项目中记录学习过程，以及一些优化实战代码

目前刚起步，后续会持续更新，并且不局限于CUDA C++，例如triton、cutlass、cute等内容也会加入进行，也会加入一些实际应用案例（比如集成到tensorRT 插件）的实现

感觉会是一个长期的大坑 + 水磨功夫的任务

欢迎大家关注，一起交流，进步！

项目地址：[vitamin-cuda](https://github.com/wingedge777/vitamin-cuda)
