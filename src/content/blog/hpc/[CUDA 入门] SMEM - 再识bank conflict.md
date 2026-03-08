---
title: "[CUDA 入门] L1/TEX/SMEM - 再识bank conflict"
categories: "code"
tags:  ["vitamin-cuda","cuda","c++", "GPU"]
id: "bff10dfdf376e83f"
date: 2026-03-06 20:40:38
cover: "/assets/images/banner/2cc757cc144b109f.webp"
---

:::note
网上介绍和解决bank conflict的文章不胜枚举。我也不想多言，但是最近确实学到了一点新理解。有关 bank conflict 详细理解和分析，不要看乱七八糟的博客了，可以直接参考 NV 技术报告：<https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/>
:::

## 0. 序

之前文章超越cuBLAS矩阵乘法中，通过 swizzling解决bank conflict后我虽然很确信没有冲突，但是ncu profile还是报shared storage bank conflict (尽管占读写wavefronts总量比例不高，~1.9%)，最后，经过反复试验发现注释掉Bs smem写入就没冲突了。当时也不理解，以为ncu判定规则问题，等文章发表后经评论区大佬提醒，才恍然大悟，还存在warp间的访问冲突。

现代gpu架构L1/TEX/Smem 都划归到一整块SRAM上的，一个SM独占SRAM，而且单个SM上都有多个sub-core调度器（一般4个），确实会存在多个 warp 瞬时并行访问 L1/TEX/SMEM 的问题。这里决定做一个测试再次验证一下。

## 1. kernel 验证代码

写了个搬运数据的 kernel

```cpp
// load fp32x4
__global__ void load_fp32x4_kernel(float *a, float *b, int n) {
    __shared__ float s[512];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = FLOAT4(a[idx * 4]);
    val.x *= 2;
    FLOAT4(s[threadIdx.x * 4]) = val;
    __syncthreads();

    FLOAT4(b[idx * 4]) = FLOAT4(s[threadIdx.x * 4]);
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int N = a.size(0);                                                                                       \
        const int threads_per_block = 128;                                                                             \
        const dim3 blocks_per_grid = N / num / threads_per_block;                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), N); \
    }
```

kernel 代码很简单，就是搬运数据随便加点计算然后写到smem最后写回。

1. 测试
kernel代码是完美对齐的float4向量化读写(1个request 对应 4 wavefronts), 理论无冲突。这里我们做小、中、大 3 种数据规模的试验：

```python
    for sz in [512, 512*128, 512*128*128]:
        a = torch.randn(sz).float().cuda()
        b = torch.zeros_like(a)
        lib.load_fp32x4(a, b)
        # print(b)
```

用ncu profile一下：

```bash
ncu -k regex:"load" \
--metrics \
smsp__inst_executed_op_shared_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
-f python test_for_ncu.py
```

输出：
![惊不惊喜](https://pic2.zhimg.com/v2-2f5da1fd4b549e40de3f6a0885f6b4fd_1440w.jpg)

惊不惊喜
三个不同规模的数据实验，前两个小规模的是 0 冲突，最后一个大规模的就有问题了。266148 wavefronts 有 4004 次冲突（1.5%），嘿嘿~

总结
这种冲突，额，目前没有看到有说解决的办法，底层物理调度无可避免，比例也不是很高。个人觉得只能尽量降低冲突概率（比如向量化访问或者使用ldmatrix命令等，让单次访问请求密度更高，但整体请求更分散，容易错开），再就是通过流水线计算时延隐藏来掩盖这种冲突开销。（现在异步拷贝+计算重叠已经是 kernel 优化必备了）

更多有关 bank conflict 详细理解和分析，不要看乱七八糟的博客了，可以直接参考 NV 技术报告：<https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/>

完整测试代码可以从github获取，同时欢迎关注我的手撕算子系列项目vitamin-cuda，共同交流学习进步！

<https://github.com/WingEdge777/vitamin-cuda/blob/main/samples/bank_conflict_ncu/readme.md>

如有问题，欢迎指正！感谢

以上
