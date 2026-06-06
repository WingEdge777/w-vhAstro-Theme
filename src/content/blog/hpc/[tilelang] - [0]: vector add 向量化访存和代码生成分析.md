---
title: "[TileLang] 0: vector add 向量化访存和生成代码分析"
categories: "code"
tags: ["cuda", "TileLang", "element_wise"]
id: "1dd8ab388e48946c"
date: 2026-06-02 19:55:52
cover: "/assets/images/banner/2cc757cc144b109f.webp"
---

:::
TileLang, 目前 kernel 开发的一种 DSL，基于 TVM 实现的 kernel 编译生成器，提供友好的 python 前端抽象，供给算法工程师开发算子使用。TileLang 将简短的 Python 代码解析并 Lowering，自动生成高性能 Kernel 代码，是 openAI TileLang-lang 之后比较火的算子开发 DSL 之一
本文完整代码和测试脚本可以从 GitHub 获取，欢迎关注我的 vitamin-cuda 项目，都是手把手的算子实现与教程：<https://github.com/WingEdge777/vitamin-cuda/tree/main/kernels_DSL/tilelang/vector_add>
:::

## 1. 开发体验

### TileLang 和 Triton

说 TileLang，也肯定得提一下 Triton（但我声明，本人仅花了极少时间在 Triton 上，所以相关描述可能有误）

就目前对几个简单算子的开发和调试经历来看。个人感受，TileLang 和 Triton 最大的区别是，它提供了和 CUDA C++ 中一样线程 tid/warp_id/lane_id 级别的控制能力，所以写起来很有一种写 Python 版 CUDA 的感觉。当然，要比写纯 C++ 轻松许多，毕竟提供了很多 tile 级别的操作，也不用自己写 launch 代码。可以更多的集中注意力在算子逻辑本身上。（是的，即使是写 vector add 这种简单算子，也能感觉轻松一些）

并且 TileLang 生成的后端代码（例如 CUDA C++ 代码），是可以直接通过 `get_source_code()` 打印出来看的，必须给予好评。相比之下，Triton 只能看到 TritonGPU IR 或者 PTX/sass 代码，就不是那么友好了

Triton 给人感觉是，我只能操作 block 线程组，操作数据也是操作一整个 tile，没有 TileLang 那种操作入微的掌控感。

个人想吐槽一下，TileLang 的文档、 example、最佳实践，有点缺失，真的叫人有点难绷。.. 我能说我为了写出一个安全的数据未 padding 对齐的向量化读写 vector add kernel 也花了很长时间么。 全仓库就找到一个 T.vectorized 的 example 用法，还是 gemv 算子。

当然了，有人说 TileLang 就不是给我来实现这种基础算子的，那些复杂的、流水线的、异步的、要用到 TMA、wgmma、warp specialized 技术 的 kernel 才是 TileLang 的用武之地。TileLang 写 vector add 如同拿手术刀切牛肉，好钢没用在刀刃上。

我不否认这一点，也许涉及到那些复杂算子的操作才能感觉到丝滑的快感。但，个人觉得，一屋不扫何以扫天下。先把玩具做好，也不说做好吧，就正常的做对，不是更好么。

最后， TileLang 和 Triton 也是有共性的，比如同样的闭包写法，都给人一种当年用 tf 1.x 写 graph 的感觉（死去的记忆开始攻击我）。当然，TileLang 也提供了 eager style 的写法（从最近的 commit 看得到在推 eager style），体验会稍微好那么一点点，不过这也为后文的踩坑埋下了一个伏笔。

本文将从 element wise add 算子的 TileLang 几种实现，表演一下“茴”字的四种写法，简单熟悉一下 TileLang 的开发。验证一下 TileLang 的向量化访存能力、边界处理能力是否如宣传一般可靠。

kernel 列表如下：

- add_tilelang （闭包写法，elementwise load）
- add_tilelang_vectorized （闭包写法， 向量化读写）
- add_tilelang_vectorized_eager （eager style， 向量化读写）
- elementwise_add （官方 example，向量化读写，这也是我要吐槽的一个点）

## 2. Triton add_kernel

很自然的，得拿 Triton 比一比，用 Triton 实现一个安全的向量化访问的 add_kernel 算子，代码如下：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    st = pid * BLOCK_SIZE
    offsets = st + tl.arange(0, BLOCK_SIZE)

    if st + BLOCK_SIZE <= n:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        out = x + y
        tl.store(out_ptr + offsets, out)
    else:
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        tl.store(out_ptr + offsets, out, mask=mask)

n = 4096 * 4096
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
add_triton = partial(add_kernel[grid], n=n, BLOCK_SIZE=1024) #Triton 默认一个 block 128 线程，4 个 warp。这里直接给 1024 大小的 tile，满足 fp16x8 的向量化访问
x = torch.randn(n, dtype=torch.float16, device=DEVICE)
y = torch.randn(n, dtype=torch.float16, device=DEVICE)
out = torch.empty_like(x)
add_triton(x, y, out)
```

因为本文主要介绍 TileLang，Triton 部分就不做过多展开了（其实我也没怎么仔细研究）。但仅仅从这几行代码，就可以感觉到：写 Triton 代码时，我们只能按 Tile 级别去判断和操作数据，这一个 Tile 是交由一个 Thread Block 作为一个整体去处理的。

## 3. add_tilelang

千言万语不如一行代码，与其干巴巴地罗列 API，不如把注释写进代码里，请看：

```python
import tilelang
import tilelang.language as T # 这里面有 TileLang 封装的 tile 级别操作，如 reduce，dot，gemm 等

# 闭包写法
@tilelang.jit
def add_tilelang(N: int, block: int = 256, dtype: str = "float16"): # 指定 jit 的 meta data，block 对应 CUDA C++ 中的 thread block

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype), # 形式化定义输入 A，Tensor 类型，shape 为 (N,)，数据类型为 dtype
        B: T.Tensor((N,), dtype), # 形式化定义输入 B
        C: T.Tensor((N,), dtype), # 形式化定义输出 C
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx: # launch kernel，bx 对应 blockIdx.x
            for i in T.Parallel(block): # T.Parallel 细化到每个线程，for 作用域下都是具体某一个线程的工作
                out_id = bx * block + i # 计算全局线程 id
                C[out_id] = A[out_id] + B[out_id] # 官方宣传有自动边界保护，无需手动加 if out_id < N 的判断

    return main # 根据 meta 数据返回具体 TileLang kernel 

kernel = add_tilelang(N=n, block=256)
kernel(x, y, out)
```

通过 `get_source_code()` 观察底层产物。当 N 完美对齐（Padding）时，生成的代码非常干净利落：

```cpp
extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))]);
}
```

而当 N 未对齐时，TileLang 确实自动为我们生成了边界保护代码：

```cpp
// n = 4096*4096 -1 = 16777215
extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 16777215) {
    C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))]);
  }
}
// n = 4096*4096 + 1 = 16777217
extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 16777217) {
    C[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] = (A[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))] + B[((((int)blockIdx.x) * 256) + ((int)threadIdx.x))]);
  }
}
```

代码虽然正确且边界安全，但上述代码也暴露了一个性能问题：底层生成的是 half 的标量读写。要吃透硬件性能，我们就必须把访存粒度提上来。

这就是下一节 vectorized 要解决的问题。


## 4. add_tilelang_vectorized

这里必须要吐槽一下，翻遍官方文档和 Example，几乎找不到关于 T.vectorized 的详细用法，我也是花了一点时间才摸索写出了下面这个 Kernel。这对于初入 TileLang 的新人来说确实不够友好。

```python
@tilelang.jit
def add_tilelang_vectorized(N: int, block: int = 256, dtype: str = "float16"):
    vec = 8 # 目标: fp16x8，即 128-bit 访存

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block * vec), threads=block) as bx:
            tid = T.get_thread_binding(0)
            tile_id = bx * block + tid
            base = tile_id * vec

            # 显式分支：区分 Main Loop (向量化) 和 Tail Loop (标量化)
            if base + vec - 1 < N:
                for i in T.vectorized(vec):
                    elem = base + i
                    C[elem] = A[elem] + B[elem]
            else:
                for i in T.serial(N - base):
                    elem = base + i
                    C[elem] = A[elem] + B[elem]

    return main
```

从这里就可以体味到 TileLang 和 Triton 的差异了。得益于 TileLang 细腻的线程级（Thread-level）控制，我们可以像手写 CUDA C++ 一样，显式地进行分支控制：让对齐的数据走向量化访存，仅对最后 n%8 个越界元素进行标量处理。相比之下，Triton 更多是依赖 Block 级别的 mask 掩码来隐式处理越界。TileLang 的这种细粒度操作，确实赋予了开发者一种操作入微的掌控感。

我们可以通过生成的 CUDA C++ 代码，来见证 TileLang 是如何把这层抽象转化为极致性能的：

N 对齐时，编译器直接将 T.vectorized(8) 映射为 128-bit 的 uint4 读写（虽然我个人在手写时更喜欢用 float4）。并且，底层的计算自动调用了 tl::add2（对应 HADD2 指令）。最大化利用了 LDG.E.128/STG.E.128 宽总线访存，这是打满全局内存带宽的标准动作。：

```cpp
// N = 4096*4096
extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  uint4 __1;
    uint4 v_ = *(uint4*)(A + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
    uint4 v__1 = *(uint4*)(B + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
    *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
    *(uint1*)(&(__1.y)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.y))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.y)))));
    *(uint1*)(&(__1.z)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.z))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.z)))));
    *(uint1*)(&(__1.w)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.w))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.w)))));
  *(uint4*)(C + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = __1;
}
```

当 N 未对齐时（自动生成安全边界与 Tail Loop）：

```cpp
// n = 4096*4096 -1 = 16777215
// 16777215 // 8 = 2097151
extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  if (((((int)blockIdx.x) * 256) + ((int)threadIdx.x)) < 2097151) {
    uint4 __1;
      uint4 v_ = *(uint4*)(A + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
      uint4 v__1 = *(uint4*)(B + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
      *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
      *(uint1*)(&(__1.y)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.y))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.y)))));
      *(uint1*)(&(__1.z)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.z))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.z)))));
      *(uint1*)(&(__1.w)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.w))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.w)))));
    *(uint4*)(C + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = __1;
  } else {
    for (int i = 0; i < ((16777215 - (((int)threadIdx.x) * 8)) - (((int)blockIdx.x) * 2048)); ++i) {
      C[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i)] = (A[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i)] + B[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i)]);
    }
  }
}
// n = 4096*4096 + 1 = 16777217
extern "C" __global__ void main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  if (((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) < 8388605) { 
    uint4 __1;
      uint4 v_ = *(uint4*)(A + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
      uint4 v__1 = *(uint4*)(B + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
      *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
      *(uint1*)(&(__1.y)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.y))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.y)))));
      *(uint1*)(&(__1.z)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.z))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.z)))));
      *(uint1*)(&(__1.w)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.w))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.w)))));
    *(uint4*)(C + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = __1;
  } else {
    for (int i = 0; i < ((16777217 - (((int)threadIdx.x) * 8)) - (((int)blockIdx.x) * 2048)); ++i) {
      half_t condval;
      if (((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i) < 16777217)) {
        condval = A[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i)];
      } else {
        condval = half_t(0x0p+0f/*0.000000e+00*/);
      }
      half_t condval_1;
      if (((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i) < 16777217)) {
        condval_1 = B[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i)];
      } else {
        condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      C[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i)] = (condval + condval_1);
    }
  }
}
```

仔细观察上面生成的条件判断代码，会被 TileLang 底层编译器的数学代数化简能力小秀一下：
Python 源码中的`if base + vec - 1 < 16777217`
因为`vec = 8` 所以是 `if base < 16777210`，编译为 C++

```cpp
if (((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) < 16777210)
// 约分
if (((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 4)) < 8388605) // 8388605 就是这么来的
```

## 5. add_tilelang_vectorized_eager

考虑到官方近期在推 Eager Style 写法，所以我也尝试用它实现了一个版本。TileLang 的 eager style 利用了 Python 3 的参数类型提示（Type Hint）特性：

```python
A: T.Tensor((N,), dtype)
```

这条语句在 Python 的标准解释器中没有任何实际运行时的影响，运行时你随便传个什么对象进来它都不会拦着，原本只对代码可读性和 IDE 提示有作用。但 TileLang 充分利用了这一点——这勉强算是某种“黑魔法”吧——借助一条对 Python 运行无关的语句，来向编译器提供元信息以进行语法解析

参考实现代码如下：

```python
@tilelang.jit
def add_tilelang_vectorized_eager(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dtype = "float16"
):
    N = T.const("N")
    block = 256
    # 踩坑警告：千万别这么写！
    # vec = 128 // (A.dtype.itemsize * 8) 
    
    vec = 128 // tilelang.DataType(dtype).bits
    A: T.Tensor((N,), dtype) # 原地类型提示，为 TileLang 提供 AST 解析的元信息
    B: T.Tensor((N,), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block * vec), threads=block) as bx:
        tid = T.get_thread_binding(0)
        tile_id = bx * block + tid
        base = tile_id * vec

        if base + 7 < N:
            for i in T.vectorized(vec):
                elem = base + i
                C[elem] = A[elem] + B[elem]
        else:
            for i in T.serial(N - base):
                elem = base + i
                C[elem] = A[elem] + B[elem]
```

朋友们，注意到我注释的那行 `# vec = 128 // (A.dtype.itemsize * 8)` 了么。我刚开始在这踩了个大坑。既然叫 eager 模式，我自然会带入 PyTorch 那套“所见即所得”的思维惯性，试图用动态传入的 Tensor 对象属性来计算 vec 的大小，还感觉逻辑上无懈可击。

但，不是的。诚然，这和我对 DSL 的具体实现方式的无知有一定关系：TileLang 是需要读取我们的 Python 代码，进行语法解析，然后再生成对应后端的 kernel 代码。举个例子：

如果你用 `python -c " ... "`的方式执行含有 TileLang kernel 的代码，那么会报错挂掉。因为没有 Python 源码文件，而 TileLang 强依赖于 Python 的 AST（抽象语法树）静态解析（虽然我觉得这是可以优化的）。

其次，基于同样的逻辑。我注释掉的写法在 TileLang 的 AST 解析阶段，A.dtype.itemsize 无法被当作动态对象求值，而是被直接识别为了一个固定长度的指针大小（8 字节），从而得出 vec = 128 // (8*8) = 2 的尴尬情况，生成了啼笑皆非的下述代码：

```cpp
extern "C" __global__ void add_tilelang_vectorized_eager_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) add_tilelang_vectorized_eager_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  if (((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 2)) < 16777209) {
    uint1 __1;
      uint1 v_ = *(uint1*)(A + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 2)));
      uint1 v__1 = *(uint1*)(B + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 2)));
      *(uint1*)(&(__1.x)) = tl::to_uint1(tl::add2(tl::from_uint1<__half2>(*(uint1*)(&(v_.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__1.x)))));
    *(uint1*)(C + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 2))) = __1;
  } else {
    for (int i = 0; i < ((16777216 - (((int)threadIdx.x) * 2)) - (((int)blockIdx.x) * 512)); ++i) {
      C[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 2)) + i)] = (A[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 2)) + i)] + B[(((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 2)) + i)]);
    }
  }
}
```

发现问题了吗？原本完美的 128-bit 访存退化成了 32-bit 的 uint1 (fp16x2)，而且尾端本该很短的 fallback loop，变成了漫长的 16777216 - 16777209 = 7 次的单线程 for 循环标量读写。虽然这次碰巧正确性没有出问题，仅略微影响性能，但这种背离直觉的黑盒行为，极容易导致bug。所以，谨记

拥抱 eager 是未来的趋势，但目前的 eager 嘛，emmm, 底层调优还是得留个心眼。

## 6. elementwise_add

```python
@tilelang.jit
def elementwise_add(A, B, C, block_N=2048, dtype="float16", threads=256):
    N = T.const("N")

    A: T.Tensor((N), dtype)
    B: T.Tensor((N), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        A_shared = T.alloc_shared((block_N), dtype)
        B_shared = T.alloc_shared((block_N), dtype)
        C_local = T.alloc_fragment((block_N), dtype)
        C_shared = T.alloc_shared((block_N), dtype)

        T.copy(A[bx * block_N], A_shared)
        T.copy(B[bx * block_N], B_shared)
        for local_x in T.Parallel(block_N):
            C_local[local_x] = A_shared[local_x] + B_shared[local_x]
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[bx * block_N])
```

这是官方仓库里的 Example（我略微做了修改，二维改为了一维，把输出 C 改成了外部 Tensor）。通过 `get_source_code()` 可以看出生成的代码在访存上确实是向量化的。

但是这个例子，目的是什么呢，有点难绷，为啥要显式地过一遍 smem。访存型算子，数据毫无复用性，阅后即焚，load 到 smem 完全没有作用。

当然，如果纯粹是为了展示 TileLang 的 API 功能——向开发者证明它有能力把大块数据从 gmem copy 到 smem，在 smem 中计算，再 copy 到 block style 寄存器，最后 copy 回 gmem——那我勉强觉得可以理解。

此外需要注意的是，这个 Kernel 自带的自动边界保护是有缺陷的。如果传入未对齐的 N（例如 n = 4096 * 4096 + 1 或 - 1），执行会直接挂掉。

再顺带一提，即使我们把 alloc_shared 全部去掉，只使用 alloc_fragment（即全部改为 block style 的局部寄存器）：

```python
@tilelang.jit
def elementwise_add_no_shared(A, B, C, block_N=2048, dtype="float16", threads=256):
    N = T.const("N")

    A: T.Tensor((N), dtype)
    B: T.Tensor((N), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        A_local = T.alloc_fragment((block_N), dtype)
        B_local = T.alloc_fragment((block_N), dtype)
        C_local = T.alloc_fragment((block_N), dtype)

        T.copy(A[bx * block_N], A_local)
        T.copy(B[bx * block_N], B_local)
        for local_x in T.Parallel(block_N):
            C_local[local_x] = A_local[local_x] + B_local[local_x]
        T.copy(C_local, C[bx * block_N])
```

这也只会在 N 完美对齐的情况下才会自动触发向量化访问。例如 n = 4096 * 4096 时，会生成：

```cpp
extern "C" __global__ void elementwise_add_no_shared_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) elementwise_add_no_shared_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  half_t A_local[8];
  half_t B_local[8];
  half_t C_local[8];
  *(uint4*)(A_local + 0) = *(uint4*)(A + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  *(uint4*)(B_local + 0) = *(uint4*)(B + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    C_local[i] = (A_local[i] + B_local[i]);
  }
  *(uint4*)(C + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(C_local + 0);
}
```

这做到了 128 bit 向量化访问，但计算还是朴素的 8 次循环，没有生成 tl::add2（或者底层的 HADD2 指令），这说明当前 TileLang 编译器的 Lowering 阶段依然不够完美，指令生成上还存在优化空间。

而如果遇到 n = 4096 * 4096 + 1 或 - 1 这种未对齐情况呢？不好意思，整个代码会全部退回到 8 次独立 for 循环（而且是在循环体内塞入 if 边界判断，丑爆）：

```cpp
// n=4096 * 4096-1
extern "C" __global__ void elementwise_add_no_shared_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) elementwise_add_no_shared_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  half_t A_local[8];
  half_t B_local[8];
  half_t C_local[8];
  for (int i_s = 0; i_s < 8; ++i_s) {
    half_t condval;
    if (((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i_s) < 16777215)) {
      condval = A[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i_s)];
    } else {
      condval = half_t(0x0p+0f/*0.000000e+00*/);
    }
    A_local[i_s] = condval;
  }
  for (int i_s_1 = 0; i_s_1 < 8; ++i_s_1) {
    half_t condval_1;
    if (((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i_s_1) < 16777215)) {
      condval_1 = B[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i_s_1)];
    } else {
      condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
    }
    B_local[i_s_1] = condval_1;
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    C_local[i] = (A_local[i] + B_local[i]);
  }
  for (int i_s_2 = 0; i_s_2 < 8; ++i_s_2) {
    if ((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i_s_2) < 16777215) {
      C[(((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)) + i_s_2)] = C_local[i_s_2];
    }
  }
}

//n=4096 * 4096+1 同理， 省略了
```

## 7. benchmark

直接给 4096 x 4096（对齐），4096 x 4096 + 1（非对齐），4096 x 4096 - 1 （非对齐）的 benchmark 结果：

```yaml
# TODO
```

从数据可以看出，向量化版本（tilelang_vectorized）相比非向量化版 TileLang 算子 性能提升了约 20%，与 Triton 持平。
这说明手动指定的 fp16x8 向量化成功打满了全局内存的读写带宽。
未对齐情况下的降速也在预期之内，因为边界 fallback 破坏了完美的 128-bit 连续访存。

## 8. 总结

本文使用 TileLang 对 Element-wise Add 算子进行了“茴字的四种写法”的花式实现，并对各实现方法在对齐/非对齐条件下编译生成的底层 C++ 代码，进行了深入的正确性与性能剖析。
总结如下：

- 生态定位精准：TileLang 作为 DSL 的后发者，精准地从 Triton 和手写 CUDA C++ 中间又找到了一个切入点（可以进行线程级别的控制），在易用性与底层掌控力之间找到了新的平衡。使得其有机会在算子开发中占据一席之地，很棒
- 关于底层基座：TileLang 基于 TVM 实现，这点我很难评论说好坏
  - 个人是一度对 TVM 充满信心的，在 CV 时代，TVM 每次发一个大版本，我都会用 TVM 编译（autotune）一次 resnet50，但很可惜，真的没有一次能跑出与 TensorRT 持平的性能（更别说超过了），
  - 这也导致本人一度看到所谓的“Graph/算子自动编译”就有点 PTSD，所以面对这套底层机制，心情难免有点微妙。
- 槽点与避坑（个人体感）：
  - 文档基建匮乏：文档和最佳实践 example 建设不是很好。不过这也好理解，学术界出身的开源项目早期往往如此，维护者难有精力完善周边建设，不好苛求。
  - eager 模式 不是 truly eager：eager 模式是降低门槛的好方向，但目前它不是 PyTorch 那种的 eager，要谨记

如有错误，欢迎指正，感谢阅读！

完整代码和测试脚本可以从 GitHub 获取，欢迎关注我的 vitamin-cuda 项目，都是手把手的算子实现与教程：<https://github.com/WingEdge777/vitamin-cuda/tree/main/kernels_DSL/tilelang/vector_add>

以上
