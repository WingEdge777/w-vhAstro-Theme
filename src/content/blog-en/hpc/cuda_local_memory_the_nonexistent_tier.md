---
title: "[CUDA Basics] Understanding CUDA's \"Nonexistent\" Memory Tier: Local Memory"
categories: "code"
tags: ["vitamin-cuda","cuda","c++", "GPU"]
id: "7f4c2d9a1e6b8c30"
date: 2026-04-01 19:44:28
cover: "/assets/images/banner/7b1491d13dfb97a4.webp"
---

:::note
There are countless articles online introducing NVIDIA's memory hierarchy, but most focus on global memory, shared memory, constant memory, texture memory, L2/L1 cache, and registers. Local memory is mentioned far less often. Let's skip the boilerplate and cut straight to the chase: a practical understanding of local memory..
:::

## 0. Concept

First, local in local memory is **logically local**. For each thread, local memory is private. While its logical scope is strictly per-thread, its physical storage actually resides in the device's global memory space. Accesses to local memory follow global memory rules, including going through L1 and L2 cache. NVIDIA also guarantees that local memory is laid out as consecutive 32-bit words by thread ID in global memory. So warp accesses to local memory can be naturally coalesced.

However, remember this: global memory latency is typically hundreds of times higher than register latency, and even L1/L2 cache latency is often tens of times higher than register latency. So even when local memory accesses are coalesced, and even when they hit L1/L2, they are still much slower than register accesses.

**If your intent is to keep variables in registers, you should aggressively avoid letting the compiler place them into local memory.**

### When local memory appears

In the CUDA Programming Guide, NVIDIA describes local memory this way:

> Local memory is thread local storage similar to registers and managed by NVCC, but the physical location of local memory is in the global memory space. The ‘local’ label refers to its logical scope, not its physical location. Local memory is used for thread local storage during the execution of a kernel. Automatic variables that the compiler is likely to place in local memory are:
>
> - Arrays for which it cannot determine that they are indexed with constant quantities,
> - Large structures or arrays that would consume too much register space,
> - Any variable if the kernel uses more registers than available, that is register spilling.
> Because the local memory space resides in device memory, local memory accesses have the same latency and bandwidth as global memory accesses and are subject to the same requirements for memory coalescing as described in Coalesced Global Memory Access. Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address, such as the same index in an array variable or the same member in a structure variable.

The official guidance says the compiler may place variables into local memory in three common cases:

- Arrays that cannot be proven to be indexed by compile-time constants.
  - This wording is subtle. It means array indices are not compile-time constants. For example:

```cpp
__global__ void kernel() {
    float arr[4];
    int idx = threadIdx.x % 4;
    arr[idx] = 1.0f;        // ❌ idx is not constant -> arr goes to local memory
}
```

- Large structures or arrays that would consume too many registers.
- Variables that exceed register limits, causing **register spilling**.

Items 2 and 3 look related, but they are not identical. Item 2 means the compiler may proactively judge that a large struct/array is unlikely to stay in physical registers and place it directly in local memory. Item 3 is a hard hardware limit: even if your code is clean and well-structured, once register usage exceeds the GPU limit, the compiler must spill some variables into local memory.

### Other trigger scenarios

Besides the three official cases above, there are additional situations that may lead to local memory allocation:

- Some complex math functions (for example, `sin()` / `cos()`) may implicitly use local memory in certain implementation paths.
- A common pitfall: taking an address (`&`) of a variable may force it into local memory.
- Other situations where the compiler decides it cannot reliably optimize/match data to physical registers.

The second item is a particularly hidden trap.

## 1. Pitfall example

When trying to improve global memory throughput with vectorized fp16/bf16 load/store, you might naturally write:

```cpp
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__global__ void load_fp16x8(const half* input) {
  half2 pack[4];
  int idx = threadIdx.x * 8;
  FLOAT4(pack[0]) = FLOAT4(input[idx]);  // ❌ taking address -> forced into local memory
  // later compute now uses local memory
}
```

**How can you confirm local memory usage?**

Compile with `-Xptxas -v` and inspect output. If you see lines like:

```bash
xxx bytes stack frame / xxx bytes cumulative stack size # stack vars, also local memory

xxx bytes lmem # shorthand for local memory

xxx bytes spill stores/loads # register spilling
```

Then yes, you got hit.

### Correct patterns

- Option 1: use native vector types directly.
- Option 2: use a `union` for strong binding (still based on native vector types).

```cpp
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// Option 1: use native vector type directly
__global__ void load_fp16x8(const half* input) {
  float4 pack;
  pack = FLOAT4(input[idx]);  // ✅ loaded directly into float4 register value
  // follow-up: split pack.x, pack.y ... to half2
}

// Option 2: use union-based strong binding
union alignas(16) Pack128 {
    float4 f4;
    half2 h2[4];
};
__global__ void load_fp16x8(const half* input) {
  Pack128 pack;
  pack.f4 = FLOAT4(input[idx]);  // ✅ loaded directly into float4 register value
  // follow-up: use pack.h2
}
```

A natural question follows: if you use non-idiomatic pointer reinterpret casts, will it always spill to local memory?

Not necessarily. Modern compilers are very good at static analysis and may recover simple cases, pulling values back into physical registers. Still, do not rely on this. Compiler behavior here can be fragile and unpredictable across contexts and versions.

Two reasons:

1. The optimization might fail.
2. Even if the value stays in registers, you can still trigger severe logic bugs.

Why? Physical registers do not have a true memory address. If a value stays in registers while you reinterpret and access it through different pointer types (such as writing through `float4*` and reading through `half2*`), this can become undefined behavior (UB). Under strict aliasing, the compiler may reorder instructions aggressively, treat accesses as unrelated, and you may end up reading garbage register bits and produce NaNs. These ghost bugs are often much harder to debug than straightforward performance regressions.

## 2. Validation with code

Let's verify with a concrete example:

```cpp
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

__device__ __noinline__ void scale_by_ptr(float4 *ptr) {
    half2 *h2_ptr = reinterpret_cast<half2 *>(ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        h2_ptr[i] = __hadd2(h2_ptr[i], h2_ptr[i]);
    }
}

__device__ __noinline__ float4 scale_by_val(float4 val) {
    union { float4 f; half2 h[4]; } tmp;
    tmp.f = val;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp.h[i] = __hadd2(tmp.h[i], tmp.h[i]);
    }
    return tmp.f;
}

// common style; syntactically unsafe, but compiler may optimize it back to registers
__global__ void load_fp16x8_native_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    half2 pack[4];
    FLOAT4(pack[0]) = FLOAT4(input[idx]); // ❌ address taken -> local memory candidate

    FLOAT4(output[idx]) = FLOAT4(pack[0]); // ❌ address taken -> local memory candidate
}

// bad example: external call and pointer escape block optimization back to registers
__global__ void load_fp16x8_bad_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    half2 pack[4];
    FLOAT4(pack[0]) = FLOAT4(input[idx]);               // ❌ address taken
    scale_by_ptr(reinterpret_cast<float4 *>(&pack[0])); // ❌ pointer escape

    FLOAT4(output[idx]) = FLOAT4(pack[0]); // ❌ address taken
}
// good example
__global__ void load_fp16x8_good_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    float4 pack = FLOAT4(input[idx]); // ✅ value copy only, no pointer trace
    pack = scale_by_val(pack);

    FLOAT4(output[idx]) = pack; // ✅ value copy only, no pointer trace
}
```

Compile output:

```bash
ptxas info    : 28 bytes gmem
ptxas info    : Compiling entry function '_Z23load_fp16x8_good_kernelP6__halfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z23load_fp16x8_good_kernelP6__halfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers
ptxas info    : Compile time = 4.757 ms
ptxas info    : Function properties for _Z12scale_by_val6float4
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Compiling entry function '_Z22load_fp16x8_bad_kernelP6__halfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z22load_fp16x8_bad_kernelP6__halfS0_i
    16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 16 bytes cumulative stack size
ptxas info    : Compile time = 1.667 ms
ptxas info    : Function properties for _Z12scale_by_ptrP6float4
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Compiling entry function '_Z25load_fp16x8_native_kernelP6__halfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z25load_fp16x8_native_kernelP6__halfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers
ptxas info    : Compile time = 0.767 ms
```

From these logs:

- `good_kernel` and `native_kernel` both show 0-byte stack frame, meaning values stayed in (or were optimized back to) registers.
- `bad_kernel`, where pointer escape occurs, shows a clear 16-byte stack frame and 16-byte cumulative stack size, validating local memory spill caused by address-taking/reinterpret path.

## 3. Warning signs

In practice, most developers almost never intend to use local memory directly. What matters more is avoiding **accidental local memory allocation**, especially these high-frequency traps:

- **Arrays with non-constant indices**: for example `arr[idx]` with non-compile-time `idx`, which may push the whole array to local memory.
- **Register spilling**: if register demand exceeds hardware limits, the compiler spills variables to local memory.
- **Taking addresses**: `&variable` or pointer-style access may force variables into local memory.

Combined with aggressive compiler optimization, if you take addresses and do cross-type pointer reinterpretation for vectorization (for example write via `float4*`, read via `half2*`), you may receive two different "surprises":

- Performance collapse (local memory spill): when kernel logic is complex and pointers escape, compiler tracking fails, and variables are forced into local memory.
- Logic explosion (silent NaNs): if powerful SROA optimizations keep values in physical registers, **strict aliasing** can become fatal.
  - Compiler assumption: "written through `float4*`, now read through `half*`; different types, so likely unrelated."
  - For performance, it may move register reads before writes. You then read uninitialized garbage bits and produce NaNs, which can corrupt the whole computation.

## 4. Conclusion

This post introduced the concept of local memory in CUDA, common trigger scenarios, and a concrete code sample that validates how to identify local memory spill and the underlying UB risks.

Local memory is not always a villain. In some cases (for example arrays too large for registers), using it appropriately can still be practical. Likewise, to balance register usage and occupancy, we may intentionally constrain block residency with `__launch_bounds__`, which can indirectly accept controlled register pressure/spill trade-offs.

CUDA 13 even introduces explicit register spilling into shared memory in some scenarios, see: <https://developer.nvidia.com/blog/how-to-improve-CUDA-kernel-performance-with-shared-memory-register-spilling/>

If you spot any mistakes or have suggestions, feedback is always welcome. The full kernels and test code for this post are available in my GitHub [vitamin-cuda project](https://github.com/WingEdge777/vitamin-cuda/tree/main/samples/local_memory), which features a growing collection of hand-tuned, step-by-step CUDA examples and optimizations.

That's all.
