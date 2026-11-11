// 第6章：每个SM上驻留的warp越多，越能把延迟藏起来。
//
// 用法：./warps read <每个block的线程数>   访存：每个线程读一个数，等它回来，再读下一个
//       ./warps fma  <每个block的线程数>   计算：每个线程做一条首尾相接的FMA链
//
// 启动的block数等于SM数，每个SM正好一个block，所以每个block的线程数决定了
// 每个SM上有几个warp：32个线程是1个warp，1024个线程是32个warp。
#include "common.cuh"
#include <cstring>

__global__ void read_kernel(const float *a, size_t n, float *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float s = 0;
#pragma unroll 1
    for (; i < n; i += stride) s += a[i];   // 加法要等读回来，所以每个warp同一时刻只有一个读请求在路上
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void fma_kernel(int iters, float x, float y, float *out) {
    float a = threadIdx.x;
    for (int i = 0; i < iters; i++) a = fmaf(a, x, y);   // 每条FMA都要等上一条的结果
    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "read";
    int threads = argc > 2 ? atoi(argv[2]) : 1024;
    int blocks = sm_count();
    float *out;
    CHECK(cudaMalloc(&out, sizeof(float) * blocks * threads));
    if (strcmp(mode, "read") == 0) {
        size_t n = (size_t)64 << 20;          // 256MiB
        float *a;
        CHECK(cudaMalloc(&a, n * sizeof(float)));
        CHECK(cudaMemset(a, 0, n * sizeof(float)));
        float ms = time_ms([&] { read_kernel<<<blocks, threads>>>(a, n, out); }, 3);
        printf("read  每个SM %2d个warp   %9.3f 毫秒   %7.1f GB/s\n", threads / 32, ms, n * 4.0 / ms / 1e6);
        CHECK(cudaFree(a));
    } else {
        int iters = 1 << 16;
        float ms = time_ms([&] { fma_kernel<<<blocks, threads>>>(iters, 0.999999f, 1e-6f, out); }, 3);
        printf("fma   每个SM %2d个warp   %9.3f 毫秒   %7.1f GFLOPS\n", threads / 32, ms,
               2.0 * iters * (double)blocks * threads / ms / 1e6);
    }
    CHECK(cudaFree(out));
    return 0;
}
