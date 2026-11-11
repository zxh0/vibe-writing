// 第5章：实测T4的FP32峰值算力。思路和第3章的peak.c一样：
// 让每个线程反复执行FMA，不访问显存，看每秒能完成多少次浮点运算。
//
// 用法：./peak
#include "common.cuh"

#define CHAINS 8      // 每个线程同时维护8个互不相关的累加器，让FMA可以流水执行
#define ITERS 8192    // 每个累加器做多少次FMA

__global__ void fma_loop(float *out, float x, float y) {
    float acc[CHAINS];
#pragma unroll
    for (int c = 0; c < CHAINS; c++) acc[c] = threadIdx.x + c;
    for (int i = 0; i < ITERS; i++) {
#pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = fmaf(acc[c], x, y);
    }
    float s = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; c++) s += acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;   // 把结果写出去，免得编译器把循环优化掉
}

int main() {
    int sms = sm_count(), clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    int threads = 256, blocks = sms * 16;
    float *out;
    CHECK(cudaMalloc(&out, sizeof(float) * blocks * threads));
    // x和y由参数传进来，编译器在编译时不知道它们的值，没法提前算出结果
    float ms = time_ms([&] { fma_loop<<<blocks, threads>>>(out, 0.999999f, 1e-6f); }, 10);
    double flops = 2.0 * CHAINS * ITERS * (double)blocks * threads;

    // 每个SM有64个FP32单元，这是Turing的数字；其他架构不同，这里只作参考
    double theory = sms * 64 * 2.0 * clock_khz / 1e6;
    printf("# GPU：%s，%d个SM，最高主频%.0fMHz，按每SM 64个FP32单元算，理论峰值%.0f GFLOPS\n",
           prop.name, sms, clock_khz / 1e3, theory);
    printf("fp32 FMA  blocks=%d  threads=%d   %8.3f 毫秒   %8.1f GFLOPS\n",
           blocks, threads, ms, flops / ms / 1e6);
    CHECK(cudaFree(out));
    return 0;
}
