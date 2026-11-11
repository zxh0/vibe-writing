// 第7章各个程序共用的小工具（与第5章相同）：错误检查、随机数据、计时。
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 每个CUDA API调用都检查返回值，出错就打印位置并退出
#define CHECK(call)                                                                \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA错误 %s:%d：%s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err_));                                     \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

// 用一个简单的线性同余发生器填充[-1, 1)之间的随机数，种子相同，结果就相同
static void fill_random(float *x, size_t n, unsigned seed) {
    unsigned s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (s >> 8) * (2.0f / 16777216.0f) - 1.0f;
    }
}

// 用CUDA event给一段GPU工作计时：先预热一次，再连续跑reps次，返回平均每次的毫秒数。
// kernel的启动是异步的，所以不能用CPU的时钟直接计时，要等GPU把事件记录下来。
template <typename F>
static float time_ms(F work, int reps) {
    work();                                  // 预热：第一次运行要加载代码、初始化，比较慢
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; i++) work();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));      // 等GPU执行到stop这个事件
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / reps;
}

// 本机GPU的SM个数
static int sm_count() {
    int v = 0;
    CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0));
    return v;
}
