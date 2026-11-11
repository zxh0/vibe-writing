// 第6章：用“指针追逐”测各级存储的访问延迟。
//
// 用法：./latency global   依次测从4KiB到64MiB的数组
//       ./latency shared
//
// 只启动一个线程，让它执行 j = next[j]：下一次读哪个地址，要等这一次读回来才知道，
// 所以每次访问都要完整地等一个延迟，没法重叠。数组小时数据都在L1里，大一些在L2里，
// 再大就只能去显存里读。每两个相邻元素隔128字节，也就是一个缓存行。
#include "common.cuh"
#include <cstring>

#define STRIDE 32   // 32个unsigned = 128字节

__global__ void chase_global(const unsigned *next, int warm, int steps, unsigned *out, long long *cycles) {
    unsigned j = 0;
    for (int i = 0; i < warm; i++) j = next[j];    // 先走一遍，把能缓存的数据都缓存起来
    long long t0 = clock64();
    for (int i = 0; i < steps; i++) j = next[j];
    long long t1 = clock64();
    out[0] = j;
    cycles[0] = t1 - t0;
}

__global__ void chase_shared(int count, int steps, unsigned *out, long long *cycles) {
    __shared__ unsigned next[4096];
    for (int i = 0; i < count; i++) next[i] = (i + 1) % count;
    unsigned j = 0;
    for (int i = 0; i < count; i++) j = next[j];
    long long t0 = clock64();
    for (int i = 0; i < steps; i++) j = next[j];
    long long t1 = clock64();
    out[0] = j;
    cycles[0] = t1 - t0;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "global";
    int clock_khz = 0;
    CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    unsigned *out;
    long long *dcyc, cyc;
    CHECK(cudaMalloc(&out, sizeof(unsigned)));
    CHECK(cudaMalloc(&dcyc, sizeof(long long)));

    if (strcmp(mode, "shared") == 0) {
        int steps = 1 << 16;
        chase_shared<<<1, 1>>>(1024, steps, out, dcyc);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(&cyc, dcyc, sizeof(cyc), cudaMemcpyDeviceToHost));
        double c = (double)cyc / steps;
        printf("shared   %8s   %6.1f 周期   %6.1f 纳秒\n", "4KiB", c, c / (clock_khz / 1e6));
    } else {
        for (size_t kib = 4; kib <= 64 * 1024; kib *= 2) {
            size_t n = kib * 1024 / sizeof(unsigned), count = n / STRIDE;
            unsigned *h = (unsigned *)calloc(n, sizeof(unsigned)), *d;
            for (size_t k = 0; k < count; k++) h[k * STRIDE] = ((k + 1) % count) * STRIDE;
            CHECK(cudaMalloc(&d, n * sizeof(unsigned)));
            CHECK(cudaMemcpy(d, h, n * sizeof(unsigned), cudaMemcpyHostToDevice));
            int steps = (int)(count > 8192 ? count : 8192);
            chase_global<<<1, 1>>>(d, (int)count, steps, out, dcyc);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(&cyc, dcyc, sizeof(cyc), cudaMemcpyDeviceToHost));
            double c = (double)cyc / steps;
            char size[32];
            if (kib < 1024) snprintf(size, sizeof size, "%zuKiB", kib);
            else snprintf(size, sizeof size, "%zuMiB", kib / 1024);
            printf("global   %8s   %6.1f 周期   %6.1f 纳秒\n", size, c, c / (clock_khz / 1e6));
            CHECK(cudaFree(d));
            free(h);
        }
    }
    CHECK(cudaFree(out));
    CHECK(cudaFree(dcyc));
    return 0;
}
