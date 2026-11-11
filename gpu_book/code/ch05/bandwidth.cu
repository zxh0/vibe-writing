// 第5章：实测T4的显存带宽。和第3章的bandwidth.c一样测两种操作：
// 只读（把数组的所有元素加起来）和复制（把一个数组复制到另一个）。
//
// 用法：./bandwidth read|copy
#include "common.cuh"
#include <cstring>

// 每个线程用float4一次读16个字节，按“网格跨步”的方式走遍整个数组
__global__ void read_kernel(const float4 *a, size_t n4, float *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float s = 0;
    for (; i < n4; i += stride) {
        float4 v = a[i];
        s += v.x + v.y + v.z + v.w;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;   // 每个线程只写一个数，可以忽略
}

__global__ void copy_kernel(const float4 *a, float4 *b, size_t n4) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) b[i] = a[i];
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "read";
    size_t n = (size_t)64 << 20;                     // 64M个float，每个数组256MiB，远大于4MB的L2
    size_t bytes = n * sizeof(float), n4 = n / 4;
    int threads = 256, blocks = sm_count() * 32;

    float *a, *b, *out;
    CHECK(cudaMalloc(&a, bytes));
    CHECK(cudaMalloc(&b, bytes));
    CHECK(cudaMalloc(&out, sizeof(float) * blocks * threads));
    CHECK(cudaMemset(a, 0, bytes));

    float ms;
    double moved;
    if (strcmp(mode, "read") == 0) {
        ms = time_ms([&] { read_kernel<<<blocks, threads>>>((const float4 *)a, n4, out); }, 20);
        moved = bytes;
    } else if (strcmp(mode, "copy") == 0) {
        ms = time_ms([&] { copy_kernel<<<blocks, threads>>>((const float4 *)a, (float4 *)b, n4); }, 20);
        moved = 2.0 * bytes;
    } else {
        fprintf(stderr, "不认识的模式：%s，应该是read或copy\n", mode);
        return 1;
    }

    // 理论带宽：显存的有效传输速率 × 位宽。CUDA报告的显存频率要乘以2才是每秒传输次数
    int mem_khz = 0, bus_bits = 0;
    CHECK(cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0));
    CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));
    double theory = 2.0 * mem_khz * 1e3 * bus_bits / 8 / 1e9;
    printf("%-5s  %.0fMiB   %8.3f 毫秒   %7.1f GB/s   （理论值%.0f GB/s）\n",
           mode, bytes / 1048576.0, ms, moved / ms / 1e6, theory);
    CHECK(cudaFree(a));
    CHECK(cudaFree(b));
    CHECK(cudaFree(out));
    return 0;
}
