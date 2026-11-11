// 第7章：访存合并的微基准。
//
// 用法：./coalesce stride <步长>   第t个线程读a[t × 步长]
//       ./coalesce offset <偏移>   第t个线程读a[t + 偏移]
//
// 每次都只“用到”同样多的数据：4M个float，16MiB。步长越大，同一个warp的32个线程
// 读的地址越分散，显存要搬运的字节就越多。有效带宽 = 用到的字节数 ÷ 时间。
#include "common.cuh"
#include <cstring>

__global__ void read_strided(const float *a, int stride, int offset, size_t n, float *out) {
    size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    float s = 0;
#pragma unroll 1
    for (; t < n; t += step) s += a[t * stride + offset];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "stride";
    int v = argc > 2 ? atoi(argv[2]) : 1;
    int stride = strcmp(mode, "stride") == 0 ? v : 1;
    int offset = strcmp(mode, "offset") == 0 ? v : 0;
    size_t n = (size_t)4 << 20;                       // 用到的float个数
    size_t len = n * stride + offset;                 // 数组实际长度
    int threads = 256, blocks = sm_count() * 32;      // 每个SM 32个warp，occupancy 100%
    float *a, *out;
    CHECK(cudaMalloc(&a, len * sizeof(float)));
    CHECK(cudaMemset(a, 0, len * sizeof(float)));
    CHECK(cudaMalloc(&out, sizeof(float) * blocks * threads));
    float ms = time_ms([&] { read_strided<<<blocks, threads>>>(a, stride, offset, n, out); }, 20);
    printf("%-6s %3d   %8.3f 毫秒   有效带宽 %7.1f GB/s\n", mode, v, ms, n * 4.0 / ms / 1e6);
    CHECK(cudaFree(a));
    CHECK(cudaFree(out));
    return 0;
}
