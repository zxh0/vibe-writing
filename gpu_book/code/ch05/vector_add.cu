// 第5章：第一个CUDA程序，向量加法 c = a + b。
//
// 用法：./vector_add [n] [每个block的线程数]
//   默认 n = 2^26（每个向量256MiB），每个block 256个线程
//
// 程序依次做这几件事：在主机内存里准备a和b，在显存里分配三个向量，
// 把a和b复制到显存，启动kernel，再把c复制回来检查。每一步都计时。
#include "common.cuh"

// kernel：每个线程算c的一个元素
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 这个线程负责第几个元素
    if (i < n) c[i] = a[i] + b[i];                   // 最后一个block可能有多余的线程
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1 << 26;
    int threads = argc > 2 ? atoi(argv[2]) : 256;
    size_t bytes = sizeof(float) * (size_t)n;

    // 主机内存
    float *a = (float *)malloc(bytes), *b = (float *)malloc(bytes), *c = (float *)malloc(bytes);
    fill_random(a, n, 1);
    fill_random(b, n, 2);

    // 显存
    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));

    // 主机 → 设备
    float h2d = time_ms([&] { CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice)); }, 5);
    CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

    // 启动kernel：block数要向上取整，保证每个元素都有线程负责
    int blocks = (n + threads - 1) / threads;
    float ms = time_ms([&] { vector_add<<<blocks, threads>>>(da, db, dc, n); }, 20);

    // 设备 → 主机
    float d2h = time_ms([&] { CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost)); }, 5);

    // 检查：浮点加法的结果是确定的，GPU和CPU应该逐位相同
    long wrong = 0;
    for (int i = 0; i < n; i++)
        if (c[i] != a[i] + b[i]) wrong++;

    double mib = bytes / 1048576.0;
    printf("主机→设备  %.0fMiB   %8.3f 毫秒   %7.1f GB/s\n", mib, h2d, bytes / h2d / 1e6);
    printf("设备→主机  %.0fMiB   %8.3f 毫秒   %7.1f GB/s\n", mib, d2h, bytes / d2h / 1e6);
    // 向量加法读两个向量、写一个向量，一共搬运3×bytes
    printf("vector_add  n=%d  block=%d  grid=%d   %8.3f 毫秒   %7.1f GB/s\n",
           n, threads, blocks, ms, 3.0 * bytes / ms / 1e6);
    printf("检查：%s（%ld个元素不一致）\n", wrong ? "有错" : "全部正确", wrong);

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
    free(a);
    free(b);
    free(c);
    return wrong != 0;
}
