// 第8章：共享内存分块的矩阵乘法。
//
// 用法：./matmul <版本> <n> [分块边长] [once | hold <秒数>]
//   版本：coalesced  第7章访存合并后的kernel，作为对照
//         smem       共享内存分块，分块边长取8、16或32（也就是block的边长），默认32
//         nosync1    smem去掉载入之后的那次__syncthreads()，演示同步缺失的后果
//         nosync2    smem去掉计算之后的那次__syncthreads()
//         cublas     调用cuBLAS的SGEMM
//         check      在n×n的矩阵上，把各个版本的结果和cuBLAS对比，n可以不是分块边长的整数倍
//   最后加上once，只运行一次kernel、不计时，供Nsight Compute采集数据；
//   加上hold 5，连续运行5秒，每秒输出一次GFLOPS，用来观察GPU发热、降频以后的速度
//
// 矩阵都按行主序存放，C = A × B，A、B、C都是n×n的float矩阵。
#include "common.cuh"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cublas_v2.h>

// 第7章的kernel：x对应列j，y对应行i，同一个warp的线程访问连续的地址
__global__ void matmul_coalesced(int n, const float *A, const float *B, float *C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}

// 共享内存分块：block是T×T个线程，负责C的一个T×T小块。
// 沿k方向每次把A的一个T×T小块和B的一个T×T小块搬进共享内存，每个线程搬一个元素；
// 然后每个线程用这两个小块算T次乘加。超出矩阵范围的位置补0，这样n不必是T的整数倍。
// SYNC1、SYNC2为false时去掉对应的__syncthreads()，只用来演示错误。
template <int T, bool SYNC1 = true, bool SYNC2 = true>
__global__ void matmul_smem(int n, const float *A, const float *B, float *C) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int tx = threadIdx.x, ty = threadIdx.y;
    int j = blockIdx.x * T + tx;  // 列
    int i = blockIdx.y * T + ty;  // 行
    float sum = 0.0f;
    for (int k0 = 0; k0 < n; k0 += T) {
        As[ty][tx] = (i < n && k0 + tx < n) ? A[i * n + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < n && j < n) ? B[(k0 + ty) * n + j] : 0.0f;
        if (SYNC1) __syncthreads();  // 等所有线程都搬完，小块才是完整的
        for (int k = 0; k < T; k++) sum += As[ty][k] * Bs[k][tx];
        if (SYNC2) __syncthreads();  // 等所有线程都算完，才能覆盖成下一个小块
    }
    if (i < n && j < n) C[i * n + j] = sum;
}

template <int T>
static void launch_smem(const char *ver, int n, const float *dA, const float *dB, float *dC) {
    dim3 block(T, T), grid((n + T - 1) / T, (n + T - 1) / T);
    if (strcmp(ver, "nosync1") == 0) matmul_smem<T, false, true><<<grid, block>>>(n, dA, dB, dC);
    else if (strcmp(ver, "nosync2") == 0) matmul_smem<T, true, false><<<grid, block>>>(n, dA, dB, dC);
    else matmul_smem<T><<<grid, block>>>(n, dA, dB, dC);
}

static cublasHandle_t handle;

static void run(const char *ver, int n, int bs, const float *dA, const float *dB, float *dC) {
    if (strcmp(ver, "cublas") == 0) {
        // cuBLAS按列主序理解矩阵，行主序的C = A × B相当于列主序的C^T = B^T × A^T（见第5章）
        const float alpha = 1, beta = 0;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n);
    } else if (strcmp(ver, "coalesced") == 0) {
        dim3 block(bs, bs), grid((n + bs - 1) / bs, (n + bs - 1) / bs);
        matmul_coalesced<<<grid, block>>>(n, dA, dB, dC);
    } else if (bs == 8) launch_smem<8>(ver, n, dA, dB, dC);
    else if (bs == 16) launch_smem<16>(ver, n, dA, dB, dC);
    else if (bs == 32) launch_smem<32>(ver, n, dA, dB, dC);
    else { fprintf(stderr, "分块边长只能是8、16或32\n"); exit(1); }
}

static double rel_err(int n, const float *x, const float *ref) {
    double e = 0, m = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        e = fmax(e, fabs((double)x[i] - ref[i]));
        m = fmax(m, fabs((double)ref[i]));
    }
    return e / m;
}

static double gflops(int n, double ms) { return 2.0 * n * n * (double)n / ms / 1e6; }

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法：%s coalesced|smem|nosync1|nosync2|cublas|check <n> [分块边长] [once | hold <秒数>]\n", argv[0]);
        return 1;
    }
    const char *ver = argv[1];
    int n = atoi(argv[2]);
    int bs = argc > 3 ? atoi(argv[3]) : 32;
    const char *mode = argc > 4 ? argv[4] : "";
    size_t bytes = sizeof(float) * (size_t)n * n;
    float *A = (float *)malloc(bytes), *B = (float *)malloc(bytes);
    fill_random(A, (size_t)n * n, 1);
    fill_random(B, (size_t)n * n, 2);
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
    // 只在用到cuBLAS时才初始化，免得Nsight Compute采集到cuBLAS自己的kernel
    bool use_cublas = strcmp(ver, "cublas") == 0 || strcmp(ver, "check") == 0;
    if (use_cublas && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS初始化失败\n"); return 1; }

    if (strcmp(ver, "check") == 0) {
        float *ref = (float *)malloc(bytes), *C = (float *)malloc(bytes);
        run("cublas", n, 32, dA, dB, dC);
        CHECK(cudaMemcpy(ref, dC, bytes, cudaMemcpyDeviceToHost));
        struct { const char *ver; int bs; } cases[] = {
            {"coalesced", 32}, {"smem", 8}, {"smem", 16}, {"smem", 32}, {"nosync1", 32}, {"nosync2", 32}};
        for (auto c : cases) {
            CHECK(cudaMemset(dC, 0, bytes));
            run(c.ver, n, c.bs, dA, dB, dC);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
            printf("check  n=%d  %-9s 分块%2d  与cuBLAS的相对误差 %.3e\n", n, c.ver, c.bs, rel_err(n, C, ref));
        }
        free(ref);
        free(C);
    } else if (strcmp(mode, "once") == 0) {
        run(ver, n, bs, dA, dB, dC);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
    } else if (strcmp(mode, "hold") == 0) {
        // 连续运行若干秒，每秒统计一次：GPU热起来以后，可能因为功耗或温度降低主频
        int secs = argc > 5 ? atoi(argv[5]) : 5;
        using clk = std::chrono::steady_clock;
        for (int s = 1; s <= secs; s++) {
            auto t0 = clk::now();
            int runs = 0;
            do {
                run(ver, n, bs, dA, dB, dC);
                CHECK(cudaDeviceSynchronize());
                runs++;
            } while (clk::now() - t0 < std::chrono::seconds(1));
            double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / runs;
            printf("hold  %-9s n=%d  第%d秒   %10.3f 毫秒   %8.1f GFLOPS\n", ver, n, s, ms, gflops(n, ms));
        }
    } else {
        int reps = n <= 1024 ? 20 : 5;
        float ms = time_ms([&] { run(ver, n, bs, dA, dB, dC); }, reps);
        if (strcmp(ver, "cublas") == 0)
            printf("%-9s  n=%d                %10.3f 毫秒   %8.1f GFLOPS\n", ver, n, ms, gflops(n, ms));
        else
            printf("%-9s  n=%d  block=%dx%d   %10.3f 毫秒   %8.1f GFLOPS\n", ver, n, bs, bs, ms, gflops(n, ms));
    }
    if (use_cublas) cublasDestroy(handle);
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    free(A);
    free(B);
    return 0;
}
