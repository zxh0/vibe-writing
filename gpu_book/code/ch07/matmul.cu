// 第7章：朴素kernel与访存合并的kernel。
//
// 用法：./matmul <版本> <n> [block边长] [once]
//   版本：naive      第5章的朴素kernel，x方向对应行
//         coalesced  x方向对应列，同一个warp的线程访问连续的地址
//         check      在n×n的矩阵上，把两个kernel的结果和cuBLAS对比
//   最后加上once，只运行一次kernel、不计时，供Nsight Compute采集数据
#include "common.cuh"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <initializer_list>

// 第5章的朴素kernel：x对应行i，y对应列j
__global__ void matmul_naive(int n, const float *A, const float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}

// 只改了两行：x对应列j，y对应行i
__global__ void matmul_coalesced(int n, const float *A, const float *B, float *C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}

static void run(const char *ver, int n, int bs, const float *dA, const float *dB, float *dC) {
    dim3 block(bs, bs), grid((n + bs - 1) / bs, (n + bs - 1) / bs);
    if (strcmp(ver, "naive") == 0) matmul_naive<<<grid, block>>>(n, dA, dB, dC);
    else matmul_coalesced<<<grid, block>>>(n, dA, dB, dC);
}

static double rel_err(int n, const float *x, const float *ref) {
    double e = 0, m = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        e = fmax(e, fabs((double)x[i] - ref[i]));
        m = fmax(m, fabs((double)ref[i]));
    }
    return e / m;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法：%s naive|coalesced|check <n> [block边长] [once]\n", argv[0]);
        return 1;
    }
    const char *ver = argv[1];
    int n = atoi(argv[2]);
    int bs = argc > 3 ? atoi(argv[3]) : 32;
    bool once = argc > 4 && strcmp(argv[4], "once") == 0;
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

    if (strcmp(ver, "check") == 0) {
        cublasHandle_t h;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS初始化失败\n"); return 1; }
        const float alpha = 1, beta = 0;
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n);
        float *ref = (float *)malloc(bytes), *C = (float *)malloc(bytes);
        CHECK(cudaMemcpy(ref, dC, bytes, cudaMemcpyDeviceToHost));
        for (const char *v : {"naive", "coalesced"}) {
            run(v, n, bs, dA, dB, dC);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
            printf("check  n=%d  %-9s 与cuBLAS的相对误差 %.3e\n", n, v, rel_err(n, C, ref));
        }
        cublasDestroy(h);
        free(ref);
        free(C);
    } else if (once) {
        run(ver, n, bs, dA, dB, dC);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
    } else {
        int reps = n <= 1024 ? 20 : 5;
        float ms = time_ms([&] { run(ver, n, bs, dA, dB, dC); }, reps);
        printf("%-9s  n=%d  block=%dx%d   %10.3f 毫秒   %8.1f GFLOPS\n", ver, n, bs, bs, ms,
               2.0 * n * n * (double)n / ms / 1e6);
    }
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    free(A);
    free(B);
    return 0;
}
