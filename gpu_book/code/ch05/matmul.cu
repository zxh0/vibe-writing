// 第5章：GPU上的朴素矩阵乘法，以及cuBLAS。
//
// 用法：./matmul <版本> <n> [block边长]
//   版本：naive   朴素kernel，一个线程算C的一个元素，block默认32×32
//         cublas  调用cuBLAS的SGEMM
//         check   在n×n的矩阵上，把naive和cublas的结果与CPU的结果对比
//   例如：./matmul naive 4096
//         ./matmul naive 4096 16
//         ./matmul cublas 4096
//         ./matmul check 1000
//
// 矩阵都按行主序存放，C = A × B，A、B、C都是n×n的float矩阵。
#include "common.cuh"
#include <cmath>
#include <cstring>
#include <cublas_v2.h>

#define CHECK_CUBLAS(call)                                                         \
    do {                                                                           \
        cublasStatus_t st_ = (call);                                               \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
            fprintf(stderr, "cuBLAS错误 %s:%d：%d\n", __FILE__, __LINE__, (int)st_); \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

// 朴素kernel：线程(i, j)计算C的第i行第j列。
// i取自x方向的索引，j取自y方向的索引，这是最直接的写法。
__global__ void matmul_naive(int n, const float *A, const float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 行
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // 列
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)
            sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}

static void run_naive(int n, int bs, const float *dA, const float *dB, float *dC) {
    dim3 block(bs, bs);
    dim3 grid((n + bs - 1) / bs, (n + bs - 1) / bs);
    matmul_naive<<<grid, block>>>(n, dA, dB, dC);
}

// cuBLAS沿用列主序。行主序的C = A × B，按列主序看就是Cᵀ = Bᵀ × Aᵀ，
// 所以把B放在前面、A放在后面传进去，就能直接得到行主序的C，不需要转置。
static void run_cublas(cublasHandle_t h, int n, const float *dA, const float *dB, float *dC) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n));
}

// CPU参考结果：用double累加，按i、k、j的顺序，误差比float小得多
static void matmul_cpu(int n, const float *A, const float *B, float *C) {
    double *acc = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        memset(acc, 0, sizeof(double) * n);
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            for (int j = 0; j < n; j++) acc[j] += a * B[k * n + j];
        }
        for (int j = 0; j < n; j++) C[i * n + j] = (float)acc[j];
    }
    free(acc);
}

// 最大绝对误差，以及它相对于参考结果最大绝对值的比例
static void compare(const char *name, int n, const float *x, const float *ref) {
    double max_err = 0, max_ref = 0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        max_err = fmax(max_err, fabs((double)x[i] - ref[i]));
        max_ref = fmax(max_ref, fabs((double)ref[i]));
    }
    printf("check   n=%d  %-7s 最大绝对误差 %.3e   相对误差 %.3e\n", n, name, max_err, max_err / max_ref);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "用法：%s naive|cublas|check <n> [block边长]\n", argv[0]);
        return 1;
    }
    const char *mode = argv[1];
    int n = atoi(argv[2]);
    int bs = argc > 3 ? atoi(argv[3]) : 32;
    size_t bytes = sizeof(float) * (size_t)n * n;

    float *A = (float *)malloc(bytes), *B = (float *)malloc(bytes), *C = (float *)malloc(bytes);
    fill_random(A, (size_t)n * n, 1);
    fill_random(B, (size_t)n * n, 2);
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
    cublasHandle_t h;
    CHECK_CUBLAS(cublasCreate(&h));

    double flops = 2.0 * n * n * (double)n;
    int reps = n <= 1024 ? 20 : 5;
    if (strcmp(mode, "naive") == 0) {
        float ms = time_ms([&] { run_naive(n, bs, dA, dB, dC); }, reps);
        printf("naive   n=%d  block=%dx%d   %10.3f 毫秒   %8.1f GFLOPS\n", n, bs, bs, ms, flops / ms / 1e6);
    } else if (strcmp(mode, "cublas") == 0) {
        float ms = time_ms([&] { run_cublas(h, n, dA, dB, dC); }, reps);
        printf("cublas  n=%d                %10.3f 毫秒   %8.1f GFLOPS\n", n, ms, flops / ms / 1e6);
    } else if (strcmp(mode, "check") == 0) {
        float *ref = (float *)malloc(bytes);
        matmul_cpu(n, A, B, ref);
        run_naive(n, bs, dA, dB, dC);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
        compare("naive", n, C, ref);
        run_cublas(h, n, dA, dB, dC);
        CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));
        compare("cublas", n, C, ref);
        free(ref);
    } else {
        fprintf(stderr, "不认识的版本：%s\n", mode);
        return 1;
    }

    CHECK_CUBLAS(cublasDestroy(h));
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    free(A);
    free(B);
    free(C);
    return 0;
}
