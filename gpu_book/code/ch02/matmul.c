// 第2章：在CPU上用C实现矩阵乘法的几个版本，并测量GFLOPS。
//
// 用法：./matmul <版本> <n> [块大小] [线程数]
//   版本：ijk、ikj、blocked、threads
//   例如：./matmul ikj 1024
//         ./matmul blocked 1024 64
//         ./matmul threads 1024 64 6
//       ./matmul check 256      验证各版本的结果与ijk一致
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 朴素版本：i、j、k的顺序，最内层沿B的一列往下走
void matmul_ijk(int n, const float *A, const float *B, float *C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

// 调换循环顺序：i、k、j，最内层沿B的一行和C的一行往右走
void matmul_ikj(int n, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

// 分块版本：计算C的第[i0, i1)行，块内仍按i、k、j的顺序
static void blocked_rows(int n, int bs, const float *A, const float *B, float *C, int i0, int i1) {
    for (int ii = i0; ii < i1; ii += bs)
        for (int kk = 0; kk < n; kk += bs)
            for (int jj = 0; jj < n; jj += bs) {
                int imax = ii + bs < i1 ? ii + bs : i1;
                int kmax = kk + bs < n ? kk + bs : n;
                int jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        float a = A[i * n + k];
                        for (int j = jj; j < jmax; j++)
                            C[i * n + j] += a * B[k * n + j];
                    }
            }
}

void matmul_blocked(int n, int bs, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    blocked_rows(n, bs, A, B, C, 0, n);
}

// 多线程版本：把C的行平均分给各个线程，每个线程各自做分块计算
typedef struct { int n, bs, i0, i1; const float *A, *B; float *C; } Task;

static void *worker(void *arg) {
    Task *t = arg;
    blocked_rows(t->n, t->bs, t->A, t->B, t->C, t->i0, t->i1);
    return NULL;
}

void matmul_threads(int n, int bs, int nt, const float *A, const float *B, float *C) {
    memset(C, 0, sizeof(float) * n * n);
    pthread_t tid[64];
    Task task[64];
    for (int t = 0; t < nt; t++) {
        task[t] = (Task){n, bs, n * t / nt, n * (t + 1) / nt, A, B, C};
        pthread_create(&tid[t], NULL, worker, &task[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
}

// ---------- 工具函数 ----------
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// 标准正态分布的随机数（Box-Muller），固定种子，保证每次运行数据相同
static void fill_randn(float *x, int len, unsigned seed) {
    srand(seed);
    for (int i = 0; i < len; i += 2) {
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0), u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double r = sqrt(-2.0 * log(u1));
        x[i] = (float)(r * cos(2 * M_PI * u2));
        if (i + 1 < len) x[i + 1] = (float)(r * sin(2 * M_PI * u2));
    }
}

static void run(const char *v, int n, int bs, int nt, const float *A, const float *B, float *C) {
    if (!strcmp(v, "ijk")) matmul_ijk(n, A, B, C);
    else if (!strcmp(v, "ikj")) matmul_ikj(n, A, B, C);
    else if (!strcmp(v, "blocked")) matmul_blocked(n, bs, A, B, C);
    else if (!strcmp(v, "threads")) matmul_threads(n, bs, nt, A, B, C);
    else { fprintf(stderr, "未知版本：%s\n", v); exit(1); }
}

static float max_rel_err(int len, const float *x, const float *ref) {
    float e = 0, m = 0;
    for (int i = 0; i < len; i++) {
        float d = fabsf(x[i] - ref[i]);
        if (d > e) e = d;
        if (fabsf(ref[i]) > m) m = fabsf(ref[i]);
    }
    return e / m;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法：%s <版本> <n> [块大小] [线程数]\n", argv[0]); return 1; }
    const char *v = argv[1];
    int n = atoi(argv[2]);
    int bs = argc > 3 ? atoi(argv[3]) : 64;
    int nt = argc > 4 ? atoi(argv[4]) : 1;
    float *A = malloc(sizeof(float) * n * n), *B = malloc(sizeof(float) * n * n);
    float *C = malloc(sizeof(float) * n * n), *R = malloc(sizeof(float) * n * n);
    fill_randn(A, n * n, 1);
    fill_randn(B, n * n, 2);

    if (!strcmp(v, "check")) {
        matmul_ijk(n, A, B, R);
        const char *vs[] = {"ikj", "blocked", "threads"};
        for (int i = 0; i < 3; i++) {
            run(vs[i], n, bs, 4, A, B, C);
            printf("%-8s 与ijk的最大相对误差：%.2e\n", vs[i], max_rel_err(n * n, C, R));
        }
        return 0;
    }

    // 预热一次，再取三次中最快的一次
    int reps = (double)n * n * n > 3e9 && !strcmp(v, "ijk") ? 1 : 3;
    double best = 1e30;
    run(v, n, bs, nt, A, B, C);
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        run(v, n, bs, nt, A, B, C);
        double t = now() - t0;
        if (t < best) best = t;
    }
    double gflops = 2.0 * n * n * n / best / 1e9;
    printf("%-8s n=%-5d bs=%-4d threads=%-2d  %9.3f 秒  %8.2f GFLOPS\n", v, n, bs, nt, best, gflops);
    return 0;
}
