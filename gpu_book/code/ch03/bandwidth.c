// 第3章：实测内存带宽，并顺便测量向量加法的速度。
//
// 三个数组各有64M个float（各256MiB），远大于L3 cache，所以每次都要从内存读写。
// 用法：./bandwidth <read|copy|add> <线程数>
//   read：只读，把数组a的所有元素加起来
//   copy：b[i] = a[i]
//   add： c[i] = a[i] + b[i]，即向量加法，每个元素1次浮点运算
// 带宽按程序实际读写的字节数计算（和STREAM的约定一致）。
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N (64L * 1024 * 1024)

static float *a, *b, *c;
static int mode, nt;
static double partial[64];

static void *worker(void *arg) {
    long t = (long)arg, i0 = N * t / nt, i1 = N * (t + 1) / nt;
    if (mode == 0) {
        __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
        for (long i = i0; i < i1; i += 32) {
            s0 = _mm256_add_ps(s0, _mm256_loadu_ps(a + i));
            s1 = _mm256_add_ps(s1, _mm256_loadu_ps(a + i + 8));
            s2 = _mm256_add_ps(s2, _mm256_loadu_ps(a + i + 16));
            s3 = _mm256_add_ps(s3, _mm256_loadu_ps(a + i + 24));
            __asm__ volatile("" : "+x"(s0), "+x"(s1), "+x"(s2), "+x"(s3));   // 编译屏障，防止循环被优化掉
        }
        float out[8];
        _mm256_storeu_ps(out, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
        partial[t] = out[0];
    } else if (mode == 1) {
        for (long i = i0; i < i1; i++) b[i] = a[i];
    } else {
        for (long i = i0; i < i1; i++) c[i] = a[i] + b[i];
    }
    return NULL;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double run(void) {
    pthread_t tid[64];
    double t0 = now();
    for (long t = 0; t < nt; t++) pthread_create(&tid[t], NULL, worker, (void *)t);
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
    return now() - t0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法：%s <read|copy|add> <线程数>\n", argv[0]); return 1; }
    mode = !strcmp(argv[1], "read") ? 0 : !strcmp(argv[1], "copy") ? 1 : 2;
    nt = atoi(argv[2]);
    a = aligned_alloc(64, N * 4); b = aligned_alloc(64, N * 4); c = aligned_alloc(64, N * 4);
    for (long i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.0f; }   // 先写一遍，让操作系统真正分配内存

    double best = 1e30;
    for (int r = 0; r < 5; r++) { double t = run(); if (t < best) best = t; }
    double bytes = (mode == 0 ? 1 : mode == 1 ? 2 : 3) * N * 4.0;
    printf("%-5s threads=%-2d  %7.4f 秒  %6.1f GB/s", argv[1], nt, best, bytes / best / 1e9);
    if (mode == 2) printf("  %6.2f GFLOPS", N / best / 1e9);
    printf("\n");
    return 0;
}
