// 第3章：实测CPU的峰值算力。
//
// 做法：让每个线程反复执行大量互不依赖的FMA指令，不访问内存，只测计算单元本身有多快。
// 用法：./peak <scalar|avx2> <线程数>
//   scalar：标量FMA，一条指令做1次乘加
//   avx2：  256位FMA，一条指令做8次乘加
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ITERS 200000000L   // 每个线程的循环次数
#define CHAINS 12          // 互不依赖的累加链条数，要多于“FMA延迟 × FMA单元数”才能喂饱计算单元

static int use_avx2;
static double sink[64];    // 把结果写出去，防止编译器把整个循环优化掉

static void *worker(void *arg) {
    long id = (long)arg;
    if (use_avx2) {
        __m256 acc[CHAINS], x = _mm256_set1_ps(0.999999f), y = _mm256_set1_ps(1e-7f);
        for (int c = 0; c < CHAINS; c++) acc[c] = _mm256_set1_ps((float)c);
        for (long i = 0; i < ITERS; i++)
            for (int c = 0; c < CHAINS; c++) {
                acc[c] = _mm256_fmadd_ps(acc[c], x, y);
                __asm__ volatile("" : "+x"(acc[c]));   // 编译屏障：告诉编译器这个值“可能被改过”，阻止它把循环算掉
            }
        float out[8];
        __m256 s = acc[0];
        for (int c = 1; c < CHAINS; c++) s = _mm256_add_ps(s, acc[c]);
        _mm256_storeu_ps(out, s);
        sink[id] = out[0];
    } else {
        float acc[CHAINS], x = 0.999999f, y = 1e-7f;
        for (int c = 0; c < CHAINS; c++) acc[c] = (float)c;
        for (long i = 0; i < ITERS; i++)
            for (int c = 0; c < CHAINS; c++) {
                acc[c] = __builtin_fmaf(acc[c], x, y);
                __asm__ volatile("" : "+x"(acc[c]));
            }
        float s = 0;
        for (int c = 0; c < CHAINS; c++) s += acc[c];
        sink[id] = s;
    }
    return NULL;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法：%s <scalar|avx2> <线程数>\n", argv[0]); return 1; }
    use_avx2 = !strcmp(argv[1], "avx2");
    int nt = atoi(argv[2]);
    pthread_t tid[64];
    double t0 = now();
    for (long t = 0; t < nt; t++) pthread_create(&tid[t], NULL, worker, (void *)t);
    for (int t = 0; t < nt; t++) pthread_join(tid[t], NULL);
    double t = now() - t0;
    double flop = 2.0 * (use_avx2 ? 8 : 1) * CHAINS * ITERS * nt;
    printf("%-6s threads=%-2d  %7.3f 秒  %8.1f GFLOPS\n", argv[1], nt, t, flop / t / 1e9);
    return 0;
}
