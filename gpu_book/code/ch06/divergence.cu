// 第6章：分支分歧的代价。
//
// 用法：./divergence none|warp|lane
//   none  所有线程走同一个分支
//   warp  按warp分支：偶数号warp走A，奇数号warp走B，同一个warp里的线程走向相同
//   lane  在warp内分支：偶数号线程走A，奇数号线程走B
//
// 两个分支做同样多的FMA，但写法不同，免得编译器把它们合并成一个。
#include "common.cuh"
#include <cstring>

__global__ void branch(int mode, int iters, float x, float y, float *out) {
    int t = threadIdx.x;
    bool a_side = mode == 0 ? true : mode == 1 ? (t / 32) % 2 == 0 : t % 2 == 0;
    float a = t;
    if (a_side) {
        for (int i = 0; i < iters; i++) a = fmaf(a, x, y);
    } else {
        for (int i = 0; i < iters / 2; i++) {
            a = fmaf(a, y, x);
            a = fmaf(a, x, y);
        }
    }
    out[blockIdx.x * blockDim.x + t] = a;
}

int main(int argc, char **argv) {
    const char *name = argc > 1 ? argv[1] : "none";
    int mode = strcmp(name, "warp") == 0 ? 1 : strcmp(name, "lane") == 0 ? 2 : 0;
    int threads = 256, blocks = sm_count() * 8, iters = 1 << 14;
    float *out;
    CHECK(cudaMalloc(&out, sizeof(float) * blocks * threads));
    float ms = time_ms([&] { branch<<<blocks, threads>>>(mode, iters, 0.999999f, 1e-6f, out); }, 10);
    printf("%-5s  %8.3f 毫秒   %8.1f GFLOPS（按每个线程做的FMA计）\n", name, ms,
           2.0 * iters * (double)blocks * threads / ms / 1e6);
    CHECK(cudaFree(out));
    return 0;
}
