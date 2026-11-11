// 第6章：block是怎样分配到SM上的。
//
// 用法：./blocks <block数> <每个block的线程数>
//
// 每个block做同样多的计算，并由它的0号线程记下：自己在哪个SM上执行（%smid），
// 什么时候开始、什么时候结束（%globaltimer，单位纳秒）。程序据此统计每个SM分到了
// 几个block，以及block分成了几轮执行。
#include "common.cuh"
#include <algorithm>
#include <map>
#include <vector>

__device__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void busy(int iters, float *out, unsigned long long *start, unsigned long long *end, unsigned *smid) {
    unsigned long long t0 = now_ns();
    float a = threadIdx.x;
    for (int i = 0; i < iters; i++) a = fmaf(a, 0.999999f, 1e-6f);
    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
    __syncthreads();                       // 等这个block的所有线程都算完
    if (threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        start[blockIdx.x] = t0;
        end[blockIdx.x] = now_ns();
        smid[blockIdx.x] = s;
    }
}

int main(int argc, char **argv) {
    int blocks = argc > 1 ? atoi(argv[1]) : 64, threads = argc > 2 ? atoi(argv[2]) : 1024;
    int iters = 1 << 16;
    float *out;
    unsigned long long *ds, *de;
    unsigned *dm;
    CHECK(cudaMalloc(&out, sizeof(float) * blocks * threads));
    CHECK(cudaMalloc(&ds, sizeof(unsigned long long) * blocks));
    CHECK(cudaMalloc(&de, sizeof(unsigned long long) * blocks));
    CHECK(cudaMalloc(&dm, sizeof(unsigned) * blocks));
    busy<<<blocks, threads>>>(iters, out, ds, de, dm);     // 预热
    CHECK(cudaDeviceSynchronize());
    busy<<<blocks, threads>>>(iters, out, ds, de, dm);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<unsigned long long> s(blocks), e(blocks);
    std::vector<unsigned> m(blocks);
    CHECK(cudaMemcpy(s.data(), ds, sizeof(unsigned long long) * blocks, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(e.data(), de, sizeof(unsigned long long) * blocks, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m.data(), dm, sizeof(unsigned) * blocks, cudaMemcpyDeviceToHost));

    unsigned long long t0 = *std::min_element(s.begin(), s.end());
    unsigned long long t1 = *std::max_element(e.begin(), e.end());
    double dur = 0;                                         // 一个block平均执行多久
    for (int b = 0; b < blocks; b++) dur += e[b] - s[b];
    dur /= blocks;

    // %smid不一定从0开始连续编号（芯片上有被屏蔽的SM），所以用map统计
    std::map<unsigned, int> per;
    for (int b = 0; b < blocks; b++) per[m[b]]++;
    int used = (int)per.size(), lo = blocks, hi = 0;
    for (auto &kv : per) { lo = std::min(lo, kv.second); hi = std::max(hi, kv.second); }

    printf("blocks=%d  threads=%d   用到%d个SM，每个SM分到%d～%d个block   一个block约%.1f微秒，全部用时%.1f微秒\n",
           blocks, threads, used, lo, hi, dur / 1e3, (t1 - t0) / 1e3);
    // 按开始时间分轮：开始时间比上一轮的第一个block晚半个block时长以上，就算新的一轮
    std::vector<unsigned long long> st = s;
    std::sort(st.begin(), st.end());
    int round = 0, cnt = 0;
    unsigned long long first = st[0];
    for (int b = 0; b <= blocks; b++) {
        if (b == blocks || st[b] - first > dur / 2) {
            printf("    第%d轮：%3d个block，在第%7.1f微秒开始\n", ++round, cnt, (first - t0) / 1e3);
            if (b == blocks) break;
            first = st[b];
            cnt = 0;
        }
        cnt++;
    }
    CHECK(cudaFree(out));
    CHECK(cudaFree(ds));
    CHECK(cudaFree(de));
    CHECK(cudaFree(dm));
    return 0;
}
