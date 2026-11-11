// 第6章：occupancy，即每个SM上实际驻留的warp数占上限的比例。
//
// 用法：./occupancy limits   打印本机SM的各项上限
//       ./occupancy naive    第5章朴素kernel在三种block大小下的寄存器用量和occupancy
//       ./occupancy smem     每个block额外占用一些共享内存，看occupancy和带宽怎样变化
#include "common.cuh"
#include <cstring>
#include <initializer_list>

static int attr(cudaDeviceAttr a) {
    int v = 0;
    CHECK(cudaDeviceGetAttribute(&v, a, 0));
    return v;
}

// 与第5章完全相同的朴素kernel
__global__ void matmul_naive(int n, const float *A, const float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) sum += A[i * n + k] * B[k * n + j];
        C[i * n + j] = sum;
    }
}

// 和warps.cu一样的只读kernel，只是每个block多申请了一块动态共享内存
__global__ void read_smem(const float *a, size_t n, float *out) {
    extern __shared__ float pad[];
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float s = 0;
#pragma unroll 1
    for (; i < n; i += stride) s += a[i];
    if (threadIdx.x == 0) pad[0] = s;      // 用一下共享内存
    __syncthreads();
    out[blockIdx.x * blockDim.x + threadIdx.x] = s + pad[0];
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "limits";
    int max_threads = attr(cudaDevAttrMaxThreadsPerMultiProcessor);
    int max_warps = max_threads / 32;
    if (strcmp(mode, "limits") == 0) {
        printf("每个SM最多驻留的线程     %d（%d个warp）\n", max_threads, max_warps);
        printf("每个SM最多驻留的block    %d\n", attr(cudaDevAttrMaxBlocksPerMultiprocessor));
        printf("每个SM的32位寄存器       %d（%dKiB）\n", attr(cudaDevAttrMaxRegistersPerMultiprocessor),
               attr(cudaDevAttrMaxRegistersPerMultiprocessor) * 4 / 1024);
        printf("每个SM的共享内存         %dKiB\n", attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor) / 1024);
        printf("每个block最多的共享内存  %dKiB\n", attr(cudaDevAttrMaxSharedMemoryPerBlockOptin) / 1024);
        printf("每个线程最多的寄存器     255\n");
        printf("L2 cache                 %dKiB\n", attr(cudaDevAttrL2CacheSize) / 1024);
        printf("warp大小                 %d\n", attr(cudaDevAttrWarpSize));
    } else if (strcmp(mode, "naive") == 0) {
        cudaFuncAttributes fa;
        CHECK(cudaFuncGetAttributes(&fa, matmul_naive));
        for (int bs : {8, 16, 32}) {
            int nb = 0;
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, matmul_naive, bs * bs, 0));
            int warps = nb * bs * bs / 32;
            printf("naive  block=%2dx%-2d  每个线程%d个寄存器   每个SM驻留%2d个block、%2d个warp   occupancy %3.0f%%\n",
                   bs, bs, fa.numRegs, nb, warps, 100.0 * warps / max_warps);
        }
    } else if (strcmp(mode, "smem") == 0) {
        int threads = 256, sms = sm_count();
        size_t n = (size_t)64 << 20;
        float *a, *out;
        CHECK(cudaMalloc(&a, n * sizeof(float)));
        CHECK(cudaMemset(a, 0, n * sizeof(float)));
        CHECK(cudaMalloc(&out, sizeof(float) * sms * 64 * threads));
        // 允许一个block用满SM的共享内存，并让SM把片上存储尽量多地划给共享内存
        int max_smem = attr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
        CHECK(cudaFuncSetAttribute(read_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
        CHECK(cudaFuncSetAttribute(read_smem, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
        for (int kib : {0, 8, 16, 24, 32, 48, 64}) {
            size_t smem = kib ? (size_t)kib * 1024 : 16;   // kernel里要写pad[0]，至少留一点
            if ((int)smem > max_smem) continue;
            int nb = 0;
            CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, read_smem, threads, smem));
            if (nb == 0) continue;
            int blocks = sms * nb;              // 正好每个SM驻留nb个block
            float ms = time_ms([&] { read_smem<<<blocks, threads, smem>>>(a, n, out); }, 5);
            int warps = nb * threads / 32;
            printf("smem  每个block %2dKiB共享内存   每个SM驻留%d个block、%2d个warp   occupancy %3.0f%%   %7.1f GB/s\n",
                   kib, nb, warps, 100.0 * warps / max_warps, n * 4.0 / ms / 1e6);
        }
        CHECK(cudaFree(a));
        CHECK(cudaFree(out));
    }
    return 0;
}
