// L2 capacity probe via pointer-chase latency sweep.
// Working set = n * sizeof(int) bytes. Single-thread dependent loads => latency bound.
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>

__global__ void chase_kernel(const int* __restrict__ next, int n, long long steps,
                             unsigned long long* __restrict__ out) {
    int idx = 0;
    long long acc = 0;
    unsigned long long t0 = clock64();
    for (long long i = 0; i < steps; ++i) {
        idx = __ldg(&next[idx]);
        acc += idx;
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = t1 - t0;
        out[1] = (unsigned long long)acc;
        out[2] = (unsigned long long)idx;
    }
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s  SMs=%d  L2(API)=%d bytes (%.2f MiB)\n",
           p.name, p.multiProcessorCount,
           p.l2CacheSize, p.l2CacheSize / 1048576.0);
    fflush(stdout);

    // working-set sizes in KiB
    int kib[] = {64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 2304, 2560,
                 2816, 3072, 3584, 4096, 5120, 6144, 8192, 12288, 16384, 24576, 32768};
    int nk = sizeof(kib) / sizeof(kib[0]);

    const long long steps = 4LL * 1024 * 1024;  // 4M dependent loads
    unsigned long long* d_out;
    cudaMalloc(&d_out, 3 * sizeof(unsigned long long));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    printf("%10s %12s %12s %12s\n", "workingset", "cycles/acc", "ns/acc", "GB/s(eff)");
    for (int k = 0; k < nk; ++k) {
        int bytes = kib[k] * 1024;
        int n = bytes / (int)sizeof(int);
        n = (n / 4) * 4;  // multiples of 4 to align permutation groups
        if (n < 8) continue;

        std::vector<int> h(n);
        for (int i = 0; i < n; ++i) h[i] = i;
        std::mt19937 rng(12345 + k);
        std::shuffle(h.begin(), h.end(), rng);
        // cyclic permutation: h[i] points to next element in the shuffled order
        std::vector<int> perm(n);
        for (int i = 0; i < n; ++i) perm[h[i]] = h[(i + 1) % n];

        int* d_next;
        cudaMalloc(&d_next, n * sizeof(int));
        cudaMemcpy(d_next, perm.data(), n * sizeof(int), cudaMemcpyHostToDevice);

        // warmup
        chase_kernel<<<1, 1>>>(d_next, n, steps / 8, d_out);
        cudaDeviceSynchronize();
        unsigned long long h_out[3];
        cudaEventRecord(ev0);
        chase_kernel<<<1, 1>>>(d_next, n, steps, d_out);
        cudaEventRecord(ev1);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("err %s\n", cudaGetErrorString(e)); }
        cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

        float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
        double cyc = (double)h_out[0] / steps;
        double ns = (double)ms * 1e6 / (double)steps;
        // effective bytes moved = 4 bytes per dependent access
        double gbs = ns > 0 ? (4.0 / ns) : 0;  // GB/s
        printf("%9d K %12.1f %12.2f %12.2f\n", kib[k], cyc, ns, gbs);
        fflush(stdout);
        cudaFree(d_next);
    }
    cudaFree(d_out);
    return 0;
}
