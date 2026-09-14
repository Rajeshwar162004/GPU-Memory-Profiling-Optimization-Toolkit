#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void strided_access_contiguous(const float* __restrict__ a, float* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] * 2.0f;
    }
}

__global__ void strided_access_stride2(const float* __restrict__ a, float* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 < n) {
        b[idx * 2] = a[idx * 2] * 2.0f;
    }
}

__global__ void strided_access_stride4(const float* __restrict__ a, float* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 4 < n) {
        b[idx * 4] = a[idx * 4] * 2.0f;
    }
}

__global__ void strided_access_stride8(const float* __restrict__ a, float* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 8 < n) {
        b[idx * 8] = a[idx * 8] * 2.0f;
    }
}

__global__ void strided_access_stride16(const float* __restrict__ a, float* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 16 < n) {
        b[idx * 16] = a[idx * 16] * 2.0f;
    }
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void run_kernel(const char* name, void (*kernel)(const float*, float*, int), const float* d_a, float* d_b, int n, int block_size, int grid_size) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel<<<grid_size, block_size>>>(d_a, d_b, n);
    }
    cudaDeviceSynchronize();
    check_cuda_error("warmup");

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<grid_size, block_size>>>(d_a, d_b, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s: %.3f ms (avg over 100 runs)\n", name, ms / 100.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    int n = 1024 * 1024;  // 1M elements
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    size_t bytes = n * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
    }

    float *d_a, *d_b;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    printf("Strided Access Benchmark (n=%d)\n", n);
    printf("================================\n");

    run_kernel("Contiguous (stride 1)", strided_access_contiguous, d_a, d_b, n, block_size, grid_size);
    run_kernel("Stride 2", strided_access_stride2, d_a, d_b, n, block_size, grid_size);
    run_kernel("Stride 4", strided_access_stride4, d_a, d_b, n, block_size, grid_size);
    run_kernel("Stride 8", strided_access_stride8, d_a, d_b, n, block_size, grid_size);
    run_kernel("Stride 16", strided_access_stride16, d_a, d_b, n, block_size, grid_size);

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}