#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void reduction_naive(const float* __restrict__ input, float* __restrict__ output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load data
    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_optimized(const float* __restrict__ input, float* __restrict__ output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load data with coalesced access
    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val += input[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

    // Unrolled reduction
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    // Warp-level reduction
    if (tid < 32) {
        volatile float* vs = sdata;
        vs[tid] += vs[tid + 32];
        vs[tid] += vs[tid + 16];
        vs[tid] += vs[tid + 8];
        vs[tid] += vs[tid + 4];
        vs[tid] += vs[tid + 2];
        vs[tid] += vs[tid + 1];
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_final(float* partial_sums, float* result, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += partial_sums[i];
    }
    *result = sum;
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void run_reduction_kernel(const char* name, void (*kernel)(const float*, float*, int),
                          const float* d_input, float* d_partial, float* d_result, int n, int block_size) {
    int grid_size = (n + block_size * 2 - 1) / (block_size * 2);
    size_t shared_mem = block_size * sizeof(float);

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel<<<grid_size, block_size, shared_mem>>>(d_input, d_partial, n);
        reduction_final<<<1, 1>>>(d_partial, d_result, grid_size);
    }
    cudaDeviceSynchronize();
    check_cuda_error("warmup");

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<grid_size, block_size, shared_mem>>>(d_input, d_partial, n);
        reduction_final<<<1, 1>>>(d_partial, d_result, grid_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s: %.3f ms (avg over 100 runs)\n", name, ms / 100.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void verify_reduction(const float* h_input, float h_result, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += h_input[i];
    }
    bool correct = fabsf(sum - h_result) < 1e-3;
    printf("Verification: %s (expected %.3f, got %.3f)\n", correct ? "PASSED" : "FAILED", sum, h_result);
}

int main(int argc, char** argv) {
    int n = 1024 * 1024;  // 1M elements
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    size_t bytes = n * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // Simple test: sum should be n
    }

    float *d_input, *d_partial, *d_result;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_partial, ((n + 1023) / 1024) * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;

    printf("Reduction Benchmark (n=%d)\n", n);
    printf("==========================\n");

    run_reduction_kernel("Naive", reduction_naive, d_input, d_partial, d_result, n, block_size);
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    verify_reduction(h_input, h_result, n);

    run_reduction_kernel("Optimized", reduction_optimized, d_input, d_partial, d_result, n, block_size);
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    verify_reduction(h_input, h_result, n);

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_partial);
    cudaFree(d_result);

    return 0;
}