#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32

__global__ void transpose_naive(const float* __restrict__ input, float* __restrict__ output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

__global__ void transpose_shared(const float* __restrict__ input, float* __restrict__ output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x_out < height && y_out < width) {
        output[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_padded(const float* __restrict__ input, float* __restrict__ output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();

    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x_out < height && y_out < width) {
        output[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void run_transpose_kernel(const char* name, void (*kernel)(const float*, float*, int, int), 
                          const float* d_input, float* d_output, int width, int height, 
                          dim3 block, dim3 grid) {
    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel<<<grid, block>>>(d_input, d_output, width, height);
    }
    cudaDeviceSynchronize();
    check_cuda_error("warmup");

    // Timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<grid, block>>>(d_input, d_output, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s: %.3f ms (avg over 100 runs)\n", name, ms / 100.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void verify_transpose(const float* h_input, const float* h_output, int width, int height) {
    bool correct = true;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (h_output[x * height + y] != h_input[y * width + x]) {
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
}

int main(int argc, char** argv) {
    int width = 1024;
    int height = 1024;
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);

    int size = width * height;
    size_t bytes = size * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        h_input[i] = (float)i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    printf("Matrix Transpose Benchmark (%dx%d)\n", width, height);
    printf("====================================\n");

    run_transpose_kernel("Naive", transpose_naive, d_input, d_output, width, height, block, grid);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    verify_transpose(h_input, h_output, width, height);

    run_transpose_kernel("Shared Memory", transpose_shared, d_input, d_output, width, height, block, grid);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    verify_transpose(h_input, h_output, width, height);

    run_transpose_kernel("Padded Shared Memory", transpose_padded, d_input, d_output, width, height, block, grid);
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    verify_transpose(h_input, h_output, width, height);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}