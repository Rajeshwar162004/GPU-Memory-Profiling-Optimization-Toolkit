#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void transpose_naive(const float* __restrict__ input, float* __restrict__ output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

int main(int argc, char** argv) {
    int width = 1024, height = 1024;
    if (argc > 1) width = atoi(argv[1]);
    if (argc > 2) height = atoi(argv[2]);

    size_t bytes = width * height * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    for (int i = 0; i < width * height; i++) h_input[i] = (float)i;

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);

    for (int i = 0; i < 50; i++) {
        transpose_naive<<<grid, block>>>(d_input, d_output, width, height);
    }
    cudaDeviceSynchronize();

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
