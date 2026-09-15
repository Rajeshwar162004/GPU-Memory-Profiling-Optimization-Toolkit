#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32

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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    for (int i = 0; i < 50; i++) {
        transpose_padded<<<grid, block>>>(d_input, d_output, width, height);
    }
    cudaDeviceSynchronize();

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
