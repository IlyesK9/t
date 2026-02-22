#include "gpu_reductions.h"
#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>

__global__ void max_reduction_kernel(const double* input, double* output, std::uint32_t n){
    extern __shared__ double shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (i < n) ? input[i] : -1e100;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_data[tid] < shared_data[tid + s]) {
                shared_data[tid] = shared_data[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

double get_max_value(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width){
    std::uint32_t n = source_image_height * source_image_width;
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    double* d_partial_max;
    cudaMalloc(&d_partial_max, num_blocks * sizeof(double));

    max_reduction_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(double)>>>((double*)(*d_source_image), d_partial_max, n);

    while (num_blocks > 1) {
        int prev_blocks = num_blocks;
        num_blocks = (num_blocks + threads_per_block - 1) / threads_per_block;

        max_reduction_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(double)>>>(d_partial_max, d_partial_max, prev_blocks);
    }

    double result;
    cudaMemcpy(&result, d_partial_max, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_partial_max);

    return result;
}