#include "gpu_NN_upscaling.h"
#include "gpu_memory_management.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

std::uint32_t get_NN_upscaled_width(std::uint32_t image_width){
    return image_width * 3;
}

std::uint32_t get_NN_upscaled_height(std::uint32_t image_height){
    return image_height * 3;
}

__global__ void nn_upscale_kernel(const double* source, std::uint32_t source_height, std::uint32_t source_width, double* result, std::uint32_t result_height, std::uint32_t result_width){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= result_width || y >= result_height) {
        return;
    }

    const std::uint32_t source_x = x / 3;
    const std::uint32_t source_y = y / 3;
    result[y * result_width + x] = source[source_y * source_width + source_x];
}


void NN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::uint32_t result_height = get_NN_upscaled_height(source_image_height);
    std::uint32_t result_width = get_NN_upscaled_width(source_image_width);
    size_t result_size = result_height * result_width;

    cudaMalloc(d_result, result_size * sizeof(double));

    dim3 block_dim(16, 16);
    dim3 grid_dim((result_width + block_dim.x - 1) / block_dim.x, (result_height + block_dim.y - 1) / block_dim.y);

    nn_upscale_kernel<<<grid_dim, block_dim>>>((double*)(*d_source_image), source_image_height, source_image_width, (double*)(*d_result), result_height, result_width);
}