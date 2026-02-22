#include "gpu_LIN_upscaling.h"
#include "gpu_memory_management.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

std::uint32_t get_LIN_upscaled_width(std::uint32_t image_width){
    return 2 * image_width - 1;
}

std::uint32_t get_LIN_upscaled_height(std::uint32_t image_height){
    return 2 * image_height - 1;
}

__global__ void copy_original_pixels_kernel(const double* source, std::uint32_t source_height, std::uint32_t source_width, double* result, std::uint32_t result_width){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= source_width || y >= source_height) {
        return;
    }

    std::uint32_t result_x = 2 * x;
    std::uint32_t result_y = 2 * y;
    result[result_y * result_width + result_x] = source[y * source_width + x];
}

__global__ void interpolate_axis_kernel(const double* result, std::uint32_t result_height, std::uint32_t result_width, double* output){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= result_width || y >= result_height) {
        return;
    }

    if (x % 2 == 0 && y % 2 == 0) {
        output[y * result_width + x] = result[y * result_width + x];
    } else if (x % 2 == 1 && y % 2 == 0) {
        double left = result[y * result_width + (x - 1)];
        double right = result[y * result_width + (x + 1)];
        output[y * result_width + x] = (left + right) / 2.0;
    } else if (x % 2 == 0 && y % 2 == 1) {
        double top = result[(y - 1) * result_width + x];
        double bottom = result[(y + 1) * result_width + x];
        output[y * result_width + x] = (top + bottom) / 2.0;
    }
}

__global__ void interpolate_diagonal_kernel(const double* result, std::uint32_t result_height, std::uint32_t result_width, double* output){
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= result_width || y >= result_height) {
        return;
    }

    if (x % 2 == 1 && y % 2 == 1) {
        double tl = result[(y - 1) * result_width + (x - 1)];
        double tr = result[(y - 1) * result_width + (x + 1)];
        double bl = result[(y + 1) * result_width + (x - 1)];
        double br = result[(y + 1) * result_width + (x + 1)];
        output[y * result_width + x] = (tl + tr + bl + br) / 4.0;
    } else {
        output[y * result_width + x] = result[y * result_width + x];
    }
}

void LIN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::uint32_t result_height = get_LIN_upscaled_height(source_image_height);
    std::uint32_t result_width = get_LIN_upscaled_width(source_image_width);
    size_t result_size = result_height * result_width;

    double* d_temp;
    cudaMalloc(d_result, result_size * sizeof(double));
    cudaMalloc(&d_temp, result_size * sizeof(double));

    dim3 block_dim(16, 16);
    dim3 grid_dim_source((source_image_width + block_dim.x - 1) / block_dim.x, (source_image_height + block_dim.y - 1) / block_dim.y);
    copy_original_pixels_kernel<<<grid_dim_source, block_dim>>>((double*)(*d_source_image), source_image_height, source_image_width, (double*)(*d_result), result_width);

    dim3 grid_dim_result((result_width + block_dim.x - 1) / block_dim.x, (result_height + block_dim.y - 1) / block_dim.y);
    interpolate_axis_kernel<<<grid_dim_result, block_dim>>>((double*)(*d_result), result_height, result_width, d_temp);
    interpolate_diagonal_kernel<<<grid_dim_result, block_dim>>>(d_temp, result_height, result_width, (double*)(*d_result));

    cudaFree(d_temp);
}