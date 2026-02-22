#include "gpu_downsampling.h"
#include "gpu_memory_management.h"
#include <cuda_runtime.h>
#include <cstdio>

std::uint32_t get_downsampled_width(std::uint32_t image_width){
    return (image_width + 2) / 3;
}

std::uint32_t get_downsampled_height(std::uint32_t image_height){
    return (image_height + 2) / 3;
}

__global__ void downsample_kernel(const double* source, std::uint32_t source_height, std::uint32_t source_width, double* result, std::uint32_t result_height, std::uint32_t result_width){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= result_width || y >= result_height) {
        return;
    }

    double sum = 0.0;
    int count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int src_x = x * 3 + dx;
            int src_y = y * 3 + dy;

            if (src_x >= 0 && src_x < source_width && src_y >= 0 && src_y < source_height) {
                sum += source[src_y * source_width + src_x];
                count++;
            }
        }
    }

    result[y * result_width + x] = sum / count;
}

void image_downsampling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result){
    std::uint32_t result_height = get_downsampled_height(source_image_height);
    std::uint32_t result_width = get_downsampled_width(source_image_width);
    size_t result_size = result_height * result_width;

    cudaMalloc(d_result, result_size * sizeof(double));

    dim3 block_dim(16, 16);
    dim3 grid_dim((result_width + block_dim.x - 1) / block_dim.x, (result_height + block_dim.y - 1) / block_dim.y);

    downsample_kernel<<<grid_dim, block_dim>>>((double*)(*d_source_image), source_image_height, source_image_width, (double*)(*d_result), result_height, result_width);
}