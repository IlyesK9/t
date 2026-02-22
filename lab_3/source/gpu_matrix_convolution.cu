#include "gpu_matrix_convolution.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <string>

__global__ void convolution_kernel(const double* source, std::uint32_t matrix_width, std::uint32_t matrix_height, const double* kernel, std::uint32_t kernel_width, std::uint32_t kernel_height, double* result){
    std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= matrix_width || y >= matrix_height) {
        return;
    }

    const std::int32_t a = kernel_width / 2;
    double sum = 0.0;

    for (std::int32_t i = 0; i < (std::int32_t)kernel_width; i++) {
        for (std::int32_t j = 0; j < (std::int32_t)kernel_height; j++) {
            std::int32_t src_x = (std::int32_t)x - i + a;
            std::int32_t src_y = (std::int32_t)y - j + a;

            double src_val = 0.0;
            if (src_x >= 0 && src_x < (std::int32_t)matrix_width && src_y >= 0 && src_y < (std::int32_t)matrix_height) {
                src_val = source[src_y * matrix_width + src_x];
            }

            double kern_val = kernel[j * kernel_width + i];
            sum += src_val * kern_val;
        }
    }

    result[y * matrix_width + x] = sum;
}

void matrix_convolution(void** d_source_matrix, std::uint32_t matrix_width, std::uint32_t matrix_height, void** d_kernel, std::uint32_t kernel_width, std::uint32_t kernel_height, void** d_result){
    const size_t result_size = matrix_height * matrix_width;
    cudaMalloc(d_result, result_size * sizeof(double));

    dim3 block_dim(16, 16);
    dim3 grid_dim((matrix_width + block_dim.x - 1) / block_dim.x, (matrix_height + block_dim.y - 1) / block_dim.y);

    convolution_kernel<<<grid_dim, block_dim>>>((double*)(*d_source_matrix), matrix_width, matrix_height, (double*)(*d_kernel), kernel_width, kernel_height, (double*)(*d_result));
}