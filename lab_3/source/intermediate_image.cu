#include <cstdio>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cmath>
#include "intermediate_image.h"
#include "gpu_matrix_convolution.h"
#include "gpu_utilities.h"

__global__ void square_kernel(double* input, double* output, std::uint32_t n){
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = input[idx] * input[idx];
}

__global__ void add_kernel(double* a, double* b, double* output, std::uint32_t n){
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = a[idx] + b[idx];
}

__global__ void sqrt_kernel(double* input, double* output, std::uint32_t n){
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = sqrt(input[idx]);
}

void IntermediateImage::apply_sobel_filter(){
    std::uint32_t n = height * width;

    double* d_image;
    cudaMalloc(&d_image, n * sizeof(double));
    cudaMemcpy(d_image, pixels.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    double sobel_kernel_host[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    double* d_kernel;
    cudaMalloc(&d_kernel, 9 * sizeof(double));
    cudaMemcpy(d_kernel, sobel_kernel_host, 9 * sizeof(double), cudaMemcpyHostToDevice);

    void* d_F;
    matrix_convolution((void**)&d_image, width, height, (void**)&d_kernel, 3, 3, &d_F);

    double* d_F_prime;
    cudaMalloc(&d_F_prime, n * sizeof(double));

    dim3 block_dim(256);
    dim3 grid_dim((n + 255) / 256);

    square_kernel<<<grid_dim, block_dim>>>((double*)d_F, d_F_prime, n);

    double* d_F_double_prime;
    cudaMalloc(&d_F_double_prime, n * sizeof(double));
    add_kernel<<<grid_dim, block_dim>>>((double*)d_F_prime, (double*)d_F_prime, d_F_double_prime, n);

    double* d_F_triple_prime;
    cudaMalloc(&d_F_triple_prime, n * sizeof(double));
    sqrt_kernel<<<grid_dim, block_dim>>>(d_F_double_prime, d_F_triple_prime, n);

    cudaMemcpy(pixels.data(), d_F_triple_prime, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_F);
    cudaFree(d_F_prime);
    cudaFree(d_F_double_prime);
    cudaFree(d_F_triple_prime);
}