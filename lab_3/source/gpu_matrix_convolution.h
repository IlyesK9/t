#pragma once
#include "common.cuh"
#include <cstdint>

// calculate the 2D convolution 
// allocates device memory for d_result
// assumes, d_source_mat and d_kernel point to valid and initialized memory on the device
// assumes, d_result does not contain any valid pointer and may be overwritten
// stores a pointer to the result on the device in d_result
// assumes, kernel is square and kernel_dim is correctly set
// it may be assumed, that the kernel is smaller than the matrix and fits within it in both dimensions
void matrix_convolution(void** d_source_matrix, std::uint32_t matrix_width, std::uint32_t matrix_height, void** d_kernel, std::uint32_t kernel_width, std::uint32_t kernel_height, void** d_result);