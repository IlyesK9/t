#include <filesystem>
#include <stdexcept>

#include "test.h"
#include "io/image_parser.h"
#include "gpu_memory_management.h"
#include "gpu_matrix_convolution.h"

TEST_F(DevTest, gpu_convolution) {
    double* source_matrix = (double*) malloc(sizeof(double)*3*3);
    source_matrix[0] = 1;
    source_matrix[1] = 2;
    source_matrix[2] = 3;

    source_matrix[3] = 4;
    source_matrix[4] = 5;
    source_matrix[5] = 6;

    source_matrix[6] = 7;
    source_matrix[7] = 8;
    source_matrix[8] = 9;

    double* kernel = (double*) malloc(sizeof(double)*3*3);
    kernel[0] = -1;
    kernel[1] = -2;
    kernel[2] = -1;

    kernel[3] = 0;
    kernel[4] = 0;
    kernel[5] = 0;

    kernel[6] = 1;
    kernel[7] = 2;
    kernel[8] = 1;

    void* d_source_matrix;
    cudaMalloc(&d_source_matrix, sizeof(double)*3*3);
    cudaMemcpy(d_source_matrix, source_matrix, sizeof(double)*3*3, cudaMemcpyHostToDevice);

    void* d_kernel;
    cudaMalloc(&d_kernel, sizeof(double)*3*3);
    cudaMemcpy(d_kernel, kernel, sizeof(double)*3*3, cudaMemcpyHostToDevice);

    void* d_result = nullptr;
    matrix_convolution(&d_source_matrix, 3,3, &d_kernel, 3,3, &d_result);
    
    ASSERT_NE(d_result, nullptr);

    double* result = (double*) malloc(sizeof(double)*3*3);
    cudaMemcpy(result, d_result, sizeof(double)*3*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    ASSERT_FLOAT_EQ(result[0], -13);
    ASSERT_FLOAT_EQ(result[1], -20);
    ASSERT_FLOAT_EQ(result[2], -17);

    ASSERT_FLOAT_EQ(result[3], -18);
    ASSERT_FLOAT_EQ(result[4], -24);
    ASSERT_FLOAT_EQ(result[5], -18);
    
    ASSERT_FLOAT_EQ(result[6], 13);
    ASSERT_FLOAT_EQ(result[7], 20);
    ASSERT_FLOAT_EQ(result[8], 17);

    cudaFree(d_source_matrix);
    cudaFree(d_kernel);
    cudaFree(d_result);

    free(source_matrix);
    free(kernel);
    free(result);
}
