#include "gpu_memory_management.h"
#include <cuda_runtime_api.h>
#include <cuda.h>


void allocate_device_memory(IntermediateImage& image, void** devPtr){
	size_t byte_count = image.pixels.size() * sizeof(double);
	cudaMalloc(devPtr, byte_count);
}

void free_device_memory(void** devPtr){
	cudaFree(*devPtr);
	*devPtr = nullptr;
}

void copy_data_to_device(IntermediateImage& image, void** devPtr){
	size_t byte_count = image.pixels.size() * sizeof(double);
	cudaMemcpy(*devPtr, image.pixels.data(), byte_count, cudaMemcpyHostToDevice);
}

void copy_data_from_device(void** devPtr, IntermediateImage& image){
	size_t byte_count = image.pixels.size() * sizeof(double);
	cudaMemcpy(image.pixels.data(), *devPtr, byte_count, cudaMemcpyDeviceToHost);
}