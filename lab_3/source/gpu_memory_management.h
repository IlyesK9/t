#pragma once
#include "intermediate_image.h"
#include "common.cuh"


void allocate_device_memory(IntermediateImage& image, void** devPtr);

void free_device_memory(void** devPtr);

void copy_data_to_device(IntermediateImage& image, void** devPtr);

void copy_data_from_device(void** devPtr, IntermediateImage& image);