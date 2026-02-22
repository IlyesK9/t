#pragma once
#include "intermediate_image.h"
#include "common.cuh"

// upscale the given image using the GPU
// allocates device memory for d_result
void LIN_image_upscaling(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width, void** d_result);

std::uint32_t get_LIN_upscaled_width(std::uint32_t image_width);

std::uint32_t get_LIN_upscaled_height(std::uint32_t image_height);