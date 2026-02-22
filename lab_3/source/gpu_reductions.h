#pragma once
#include "intermediate_image.h"
#include "common.cuh"

double get_max_value(void** d_source_image, std::uint32_t source_image_height, std::uint32_t source_image_width);