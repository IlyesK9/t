#include "grayscale_image.h"
#include <iostream>
#include <omp.h>
#include "intermediate_image.h"


void GrayscaleImage::convert_bitmap(BitmapImage& bitmap){
	height = bitmap.get_height();
	width = bitmap.get_width();
	pixels.resize(height * width);

    auto total_pixels = height * width;

	#pragma omp parallel for
	for (std::int64_t i = 0; i < total_pixels; ++i) {
		auto& pixel = bitmap.pixels[i];
		double r = static_cast<double>(pixel.get_red_channel());
		double g = static_cast<double>(pixel.get_green_channel());
		double b = static_cast<double>(pixel.get_blue_channel());
		double L = 0.2126 * r + 0.7152 * g + 0.0722 * b + 0.5;
		pixels[i] = static_cast<std::uint8_t>(L);
	}
}

void GrayscaleImage::convert_intermediate_image(IntermediateImage& image){
	image.update_min_pixel_value();
	image.update_max_pixel_value();

	height = image.height;
	width = image.width;
	pixels.resize(height * width);

	double min_value = image.min_pixel_value;
	double max_value = image.max_pixel_value;
	if (min_value >= 0.0 && max_value <= 255.0) {
		min_value = 0.0;
		max_value = 255.0;
	}

	auto total_pixels = height * width;

	#pragma omp parallel for
	for (std::int64_t i = 0; i < total_pixels; ++i) {
		double v = image.pixels[i];
		double g = 0.0;
		if (max_value != min_value) {
			g = (v - min_value) * 255.0 / (max_value - min_value);
		}
		if (g < 0.0) {
			g = 0.0;
		} else if (g > 255.0) {
			g = 255.0;
		}
		pixels[i] = static_cast<std::uint8_t>(g);
	}
}