#pragma once
#include <cstdint>
#include <vector>
#include "image/bitmap_image.h"
#include "grayscale_image.h"

class GrayscaleImage;

class IntermediateImage {
    public:
        std::uint32_t height;
        std::uint32_t width;
        double min_pixel_value;
        double max_pixel_value;
        std::vector<double> pixels;

        IntermediateImage(): height(0), width(0){};
        IntermediateImage(std::uint32_t height, std::uint32_t width): height(height), width(width){
            pixels.resize(height*width);
        };

        void resize(std::uint32_t new_height, std::uint32_t new_width);

        void load_grayscale_image(GrayscaleImage& image);

        void update_min_pixel_value();

        void update_max_pixel_value();

        void apply_sobel_filter();
};