#pragma once
#include <cstdint>
#include <vector>
#include "image/bitmap_image.h"
#include "intermediate_image.h"

class IntermediateImage;

class GrayscaleImage {
    public:
        std::uint32_t height;
        std::uint32_t width;
        std::vector<std::uint8_t> pixels;

        GrayscaleImage(): height(0), width(0){};

        void convert_bitmap(BitmapImage& bitmap); // convert bitmap and overwrite the stored image in GrayscaleImage using OpenMP
        void convert_intermediate_image(IntermediateImage& image);
};