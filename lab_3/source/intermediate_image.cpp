#include "intermediate_image.h"

void IntermediateImage::resize(std::uint32_t new_height, std::uint32_t new_width){
    height = new_height;
    width = new_width;
    pixels.resize(new_height*new_width);
}

void IntermediateImage::load_grayscale_image(GrayscaleImage& image){
    resize(image.height, image.width);
    #pragma omp parallel for shared(pixels, image)
    for(std::int32_t i = 0; i < image.height; ++i){
        for(std::int32_t j = 0; j < image.width; ++j){
            pixels[i*image.width + j] = image.pixels[i*image.width + j];
        }
    }
}

void IntermediateImage::update_min_pixel_value(){
    if(height == 0 && width == 0){
        min_pixel_value = 0;
        return;
    }
    double min_value = pixels[0];
    #pragma omp parallel for shared(pixels) reduction(min:min_value)
    for(std::int32_t i = 0; i < height; ++i){
        for(std::int32_t j = 0; j < width; ++j){
            double pixel_value = pixels[i*width +j];
            if(pixel_value < min_value){
                min_value = pixel_value;
            }
        }
    }
    min_pixel_value = min_value;
}

void IntermediateImage::update_max_pixel_value(){
    if(height == 0 && width == 0){
        max_pixel_value = 0;
        return;
    }
    double max_value = pixels[0];
    #pragma omp parallel for shared(pixels) reduction(max:max_value)
    for(std::int32_t i = 0; i < height; ++i){
        for(std::int32_t j = 0; j < width; ++j){
            double pixel_value = pixels[i*width +j];
            if(pixel_value > max_value){
                max_value = pixel_value;
            }
        }
    }
    max_pixel_value = max_value;
}

