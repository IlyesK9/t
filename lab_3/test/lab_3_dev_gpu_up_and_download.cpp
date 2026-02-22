#include <filesystem>
#include <stdexcept>

#include "test.h"
#include "io/image_parser.h"
#include "gpu_memory_management.h"

TEST_F(DevTest, gpu_up_and_download) {
    std::filesystem::path input_path = "../input/test_image_5.bmp";
    ImageParser parser;
    auto bitmap = parser.read_bitmap(input_path);
    GrayscaleImage image;
    image.convert_bitmap(bitmap);

    IntermediateImage ii;
    ii.load_grayscale_image(image);

    IntermediateImage ii_buffer;
    ii_buffer.resize(image.height, image.width);

    void* d_pixels = nullptr;
    allocate_device_memory(ii, &d_pixels);
    ASSERT_NE(d_pixels, nullptr);
    copy_data_to_device(ii, &d_pixels);
    copy_data_from_device(&d_pixels, ii_buffer);
    free_device_memory(&d_pixels);

    ASSERT_EQ(ii.width, ii_buffer.width);
    ASSERT_EQ(ii.height, ii_buffer.height);
    
    for(std::uint32_t i = 0; i < ii.height; ++i){
        for(std::uint32_t j = 0; j < ii.width; ++j){
            ASSERT_EQ(ii.pixels[i*ii.width + j], ii_buffer.pixels[i*ii.width + j]);
        }
    }
}
