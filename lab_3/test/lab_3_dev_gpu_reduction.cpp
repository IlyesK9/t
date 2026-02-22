#include <filesystem>
#include <stdexcept>

#include "test.h"
#include "io/image_parser.h"
#include "gpu_memory_management.h"
#include "gpu_reductions.h"
#include "intermediate_image.h"

TEST_F(DevTest, gpu_reductions) {
    std::filesystem::path input_path = "../input/test_image_1.bmp";
    ImageParser parser;
    auto bitmap = parser.read_bitmap(input_path);
    GrayscaleImage image;
    image.convert_bitmap(bitmap);

    IntermediateImage ii;
    ii.load_grayscale_image(image);

    ASSERT_NE(ii.width, 0);
    ASSERT_NE(ii.height, 0);

    void* d_pixels = nullptr;
    allocate_device_memory(ii, &d_pixels);
    ASSERT_NE(d_pixels, nullptr);
    copy_data_to_device(ii, &d_pixels);

    // calculate seq results
    double seq_max = ii.pixels[0];
    for(std::size_t i = 0; i < ii.pixels.size(); ++i){
        if(ii.pixels[i] > seq_max){
            seq_max = ii.pixels[i];
        }
    }

    // calculate gpu results
    double gpu_max = get_max_value(&d_pixels, image.height, image.width);
    
    free_device_memory(&d_pixels);

    ASSERT_FLOAT_EQ(gpu_max, seq_max);

    
}
