#include <filesystem>
#include <stdexcept>

#include "test.h"
#include "io/image_parser.h"
#include "gpu_memory_management.h"
#include "gpu_NN_upscaling.h"
#include "gpu_LIN_upscaling.h"

TEST_F(DevTest, gpu_NN_upscaling) {
    // define a test "image"
    IntermediateImage ii;
    ii.resize(2,2);
    ii.pixels[0] = 4;
    ii.pixels[1] = 214;
    ii.pixels[2] = 9;
    ii.pixels[3] = 167;
    // upload to gpu
    void* d_pixels = nullptr;
    allocate_device_memory(ii, &d_pixels);

    ASSERT_NE(d_pixels, nullptr);

    copy_data_to_device(ii, &d_pixels);

    // apply NN upscaling
    void* d_upscaling_result = nullptr;
    NN_image_upscaling(&d_pixels, ii.height, ii.width, &d_upscaling_result);

    ASSERT_NE(d_upscaling_result, nullptr);

    IntermediateImage image_upscaled;
    image_upscaled.resize(get_NN_upscaled_height(ii.height), get_NN_upscaled_width(ii.width));
    copy_data_from_device(&d_upscaling_result, image_upscaled);
    // cleanup
    free_device_memory(&d_pixels);
    free_device_memory(&d_upscaling_result);

    ASSERT_NE(image_upscaled.height, 0);
    ASSERT_NE(image_upscaled.width, 0);

    // check results
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[0]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[1]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[2]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[6]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[7]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[8]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[12]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[13]);
    ASSERT_EQ(ii.pixels[0], image_upscaled.pixels[14]);

    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[3]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[4]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[5]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[9]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[10]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[11]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[15]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[16]);
    ASSERT_EQ(ii.pixels[1], image_upscaled.pixels[17]);

    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[18]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[19]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[20]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[24]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[25]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[26]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[30]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[31]);
    ASSERT_EQ(ii.pixels[2], image_upscaled.pixels[32]);

    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[21]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[22]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[23]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[27]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[28]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[29]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[33]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[34]);
    ASSERT_EQ(ii.pixels[3], image_upscaled.pixels[35]);
}
