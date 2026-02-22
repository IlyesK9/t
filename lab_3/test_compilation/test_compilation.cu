#include "test.h"

#include <cstdint>
#include <string>
#include <vector>

#include "image/bitmap_image.h"
#include "gpu_reductions.h"
#include "gpu_memory_management.h"
#include "gpu_NN_upscaling.h"
#include "gpu_LIN_upscaling.h"
#include "gpu_downsampling.h"
#include "gpu_matrix_convolution.h"



TEST_F(CompileTest, test_node) {
    GrayscaleImage g;
    std::uint32_t x = g.height;
    x = g.width;
    g.pixels.resize(sizeof(double)*4);
    BitmapImage bitmap(2,2);
    g.convert_bitmap(bitmap);
    IntermediateImage ii;
    IntermediateImage ii2(40,60);
    ii.resize(1,5);
    g.convert_intermediate_image(ii);
    ii2.load_grayscale_image(g);
    ii.update_max_pixel_value();
    ii.update_min_pixel_value();
    ii.apply_sobel_filter();


    void* dptr;
    allocate_device_memory(ii, &dptr);
    copy_data_to_device(ii, &dptr);
    copy_data_from_device(&dptr, ii);
    double r = get_max_value(&dptr, ii.height, ii.width);

    void* r2;
    NN_image_upscaling(&dptr, ii.height, ii.width, &r2);
    std::uint32_t a = get_NN_upscaled_width(ii.width);

    std::uint32_t b = get_NN_upscaled_height(ii.height);    

    
    void* dptr_2;
    allocate_device_memory(ii2, &dptr_2);
    void* kernel;
    cudaMalloc(&kernel, sizeof(double)*9);
    void* res2;
    matrix_convolution(&dptr_2, ii2.width, ii2.height, &kernel, 3, 3, &res2);
    cudaFree(kernel);

    void* res3;
    LIN_image_upscaling(&dptr_2, ii2.height, ii2.width, &res3);    
    std::uint32_t c= get_LIN_upscaled_width(ii2.width);

    std::uint32_t d = get_LIN_upscaled_height(ii2.height);
    
    void* res4;
    image_downsampling(&dptr_2, ii2.height, ii2.width, &res4);

}
