add_library(lab_lib STATIC)
target_sources(
  lab_lib
  PRIVATE 		  
      main.cpp
      
      image/bitmap_image.cpp
      io/image_parser.cpp

      grayscale_image.cpp
      intermediate_image.cpp
      intermediate_image.cu
      gpu_memory_management.cu
      gpu_downsampling.cu
      gpu_NN_upscaling.cu
      gpu_LIN_upscaling.cu
      gpu_reductions.cu
      gpu_matrix_convolution.cu
      gpu_utilities.cu
      utilities.cpp
	
		  # for visual studio
		  ${lab_lib_additional_files})

target_include_directories(lab_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(lab_lib PROPERTIES ENABLE_EXPORTS 1)
target_link_libraries(lab_lib PUBLIC project_options project_libraries)

# lab executable
add_executable(lab)
target_sources(lab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(lab PRIVATE lab_lib)

set_target_properties(lab PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(lab_lib PUBLIC project_options project_libraries)

get_target_property(lab_lib_sources lab_lib SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${lab_lib_sources})
