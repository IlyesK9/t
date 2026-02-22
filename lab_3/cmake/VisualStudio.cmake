set(lab_benchmark_additional_files "" CACHE INTERNAL "")
set(lab_lib_additional_files "" CACHE INTERNAL "")
set(lab_test_additional_files "" CACHE INTERNAL "")

if(WIN32) # Benchmark
	list(APPEND lab_benchmark_additional_files "benchmark.h")
endif()
	
if(WIN32) # Lib	
	list(APPEND lab_lib_additional_files "authors.h")
	list(APPEND lab_lib_additional_files "common.cuh")
	list(APPEND lab_lib_additional_files "gpu_downsampling.h")
	list(APPEND lab_lib_additional_files "gpu_image_operations.h")
	list(APPEND lab_lib_additional_files "gpu_LIN_upscaling.h")
	list(APPEND lab_lib_additional_files "gpu_matrix_convolution.h")
	list(APPEND lab_lib_additional_files "gpu_memory_management.h")
	list(APPEND lab_lib_additional_files "gpu_NN_upscaling.h")
	list(APPEND lab_lib_additional_files "gpu_reductions.h")
	list(APPEND lab_lib_additional_files "gpu_utilities.h")
	list(APPEND lab_lib_additional_files "grayscale_image.h")
	list(APPEND lab_lib_additional_files "intermediate_image.h")
	list(APPEND lab_lib_additional_files "utilities.h")
endif()

if(WIN32) # Tests
	list(APPEND lab_test_additional_files "test.h")	
endif()
