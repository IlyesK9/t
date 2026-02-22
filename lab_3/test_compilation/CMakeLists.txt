enable_testing()

# add custom target to check compilability and linking of code without executing the tests

# # googletest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest
  GIT_TAG release-1.12.1)

# google test
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(
  WRITE ${CMAKE_BINARY_DIR}/test_get_path.cpp
  "#include <filesystem>\nstd::filesystem::path get_path() { return \"${CMAKE_SOURCE_DIR}\"; }"
)

add_executable(lab_test_compilation ${SRC_TST})
target_sources(
  lab_test_compilation
  PRIVATE ${CMAKE_BINARY_DIR}/test_get_path.cpp
          test.cpp
          test_compilation.cu 

		  # for visual studio
		  ${lab_test_additional_files})
		  
target_include_directories(lab_test_compilation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab_test_compilation PRIVATE project_options project_libraries)
target_link_libraries(lab_test_compilation PRIVATE lab_lib)
target_link_libraries(lab_test_compilation PRIVATE gtest)


add_test(NAME SerialTests COMMAND lab_test_compilation)

get_target_property(lab_test_sources lab_test_compilation SOURCES)
source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${lab_test_sources})
