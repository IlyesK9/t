#pragma once
#include <cstdint>
#include <vector>

double get_max_value_serial(const std::vector<double>& data);

double get_max_value_openmp(const std::vector<double>& data);
