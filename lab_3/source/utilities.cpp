#include "utilities.h"
#include <omp.h>
#include <algorithm>

double get_max_value_serial(const std::vector<double>& data) {
    if (data.empty()) return -1e100;
    double max_val = data[0];
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

double get_max_value_openmp(const std::vector<double>& data) {
    if (data.empty()) return -1e100;
    double max_val = data[0];
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 1; i < (int)data.size(); i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}
