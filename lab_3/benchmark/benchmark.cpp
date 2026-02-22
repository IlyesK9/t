#include "benchmark.h"

#include <cstdint>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include "../source/utilities.h"
#include "../source/gpu_reductions.h"

static void serial_max_benchmark(benchmark::State& state) {
    const auto n = state.range(0);
    std::vector<double> data(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(get_max_value_serial(data));
    }
}

static void openmp_max_benchmark(benchmark::State& state) {
    const auto n = state.range(0);
    std::vector<double> data(n);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }

    for (auto _ : state) {
        benchmark::DoNotOptimize(get_max_value_openmp(data));
    }
}

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    
    std::vector<int> sizes = {1000, 10000, 100000, 1000000, 10000000};
    for (int size : sizes) {
        ::benchmark::RegisterBenchmark("serial_max", serial_max_benchmark)->Unit(benchmark::kMillisecond)->Arg(size);
        ::benchmark::RegisterBenchmark("openmp_max", openmp_max_benchmark)->Unit(benchmark::kMillisecond)->Arg(size);
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}


// DUMMY
BENCHMARK(serial_max_benchmark)->Unit(benchmark::kMillisecond)->Arg(100);
BENCHMARK(openmp_max_benchmark)->Unit(benchmark::kMillisecond)->Arg(100);
