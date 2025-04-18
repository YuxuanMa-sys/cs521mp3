#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <functional>
#include "dual_number.h"

// Function to measure the execution time of a function
template<typename Func>
double measure_time(Func func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    func(iterations);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

// Test different functions with varying complexity
void test_performance() {
    std::ofstream csv_file("performance_data.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open output file!" << std::endl;
        return;
    }

    // CSV header
    csv_file << "Function,Regular Time (ms),Dual Number Time (ms),Overhead Factor\n";

    // Number of iterations for reliable measurements
    const int iterations = 10000000;

    // Test function 1: f(x) = sin(x)
    {
        float x_val = 1.5f;
        dual_number x_dual(x_val, 1.0f);

        auto regular_func = [x_val](int n) {
            float result = 0.0f;
            for (int i = 0; i < n; ++i) {
                result = std::sin(x_val);
            }
            return result;
        };

        auto dual_func = [x_dual](int n) {
            dual_number result(0.0f, 0.0f);
            for (int i = 0; i < n; ++i) {
                result = sin(x_dual);
            }
            return result;
        };

        double regular_time = measure_time(regular_func, iterations);
        double dual_time = measure_time(dual_func, iterations);
        double overhead = dual_time / regular_time;

        csv_file << "sin(x)," << regular_time << "," << dual_time << "," << overhead << "\n";
        
        std::cout << "sin(x):" << std::endl;
        std::cout << "  Regular: " << regular_time << " ms" << std::endl;
        std::cout << "  Dual: " << dual_time << " ms" << std::endl;
        std::cout << "  Overhead: " << overhead << "x" << std::endl << std::endl;
    }

    // Test function 2: f(x) = x^2
    {
        float x_val = 1.5f;
        dual_number x_dual(x_val, 1.0f);

        auto regular_func = [x_val](int n) {
            float result = 0.0f;
            for (int i = 0; i < n; ++i) {
                result = x_val * x_val;
            }
            return result;
        };

        auto dual_func = [x_dual](int n) {
            dual_number result(0.0f, 0.0f);
            for (int i = 0; i < n; ++i) {
                result = x_dual * x_dual;
            }
            return result;
        };

        double regular_time = measure_time(regular_func, iterations);
        double dual_time = measure_time(dual_func, iterations);
        double overhead = dual_time / regular_time;

        csv_file << "x^2," << regular_time << "," << dual_time << "," << overhead << "\n";
        
        std::cout << "x^2:" << std::endl;
        std::cout << "  Regular: " << regular_time << " ms" << std::endl;
        std::cout << "  Dual: " << dual_time << " ms" << std::endl;
        std::cout << "  Overhead: " << overhead << "x" << std::endl << std::endl;
    }

    // Test function 3: f(x) = sin(x) * cos(x)
    {
        float x_val = 1.5f;
        dual_number x_dual(x_val, 1.0f);

        auto regular_func = [x_val](int n) {
            float result = 0.0f;
            for (int i = 0; i < n; ++i) {
                result = std::sin(x_val) * std::cos(x_val);
            }
            return result;
        };

        auto dual_func = [x_dual](int n) {
            dual_number result(0.0f, 0.0f);
            for (int i = 0; i < n; ++i) {
                result = sin(x_dual) * cos(x_dual);
            }
            return result;
        };

        double regular_time = measure_time(regular_func, iterations);
        double dual_time = measure_time(dual_func, iterations);
        double overhead = dual_time / regular_time;

        csv_file << "sin(x)*cos(x)," << regular_time << "," << dual_time << "," << overhead << "\n";
        
        std::cout << "sin(x)*cos(x):" << std::endl;
        std::cout << "  Regular: " << regular_time << " ms" << std::endl;
        std::cout << "  Dual: " << dual_time << " ms" << std::endl;
        std::cout << "  Overhead: " << overhead << "x" << std::endl << std::endl;
    }

    // Test function 4: f(x) = exp(sin(x))
    {
        float x_val = 1.5f;
        dual_number x_dual(x_val, 1.0f);

        auto regular_func = [x_val](int n) {
            float result = 0.0f;
            for (int i = 0; i < n; ++i) {
                result = std::exp(std::sin(x_val));
            }
            return result;
        };

        auto dual_func = [x_dual](int n) {
            dual_number result(0.0f, 0.0f);
            for (int i = 0; i < n; ++i) {
                result = exp(sin(x_dual));
            }
            return result;
        };

        double regular_time = measure_time(regular_func, iterations);
        double dual_time = measure_time(dual_func, iterations);
        double overhead = dual_time / regular_time;

        csv_file << "exp(sin(x))," << regular_time << "," << dual_time << "," << overhead << "\n";
        
        std::cout << "exp(sin(x)):" << std::endl;
        std::cout << "  Regular: " << regular_time << " ms" << std::endl;
        std::cout << "  Dual: " << dual_time << " ms" << std::endl;
        std::cout << "  Overhead: " << overhead << "x" << std::endl << std::endl;
    }

    // Test function 5: f(x) = ln(x^2 + 1)
    {
        float x_val = 1.5f;
        dual_number x_dual(x_val, 1.0f);

        auto regular_func = [x_val](int n) {
            float result = 0.0f;
            for (int i = 0; i < n; ++i) {
                result = std::log(x_val * x_val + 1.0f);
            }
            return result;
        };

        auto dual_func = [x_dual](int n) {
            dual_number result(0.0f, 0.0f);
            for (int i = 0; i < n; ++i) {
                result = ln(x_dual * x_dual + dual_number(1.0f));
            }
            return result;
        };

        double regular_time = measure_time(regular_func, iterations);
        double dual_time = measure_time(dual_func, iterations);
        double overhead = dual_time / regular_time;

        csv_file << "ln(x^2+1)," << regular_time << "," << dual_time << "," << overhead << "\n";
        
        std::cout << "ln(x^2+1):" << std::endl;
        std::cout << "  Regular: " << regular_time << " ms" << std::endl;
        std::cout << "  Dual: " << dual_time << " ms" << std::endl;
        std::cout << "  Overhead: " << overhead << "x" << std::endl << std::endl;
    }

    // Test function 6: f(x) = sin(x) * cos(x) * exp(x) * ln(x^2 + 1)
    {
        float x_val = 1.5f;
        dual_number x_dual(x_val, 1.0f);

        auto regular_func = [x_val](int n) {
            float result = 0.0f;
            for (int i = 0; i < n; ++i) {
                result = std::sin(x_val) * std::cos(x_val) * std::exp(x_val) * std::log(x_val * x_val + 1.0f);
            }
            return result;
        };

        auto dual_func = [x_dual](int n) {
            dual_number result(0.0f, 0.0f);
            for (int i = 0; i < n; ++i) {
                result = sin(x_dual) * cos(x_dual) * exp(x_dual) * ln(x_dual * x_dual + dual_number(1.0f));
            }
            return result;
        };

        double regular_time = measure_time(regular_func, iterations);
        double dual_time = measure_time(dual_func, iterations);
        double overhead = dual_time / regular_time;

        csv_file << "complex," << regular_time << "," << dual_time << "," << overhead << "\n";
        
        std::cout << "Complex function:" << std::endl;
        std::cout << "  Regular: " << regular_time << " ms" << std::endl;
        std::cout << "  Dual: " << dual_time << " ms" << std::endl;
        std::cout << "  Overhead: " << overhead << "x" << std::endl << std::endl;
    }

    csv_file.close();
}

int main() {
    std::cout << "Measuring performance overhead of dual number calculations" << std::endl;
    std::cout << "=====================================================" << std::endl << std::endl;
    
    test_performance();
    
    std::cout << "Performance data has been written to performance_data.csv" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Classical compiler optimizations that help keep the overhead low:" << std::endl;
    std::cout << "1. Function inlining - reduces function call overhead" << std::endl;
    std::cout << "2. Constant folding - evaluates constant expressions at compile time" << std::endl;
    std::cout << "3. Common subexpression elimination - avoids redundant calculations" << std::endl;
    std::cout << "4. Loop unrolling - reduces loop overhead" << std::endl;
    std::cout << "5. SIMD vectorization - can perform multiple operations simultaneously" << std::endl;
    std::cout << "6. Register allocation - keeps frequently used values in CPU registers" << std::endl;
    std::cout << "7. Dead code elimination - removes unnecessary computations" << std::endl;
    std::cout << "8. Strength reduction - replaces expensive operations with cheaper ones" << std::endl;
    
    return 0;
} 