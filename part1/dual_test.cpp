#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <functional>
#include "dual_number.h"

// Test function to compute the value and derivative
void test_function(const std::string& name, 
                   const dual_number& x, 
                   const dual_number& result,
                   float expected_value,
                   float expected_derivative,
                   float tolerance = 1e-5) {
    bool value_ok = std::abs(result.value() - expected_value) < tolerance;
    bool dual_ok = std::abs(result.dual() - expected_derivative) < tolerance;
    
    std::cout << name << " test: " 
              << (value_ok && dual_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  x = " << x.value() << ", d = " << x.dual() << std::endl;
    std::cout << "  Result: value = " << result.value() 
              << ", derivative = " << result.dual() << std::endl;
    std::cout << "  Expected: value = " << expected_value 
              << ", derivative = " << expected_derivative << std::endl;
    std::cout << std::endl;
}

// Simple derivative test using dual numbers
void basic_derivative_tests() {
    std::cout << "=== Basic Derivative Tests ===" << std::endl;
    
    // Test with x = 2.0, setting dual part to 1.0 to compute derivative
    dual_number x(2.0f, 1.0f);
    
    // Addition: f(x) = x + 3
    dual_number y1 = x + dual_number(3.0f);
    test_function("Addition", x, y1, 5.0f, 1.0f);
    
    // Subtraction: f(x) = x - 1
    dual_number y2 = x - dual_number(1.0f);
    test_function("Subtraction", x, y2, 1.0f, 1.0f);
    
    // Multiplication: f(x) = x * 4
    dual_number y3 = x * dual_number(4.0f);
    test_function("Multiplication", x, y3, 8.0f, 4.0f);
    
    // Division: f(x) = x / 2
    dual_number y4 = x / dual_number(2.0f);
    test_function("Division", x, y4, 1.0f, 0.5f);
    
    // Sin: f(x) = sin(x)
    dual_number y5 = sin(x);
    test_function("Sin", x, y5, std::sin(2.0f), std::cos(2.0f));
    
    // Cos: f(x) = cos(x)
    dual_number y6 = cos(x);
    test_function("Cos", x, y6, std::cos(2.0f), -std::sin(2.0f));
    
    // Exp: f(x) = exp(x)
    dual_number y7 = exp(x);
    test_function("Exp", x, y7, std::exp(2.0f), std::exp(2.0f));
    
    // Log: f(x) = ln(x)
    dual_number y8 = ln(x);
    test_function("Ln", x, y8, std::log(2.0f), 1.0f/2.0f);
    
    // ReLU: f(x) = relu(x)
    dual_number y9 = relu(x);
    test_function("ReLU (positive)", x, y9, 2.0f, 1.0f);
    
    // ReLU with negative input
    dual_number neg_x(-1.0f, 1.0f);
    dual_number y10 = relu(neg_x);
    test_function("ReLU (negative)", neg_x, y10, 0.0f, 0.0f);
    
    // Sigmoid: f(x) = sigmoid(x)
    dual_number y11 = sigmoid(x);
    float sig_val = 1.0f / (1.0f + std::exp(-2.0f));
    float sig_deriv = sig_val * (1.0f - sig_val);
    test_function("Sigmoid", x, y11, sig_val, sig_deriv);
    
    // Tanh: f(x) = tanh(x)
    dual_number y12 = tanh(x);
    float tanh_val = std::tanh(2.0f);
    float tanh_deriv = 1.0f - tanh_val * tanh_val;
    test_function("Tanh", x, y12, tanh_val, tanh_deriv);
}

// Test more complex functions combining operations
void complex_function_tests() {
    std::cout << "=== Complex Function Tests ===" << std::endl;
    
    // f(x) = sin(x) * cos(x) = 0.5 * sin(2x)
    // f'(x) = cos^2(x) - sin^2(x) = cos(2x)
    dual_number x(1.0f, 1.0f);
    dual_number y = sin(x) * cos(x);
    float expected_val = std::sin(1.0f) * std::cos(1.0f);
    float expected_deriv = std::cos(2.0f);
    test_function("sin(x) * cos(x)", x, y, expected_val, expected_deriv);
    
    // f(x) = exp(sin(x))
    // f'(x) = exp(sin(x)) * cos(x)
    dual_number z = exp(sin(x));
    float expected_val_2 = std::exp(std::sin(1.0f));
    float expected_deriv_2 = expected_val_2 * std::cos(1.0f);
    test_function("exp(sin(x))", x, z, expected_val_2, expected_deriv_2);
    
    // f(x) = ln(x^2 + 1)
    // f'(x) = 2x / (x^2 + 1)
    dual_number w = ln(x * x + dual_number(1.0f));
    float expected_val_3 = std::log(1.0f * 1.0f + 1.0f);
    float expected_deriv_3 = 2.0f * 1.0f / (1.0f * 1.0f + 1.0f);
    test_function("ln(x^2 + 1)", x, w, expected_val_3, expected_deriv_3);
}

// Test vector operations
void vector_operation_tests() {
    std::cout << "=== Vector Operation Tests ===" << std::endl;
    
    // Create vectors of dual numbers
    dual_vector v1(3);
    dual_vector v2(3);
    
    // Initialize the vectors
    for (size_t i = 0; i < 3; ++i) {
        v1[i] = dual_number(i + 1.0f, 1.0f);  // Values: 1.0, 2.0, 3.0
        v2[i] = dual_number(i * 2.0f, 0.0f);  // Values: 0.0, 2.0, 4.0
    }
    
    // Test vector addition
    dual_vector v3 = v1 + v2;
    std::cout << "Vector addition test:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  v3[" << i << "] = " << v3[i].value() 
                  << ", derivative = " << v3[i].dual() << std::endl;
    }
    std::cout << std::endl;
    
    // Test element-wise functions on vectors
    dual_vector v4 = sin(v1);
    std::cout << "Vector sin(v1) test:" << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        std::cout << "  sin(v1[" << i << "]) = " << v4[i].value() 
                  << ", derivative = " << v4[i].dual() << std::endl;
    }
    std::cout << std::endl;
}

// Performance comparison: dual_number vs regular calculation
void performance_test() {
    std::cout << "=== Performance Tests ===" << std::endl;
    const int iterations = 100000000;
    
    // Function to test: f(x) = sin(x) * cos(x) * exp(x)
    // Regular calculation (value only)
    auto start_regular = std::chrono::high_resolution_clock::now();
    float result_regular = 0.0f;
    float x_val = 1.5f;
    
    for (int i = 0; i < iterations; ++i) {
        result_regular = std::sin(x_val) * std::cos(x_val) * std::exp(x_val);
    }
    
    auto end_regular = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_regular = end_regular - start_regular;
    
    // Dual number calculation (value and derivative)
    auto start_dual = std::chrono::high_resolution_clock::now();
    dual_number x_dual(x_val, 1.0f);
    dual_number result_dual(0.0f, 0.0f);
    
    for (int i = 0; i < iterations; ++i) {
        result_dual = sin(x_dual) * cos(x_dual) * exp(x_dual);
    }
    
    auto end_dual = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_dual = end_dual - start_dual;
    
    // Print results
    std::cout << "Regular calculation (value only):" << std::endl;
    std::cout << "  Result: " << result_regular << std::endl;
    std::cout << "  Time: " << time_regular.count() << " ms" << std::endl;
    
    std::cout << "Dual number calculation (value and derivative):" << std::endl;
    std::cout << "  Result: " << result_dual.value() << ", derivative: " 
              << result_dual.dual() << std::endl;
    std::cout << "  Time: " << time_dual.count() << " ms" << std::endl;
    
    std::cout << "Overhead factor: " << (time_dual.count() / time_regular.count()) 
              << "x" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "Testing Dual Number Automatic Differentiation" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;
    
    basic_derivative_tests();
    complex_function_tests();
    vector_operation_tests();
    performance_test();
    
    std::cout << "Compiler optimizations that help keep overhead low:" << std::endl;
    std::cout << "1. Function inlining - reduces function call overhead" << std::endl;
    std::cout << "2. Constant folding - evaluates constant expressions at compile time" << std::endl;
    std::cout << "3. Common subexpression elimination - avoids redundant calculations" << std::endl;
    std::cout << "4. Loop unrolling - reduces loop overhead" << std::endl;
    std::cout << "5. SIMD vectorization - can perform multiple operations at once" << std::endl;
    std::cout << "6. Register allocation - keeps frequently used values in CPU registers" << std::endl;
    
    return 0;
} 