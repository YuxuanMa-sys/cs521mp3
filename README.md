# CS521 MP3: Automatic Differentiation

This project explores programming systems for automatic differentiation (AD) in C++ and JAX, implementing:

1. Forward-mode AD in C++ using dual numbers
2. Reverse-mode AD in JAX

## Project Structure

### Part 1: Manual Forward-Mode AD in C++

Implementation of a dual number class in C++ that supports forward-mode automatic differentiation:

- `dual_number.h`: Header-only library implementing the dual number class
- `dual_test.cpp`: Test program for the dual number implementation
- `performance_plot.cpp`: Performance comparison between dual numbers and regular calculations
- `plot_performance.py`: Python script for generating performance plots

### Part 2: Reverse-Mode AD in JAX

Implementation of reverse-mode AD using JAX for the function:
```
f(x1, x2) = ln(x1) + x1*x2 - sin(x2)
```

- `reverse_ad_jax.py`: Complete implementation comparing JAX's reverse-mode AD with a manual implementation, JIT compilation strategies, and vectorization using vmap

## Requirements

### Part 1 (C++)
- C++ compiler with C++14 support
- Python 3.x with matplotlib, numpy, and pandas (for plotting)

### Part 2 (JAX)
- Python 3.x
- JAX
- NumPy
- (Optional) GPU support via CUDA for acceleration

## Running the Code

### Part 1: C++ Implementation

```bash
# Compile and run the dual number tests
cd part1
g++ -std=c++14 -O3 dual_test.cpp -o dual_test
./dual_test

# Compile and run the performance test
g++ -std=c++14 -O3 performance_plot.cpp -o performance_plot
./performance_plot

# Generate the performance plot
python3 plot_performance.py
```

### Part 2: JAX Implementation

```bash
# Run the JAX implementation
cd part2
python3 reverse_ad_jax.py
``` 