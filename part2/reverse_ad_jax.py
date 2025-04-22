import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, make_jaxpr
import timeit
import numpy as np

# Start with CPU platform
jax.config.update('jax_platform_name', 'cpu')

# Check what devices we have
cpu_devices = jax.devices('cpu')
try:
    gpu_devices = jax.devices('gpu')
    gpu_available = len(gpu_devices) > 0
    if gpu_available:
        print(f"GPU device(s) found: {len(gpu_devices)}")
        for i, dev in enumerate(gpu_devices):
            print(f"  GPU {i}: {dev}")
    else:
        print("No GPU devices found, will use CPU only")
except:
    gpu_available = False
    print("GPU not available or JAX GPU support not installed")

print(f"CPU device(s) found: {len(cpu_devices)}")
for i, dev in enumerate(cpu_devices):
    print(f"  CPU {i}: {dev}")

def f(x1, x2):
    """Function: f(x1, x2) = ln(x1) + x1*x2 - sin(x2)"""
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

# 1. Compute gradients using jax.grad
def compute_gradients_positional():
    """Compute gradients using jax.grad with positional parameters"""
    # Get gradients for each input
    dy_dx1 = grad(f, argnums=0)
    dy_dx2 = grad(f, argnums=1)
    
    # Test values
    x1_val = 2.0
    x2_val = 5.0
    
    # Compute values
    f_val = f(x1_val, x2_val)
    grad_x1 = dy_dx1(x1_val, x2_val)
    grad_x2 = dy_dx2(x1_val, x2_val)
    
    print(f"Function value at x1={x1_val}, x2={x2_val}: {f_val}")
    print(f"Gradient with respect to x1: {grad_x1}")
    print(f"Gradient with respect to x2: {grad_x2}")
    
    return dy_dx1, dy_dx2

# 2. Print the jaxpr code for gradients
def print_jaxpr_code():
    """Print and analyze the jaxpr code for gradients"""
    dy_dx1 = grad(f, argnums=0)
    dy_dx2 = grad(f, argnums=1)
    
    # Get jaxpr representation
    jaxpr_dx1 = make_jaxpr(dy_dx1)(2.0, 5.0)
    jaxpr_dx2 = make_jaxpr(dy_dx2)(2.0, 5.0)
    
    print("\nJAXPR for gradient with respect to x1:")
    print(jaxpr_dx1)
    
    print("\nJAXPR for gradient with respect to x2:")
    print(jaxpr_dx2)
    
    # Count lines in jaxpr code
    jaxpr_dx1_str = str(jaxpr_dx1)
    jaxpr_dx2_str = str(jaxpr_dx2)
    
    dx1_lines = jaxpr_dx1_str.count('\n')
    dx2_lines = jaxpr_dx2_str.count('\n')
    
    print(f"\nNumber of lines in jaxpr for gradient with respect to x1: {dx1_lines}")
    print(f"Number of lines in jaxpr for gradient with respect to x2: {dx2_lines}")
    
    # Compare with original function
    jaxpr_f = make_jaxpr(f)(2.0, 5.0)
    f_lines = str(jaxpr_f).count('\n')
    
    print(f"Number of lines in jaxpr for original function: {f_lines}")
    print(f"Ratio of lines for grad(x1) vs original: {dx1_lines/f_lines:.2f}x")
    print(f"Ratio of lines for grad(x2) vs original: {dx2_lines/f_lines:.2f}x")

# 3. Classical reverse mode AD implementation
def classical_reverse_ad(x1, x2):
    """Manual implementation of reverse-mode AD"""
    # Forward pass
    v1 = jnp.log(x1)    # ln(x1)
    v2 = x1 * x2        # x1*x2
    v3 = jnp.sin(x2)    # sin(x2)
    v4 = v1 + v2        # ln(x1) + x1*x2
    y = v4 - v3         # ln(x1) + x1*x2 - sin(x2)
    
    # Backward pass
    dy_dy = 1.0
    dy_dv4 = dy_dy
    dy_dv3 = -dy_dy
    
    dy_dv1 = dy_dv4
    dy_dv2 = dy_dv4
    
    dy_dx1_v1 = 1.0 / x1
    dy_dx1_v2 = x2
    
    dy_dx2_v2 = x1
    dy_dx2_v3 = jnp.cos(x2)
    
    # Accumulate gradients
    dy_dx1 = dy_dv1 * dy_dx1_v1 + dy_dv2 * dy_dx1_v2
    dy_dx2 = dy_dv2 * dy_dx2_v2 + dy_dv3 * dy_dx2_v3
    
    return y, dy_dx1, dy_dx2

# 4. Compile to HLO representation
def compile_to_hlo():
    """Compile to HLO and save for analysis"""
    dy_dx1 = jit(grad(f, argnums=0))
    dy_dx2 = jit(grad(f, argnums=1))
    
    # Get HLO representation
    hlo_dx1 = jit(dy_dx1).lower(2.0, 5.0).compiler_ir(dialect='hlo')
    hlo_dx2 = jit(dy_dx2).lower(2.0, 5.0).compiler_ir(dialect='hlo')
    
    print("\nHLO for gradient with respect to x1 (truncated):")
    hlo_dx1_str = str(hlo_dx1)
    print(hlo_dx1_str[:500] + "...\n[output truncated]")
    
    print("\nHLO for gradient with respect to x2 (truncated):")
    hlo_dx2_str = str(hlo_dx2)
    print(hlo_dx2_str[:500] + "...\n[output truncated]")
    
    # Save complete HLO to files
    with open("hlo_dx1.txt", "w") as file:
        file.write(hlo_dx1_str)
    
    with open("hlo_dx2.txt", "w") as file:
        file.write(hlo_dx2_str)
    
    print("\nFull HLO saved to part2/hlo_dx1.txt and part2/hlo_dx2.txt")

# 5. Compare runtime of JIT approaches
def compare_runtime():
    """Compare performance of multiple vs single JIT calls"""
    # Test inputs
    x1 = 2.0
    x2 = 5.0
    
    # Approach 1: Multiple jit calls
    g1 = lambda x1, x2: (jit(f)(x1, x2), jit(grad(f, argnums=0))(x1, x2), jit(grad(f, argnums=1))(x1, x2))
    
    # Approach 2: Single jit call
    g2 = jit(lambda x1, x2: (f(x1, x2), grad(f, argnums=0)(x1, x2), grad(f, argnums=1)(x1, x2)))
    
    # Warm up
    _ = g1(x1, x2)
    _ = g2(x1, x2)
    
    print("\nPerformance comparison:")
    
    # Benchmark parameters
    number = 300  # Loops per run
    repeat = 3    # Number of runs
    
    # CPU benchmarks
    print("\nCPU Performance:")
    jax.config.update('jax_platform_name', 'cpu')
    
    # Time g1 (multiple jit)
    times_g1_cpu = timeit.repeat(lambda: g1(x1, x2), repeat=repeat, number=number)
    avg_time_g1_cpu = np.mean(times_g1_cpu) / number * 1e6  # µs
    std_time_g1_cpu = np.std(times_g1_cpu) / number * 1e6
    
    # Time g2 (single jit)
    times_g2_cpu = timeit.repeat(lambda: g2(x1, x2), repeat=repeat, number=number)
    avg_time_g2_cpu = np.mean(times_g2_cpu) / number * 1e6
    std_time_g2_cpu = np.std(times_g2_cpu) / number * 1e6
    
    print(f"g1 (multiple jit): {avg_time_g1_cpu:.2f} µs ± {std_time_g1_cpu:.2f} µs per loop ({repeat} runs, {number} loops each)")
    print(f"g2 (single jit): {avg_time_g2_cpu:.2f} µs ± {std_time_g2_cpu:.2f} µs per loop ({repeat} runs, {number} loops each)")
    print(f"Speedup of g2 over g1: {avg_time_g1_cpu/avg_time_g2_cpu:.2f}x")
    
    # Try GPU if available
    try:
        gpu_devices = jax.devices('gpu')
        if len(gpu_devices) > 0:
            print("\nGPU Performance:")
            jax.config.update('jax_platform_name', 'gpu')
            
            # Warm up on GPU
            _ = g1(x1, x2)
            _ = g2(x1, x2)
            
            # GPU benchmark settings
            gpu_number = 500
            gpu_repeat = 3
            
            # Time g1 on GPU
            times_g1_gpu = timeit.repeat(lambda: g1(x1, x2), repeat=gpu_repeat, number=gpu_number)
            avg_time_g1_gpu = np.mean(times_g1_gpu) / gpu_number * 1e6
            std_time_g1_gpu = np.std(times_g1_gpu) / gpu_number * 1e6
            
            # Time g2 on GPU
            times_g2_gpu = timeit.repeat(lambda: g2(x1, x2), repeat=gpu_repeat, number=gpu_number)
            avg_time_g2_gpu = np.mean(times_g2_gpu) / gpu_number * 1e6
            std_time_g2_gpu = np.std(times_g2_gpu) / gpu_number * 1e6
            
            print(f"g1 (multiple jit): {avg_time_g1_gpu:.2f} µs ± {std_time_g1_gpu:.2f} µs per loop ({gpu_repeat} runs, {gpu_number} loops each)")
            print(f"g2 (single jit): {avg_time_g2_gpu:.2f} µs ± {std_time_g2_gpu:.2f} µs per loop ({gpu_repeat} runs, {gpu_number} loops each)")
            print(f"Speedup of g2 over g1 on GPU: {avg_time_g1_gpu/avg_time_g2_gpu:.2f}x")
            
            # CPU vs GPU comparison
            print("\nCPU vs GPU Speedup:")
            print(f"g1 CPU vs GPU: {avg_time_g1_cpu/avg_time_g1_gpu:.2f}x")
            print(f"g2 CPU vs GPU: {avg_time_g2_cpu/avg_time_g2_gpu:.2f}x")
        else:
            print("\nNo GPU device found, skipping GPU performance testing.")
    except Exception as e:
        print(f"\nError during GPU testing: {e}")
        print("GPU performance testing skipped.")
    
    # Back to CPU
    jax.config.update('jax_platform_name', 'cpu')

# 6. Vector input handling with vmap
def vector_inputs():
    """Use vmap to handle vectorized inputs"""
    # Our function with gradients
    g2 = jit(lambda x1, x2: (f(x1, x2), grad(f, argnums=0)(x1, x2), grad(f, argnums=1)(x1, x2)))
    
    # Generate test data
    x1s = jnp.linspace(1.0, 10.0, 1000)
    x2s = x1s + 1.0
    
    # a) Vectorize across both inputs
    vmap_both = vmap(g2, in_axes=(0, 0))
    jaxpr_both = make_jaxpr(vmap_both)(x1s, x2s)
    
    print("\nJAXPR for vmap across both parameters (truncated):")
    jaxpr_both_str = str(jaxpr_both)
    print(jaxpr_both_str[:500] + "...\n[output truncated]")
    
    # Save full jaxpr
    with open("jaxpr_vmap_both.txt", "w") as file:
        file.write(jaxpr_both_str)
    
    # b) Vectorize across first input only
    vmap_first = vmap(g2, in_axes=(0, None))
    jaxpr_first = make_jaxpr(vmap_first)(x1s, 0.5)
    
    print("\nJAXPR for vmap across only the first parameter (truncated):")
    jaxpr_first_str = str(jaxpr_first)
    print(jaxpr_first_str[:500] + "...\n[output truncated]")
    
    # Save full jaxpr
    with open("jaxpr_vmap_first.txt", "w") as file:
        file.write(jaxpr_first_str)
    
    print("\nFull jaxpr saved to part2/jaxpr_vmap_both.txt and part2/jaxpr_vmap_first.txt")
    
    # Test with small sample
    print("\nTesting vectorized functions:")
    small_x1s = jnp.array([1.0, 2.0, 3.0])
    small_x2s = jnp.array([2.0, 3.0, 4.0])
    
    # Test both-vector case
    results_both = vmap_both(small_x1s, small_x2s)
    print(f"\nResults for vmap across both parameters (sample of 3):")
    print(f"Function values: {results_both[0]}")
    print(f"Gradients wrt x1: {results_both[1]}")
    print(f"Gradients wrt x2: {results_both[2]}")
    
    # Test first-vector-only case
    x2_scalar = 0.5
    results_first = vmap_first(small_x1s, x2_scalar)
    print(f"\nResults for vmap across only the first parameter (sample of 3, x2={x2_scalar}):")
    print(f"Function values: {results_first[0]}")
    print(f"Gradients wrt x1: {results_first[1]}")
    print(f"Gradients wrt x2: {results_first[2]}")


if __name__ == "__main__":
    print("JAX Reverse-Mode Automatic Differentiation")
    print("==========================================")
    
    # 1. Compute gradients using jax.grad
    print("\n1. Computing gradients using jax.grad:")
    dy_dx1, dy_dx2 = compute_gradients_positional()
    
    # 2. Print jaxpr code
    print("\n2. JAX generated jaxpr code for gradients:")
    print_jaxpr_code()
    
    # 3. Classical reverse-mode AD comparison
    print("\n3. Classical reverse-mode AD comparison:")
    x1_val = 2.0
    x2_val = 5.0
    classical_y, classical_dy_dx1, classical_dy_dx2 = classical_reverse_ad(x1_val, x2_val)
    
    print(f"Classical implementation at x1={x1_val}, x2={x2_val}:")
    print(f"Function value: {classical_y}")
    print(f"Gradient with respect to x1: {classical_dy_dx1}")
    print(f"Gradient with respect to x2: {classical_dy_dx2}")
    
    # Compare with JAX results
    jax_y = f(x1_val, x2_val)
    jax_dy_dx1 = dy_dx1(x1_val, x2_val)
    jax_dy_dx2 = dy_dx2(x1_val, x2_val)
    
    print("\nComparison with JAX implementation:")
    print(f"Function value: JAX={jax_y}, Classical={classical_y}, Difference={jax_y-classical_y}")
    print(f"Gradient wrt x1: JAX={jax_dy_dx1}, Classical={classical_dy_dx1}, Difference={jax_dy_dx1-classical_dy_dx1}")
    print(f"Gradient wrt x2: JAX={jax_dy_dx2}, Classical={classical_dy_dx2}, Difference={jax_dy_dx2-classical_dy_dx2}")
    
    # 4. Compile to HLO
    print("\n4. Compiling to HLO representation:")
    compile_to_hlo()
    
    # 5. Compare runtime
    print("\n5. Comparing runtime of different JIT approaches:")
    compare_runtime()
    
    # 6. Vector inputs using vmap
    print("\n6. Extending for vector inputs using vmap:")
    vector_inputs() 