using Test
using CUDA
using Flux

# Include the neural architecture module
include("../../src/metamodel/neural_architecture.jl")

using .NeuralArchitecture

println("Testing Neural Architecture Basic Functionality...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Test 1: Create model
config = create_metamodel_config()
model = create_metamodel(config)
println("✓ Model created successfully")

# Test 2: CPU forward pass
x_cpu = randn(Float32, 500, 8)
y_cpu = model(x_cpu)
println("✓ CPU forward pass: input $(size(x_cpu)) -> output $(size(y_cpu))")

# Test 3: GPU model
gpu_model = create_gpu_metamodel(config)
println("✓ GPU model created")

# Test 4: GPU forward pass
x_gpu = CUDA.randn(Float32, 500, 8)
y_gpu = gpu_model(x_gpu)
println("✓ GPU forward pass: input $(size(x_gpu)) -> output $(size(y_gpu))")

# Test 5: Parameter count
n_params = count_parameters(model)
println("✓ Model has $n_params parameters")

# Test 6: Memory estimation
mem_info = estimate_memory_usage(model)
println("✓ Estimated memory usage: $(round(mem_info.total_memory_mb, digits=2)) MB")

println("\n✅ All basic tests passed!")