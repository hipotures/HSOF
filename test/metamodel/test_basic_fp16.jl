using Test
using CUDA
using Flux

# Include modules
include("../../src/metamodel/neural_architecture.jl")
include("../../src/metamodel/fp16_optimization.jl")

using .NeuralArchitecture
using .FP16Optimization

println("Testing Basic FP16 Functionality...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Test 1: Create FP32 model
config32 = create_metamodel_config(input_dim=32, hidden_dims=[16, 8, 4])
model32 = create_gpu_metamodel(config32)
println("✓ FP32 model created")
println("  Weight type: $(typeof(model32.input_layer.weight))")

# Test 2: Create FP16 model
config16 = create_metamodel_config(input_dim=32, hidden_dims=[16, 8, 4], use_fp16=true)
model16 = create_gpu_metamodel(config16)
model16 = NeuralArchitecture.to_fp16(model16)
println("✓ FP16 model created")
println("  Weight type: $(typeof(model16.input_layer.weight))")

# Test 3: Forward passes
x32 = CUDA.rand(Float32, 32, 8)
x16 = Float16.(x32)

y32 = model32(x32)
y16 = model16(x16)

println("✓ Forward passes completed")
println("  FP32 output: $(size(y32)), type: $(eltype(y32))")
println("  FP16 output: $(size(y16)), type: $(eltype(y16))")

# Test 4: Compare outputs
y16_as_32 = Float32.(y16)
diff = maximum(abs.(y32 .- y16_as_32))
println("✓ Max difference between FP32 and FP16: $diff")

# Test 5: Performance comparison
println("\nPerformance test (100 forward passes):")

# FP32 timing
CUDA.synchronize()
t32 = @elapsed for _ in 1:100
    model32(x32)
end
CUDA.synchronize()

# FP16 timing
CUDA.synchronize()
t16 = @elapsed for _ in 1:100
    model16(x16)
end
CUDA.synchronize()

println("  FP32 time: $(round(t32*1000, digits=2))ms")
println("  FP16 time: $(round(t16*1000, digits=2))ms")
println("  Speedup: $(round(t32/t16, digits=2))x")

# Test 6: FP16 optimization features
fp16_config = create_fp16_config()
println("\n✓ FP16 optimization config:")
println("  Use Tensor Cores: $(fp16_config.use_tensor_cores)")
println("  Initial loss scale: $(fp16_config.initial_loss_scale)")

# Test 7: Dynamic loss scaler
scaler = create_loss_scaler(fp16_config)
println("✓ Dynamic loss scaler created with scale: $(scaler.scale)")

# Test 8: Memory optimization
test_arr = CUDA.rand(Float16, 30, 30)
aligned = optimize_memory_layout(test_arr)
println("✓ Memory alignment: $(size(test_arr)) → $(size(aligned))")

println("\n✅ Basic FP16 tests completed!")