using Test
using CUDA
using Flux
using Statistics

# Include modules
include("../../src/metamodel/neural_architecture.jl")
include("../../src/metamodel/fp16_optimization.jl")

using .NeuralArchitecture
using .FP16Optimization

println("Testing FP16 Optimization...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Test 1: Configuration
config = create_fp16_config()
println("✓ FP16 configuration created")
@test config.use_tensor_cores == true
@test config.initial_loss_scale == 2f0^16

# Test 2: Create models
model_config = create_metamodel_config(input_dim=64, hidden_dims=[32, 16, 8])
fp32_model = create_gpu_metamodel(model_config)
println("✓ FP32 model created")

# Test 3: Convert to FP16
# First create a model configured for FP16
fp16_config = create_metamodel_config(input_dim=64, hidden_dims=[32, 16, 8], use_fp16=true)
fp16_model = create_gpu_metamodel(fp16_config)
fp16_model = NeuralArchitecture.to_fp16(fp16_model)
println("✓ Model converted to FP16")
@test fp16_model.config.use_fp16 == true
@test eltype(fp16_model.input_layer.weight) == Float16

# Test 4: Test forward pass
test_input_fp32 = CUDA.rand(Float32, 64, 4)
test_input_fp16 = Float16.(test_input_fp32)

# FP32 forward
output_fp32 = fp32_model(test_input_fp32)
println("✓ FP32 forward pass: output shape = $(size(output_fp32))")

# FP16 forward
output_fp16 = fp16_model(test_input_fp16)
println("✓ FP16 forward pass: output shape = $(size(output_fp16))")

# Test 5: Dynamic loss scaler
scaler = create_loss_scaler(config)
println("✓ Dynamic loss scaler created")
@test scaler.scale == config.initial_loss_scale

# Test no infinity
update_loss_scaler!(scaler, false)
@test scaler.growth_tracker == 1

# Test infinity found
update_loss_scaler!(scaler, true)
@test scaler.scale < config.initial_loss_scale
@test scaler.growth_tracker == 0

# Test 6: Memory alignment
test_array = CUDA.rand(Float16, 30, 30)
aligned_array = optimize_memory_layout(test_array, 128)
println("✓ Memory layout optimized: $(size(test_array)) → $(size(aligned_array))")
@test size(aligned_array, 1) % 64 == 0  # 128 bytes / 2 bytes per Float16 = 64 elements
@test size(aligned_array, 2) % 64 == 0

# Test 7: Accuracy validation
validation_result = validate_fp16_accuracy(
    fp32_model,
    fp16_model,
    test_input_fp32,
    0.01f0
)
println("✓ FP16 accuracy validation:")
println("  - Max difference: $(validation_result.max_diff)")
println("  - Mean difference: $(validation_result.mean_diff)")
println("  - Passed: $(validation_result.passed)")

# Test 8: Performance profiling
println("\nProfiling FP16 vs FP32 performance:")
batch_sizes = [16, 32, 64, 128]

# Profile FP32
println("\nFP32 Performance:")
fp32_results = profile_fp16_performance(fp32_model, batch_sizes, config)
for result in fp32_results
    println("  Batch $(result.batch_size): $(round(result.avg_time_ms, digits=3))ms, " *
            "$(round(result.samples_per_ms, digits=1)) samples/ms")
end

# Profile FP16
println("\nFP16 Performance:")
fp16_results = profile_fp16_performance(fp16_model, batch_sizes, config)
for result in fp16_results
    println("  Batch $(result.batch_size): $(round(result.avg_time_ms, digits=3))ms, " *
            "$(round(result.samples_per_ms, digits=1)) samples/ms")
end

# Calculate speedup
println("\nSpeedup (FP16 vs FP32):")
for i in 1:length(batch_sizes)
    speedup = fp32_results[i].avg_time_ms / fp16_results[i].avg_time_ms
    println("  Batch $(batch_sizes[i]): $(round(speedup, digits=2))x")
end

# Test 9: Gradient scaling
test_grads = (
    w1 = CUDA.rand(Float16, 10, 10),
    w2 = CUDA.rand(Float16, 5, 5),
    b1 = nothing,
    b2 = CUDA.rand(Float16, 5)
)

original_w1 = copy(test_grads.w1)
scale_gradients!(test_grads, 100.0f0)
@test all(test_grads.w1 .≈ original_w1 .* 100.0f0)
println("✓ Gradient scaling works")

# Test 10: Check for inf/nan
clean_grads = (w = CUDA.rand(Float16, 5, 5),)
@test !has_inf_or_nan(clean_grads)

inf_grads = (w = CUDA.fill(Float16(Inf), 5, 5),)
@test has_inf_or_nan(inf_grads)
println("✓ Inf/NaN detection works")

println("\n✅ All FP16 optimization tests passed!")