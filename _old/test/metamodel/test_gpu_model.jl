using Flux
using CUDA

include("../../src/metamodel/neural_architecture.jl")
using .NeuralArchitecture

# Create a simple test
println("Testing GPU model creation...")

# Create model
config = create_metamodel_config(input_dim=10, hidden_dims=[8, 4, 2])
model = create_metamodel(config)

# Check initial state
println("Initial weight type: ", typeof(model.input_layer.weight))

# Move to GPU
gpu_model = model |> gpu

# Check GPU state
println("GPU weight type: ", typeof(gpu_model.input_layer.weight))

# Test with create_gpu_metamodel
gpu_model2 = create_gpu_metamodel(config)
println("create_gpu_metamodel weight type: ", typeof(gpu_model2.input_layer.weight))

# Test forward pass
test_input = CUDA.rand(Float32, 10, 2)
output = gpu_model2(test_input)
println("Output shape: ", size(output))
println("Output type: ", typeof(output))