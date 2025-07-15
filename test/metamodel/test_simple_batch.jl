using Test
using CUDA

# Include the experience replay module
include("../../src/metamodel/experience_replay.jl")

using .ExperienceReplay

println("Testing simple batch insertion...")

# Skip tests if no GPU available
if !CUDA.functional()
    @warn "CUDA not functional, skipping GPU tests"
    exit(0)
end

# Create a small buffer
config = create_replay_config(buffer_size=10, max_features=5)
buffer = create_replay_buffer(config)

# Create very simple batch data
batch_size = 2
features_batch = Int32[1 2;
                      3 4;
                      0 0;
                      0 0;
                      0 0]
predicted_batch = Float32[0.5, 0.6]
actual_batch = Float32[0.5, 0.6]

println("Batch info:")
println("  Size: ", size(features_batch))
println("  Batch size: ", batch_size)
println("  Buffer size: ", buffer.config.buffer_size)
println("  Max features: ", buffer.config.max_features)

# Insert batch
try
    batch_insert!(buffer, features_batch, predicted_batch, actual_batch)
    println("\nBatch insertion successful!")
    
    # Check results
    stats = get_buffer_stats(buffer)
    println("Buffer stats: n_valid=$(stats.n_valid), total_count=$(stats.total_count)")
catch e
    println("\nError during batch insertion: ", e)
    rethrow(e)
end