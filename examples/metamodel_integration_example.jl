"""
Example of using the Metamodel Integration module for neural network inference
in the MCTS GPU engine.
"""

using CUDA
using Statistics

# Include the metamodel integration module
include("../src/gpu/kernels/metamodel_integration.jl")

using .MetamodelIntegration

# Example neural network metamodel function
# In practice, this would call your actual neural network
function my_neural_network(features::CuArray{Float32, 2}, batch_size::Int32)
    # Simulate neural network inference
    # Input: features matrix of size (feature_dim, batch_size)
    # Output: scores vector of size (batch_size,)
    
    # Example: simple linear transformation + sigmoid
    weights = CUDA.randn(Float32, 1, size(features, 1)) * 0.1f0
    biases = CUDA.fill(0.5f0, 1)
    
    # Compute scores
    scores = weights * features .+ biases
    scores = 1.0f0 ./ (1.0f0 .+ exp.(-scores))
    
    return vec(scores)  # Return as 1D vector
end

# Configure the metamodel integration
config = MetamodelConfig(
    Int32(32),      # batch_size - process up to 32 nodes at once
    Int32(100),     # feature_dim - 100 input features
    Int32(1),       # output_dim - single score output
    Int32(1000),    # max_queue_size - queue up to 1000 requests
    10.0f0,         # timeout_ms - wait max 10ms before processing partial batch
    Int32(512),     # cache_size - cache 512 recent evaluations
    0.5f0,          # fallback_score - use 0.5 if metamodel fails
    false           # use_mixed_precision - use FP32 (not FP16)
)

# Create the metamodel manager
manager = MetamodelManager(config)

println("Metamodel manager created with config:")
println("  Batch size: $(config.batch_size)")
println("  Feature dimension: $(config.feature_dim)")
println("  Max queue size: $(config.max_queue_size)")
println("  Cache size: $(config.cache_size)")
println()

# Simulate MCTS expansion requesting metamodel evaluations
println("Enqueueing evaluation requests...")
node_indices = Int32[]
for i in 1:50
    node_idx = Int32(i * 10)  # Example node indices
    push!(node_indices, node_idx)
    
    # Enqueue with priority (higher priority = more urgent)
    priority = i <= 10 ? Int32(5) : Int32(1)
    request_id = enqueue_evaluation!(manager, node_idx, priority)
    
    if i % 10 == 0
        println("  Enqueued $i requests...")
    end
end

println("\nProcessing batches...")
total_processed = 0
batch_count = 0

# Process all queued requests
while total_processed < length(node_indices)
    # Small delay to simulate other work
    sleep(0.01)
    
    # Process a batch
    processed = process_batch!(manager, my_neural_network)
    
    if processed > 0
        batch_count += 1
        total_processed += processed
        println("  Batch $batch_count: processed $processed nodes (total: $total_processed)")
        
        # Check results for the nodes we care about
        results = check_results(manager, node_indices)
        
        if !isempty(results)
            # Show some results
            sample_nodes = collect(keys(results))[1:min(3, length(results))]
            for node in sample_nodes
                println("    Node $node: score = $(round(results[node], digits=3))")
            end
        end
    end
end

# Get final statistics
println("\nFinal statistics:")
stats = get_eval_statistics(manager)
for (key, value) in sort(collect(stats))
    if value isa Number
        println("  $key: $(round(value, digits=2))")
    else
        println("  $key: $value")
    end
end

# Example of using with caching
println("\nTesting cache performance...")

# Re-evaluate some nodes - should hit cache
cache_test_nodes = node_indices[1:10]
for node_idx in cache_test_nodes
    enqueue_evaluation!(manager, node_idx)
end

# Process - should be faster due to cache hits
sleep(0.02)
process_batch!(manager, my_neural_network)

# Check cache statistics
final_stats = get_eval_statistics(manager)
println("\nCache performance:")
println("  Cache hits: $(final_stats["cache_hits"])")
println("  Cache misses: $(final_stats["cache_misses"])")
println("  Cache hit rate: $(round(final_stats["cache_hit_rate"] * 100, digits=1))%")

println("\n✅ Metamodel integration example completed!")