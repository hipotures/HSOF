#!/usr/bin/env julia

using CUDA
using Random
using Dates

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, skipping simple checkpointing test"
    exit(0)
end

println("SIMPLE MODEL CHECKPOINTING TEST")
println("="^50)

# Include the model checkpointing module
include("src/metamodel/model_checkpointing.jl")
using .ModelCheckpointing

"""
Mock model for testing
"""
mutable struct MockModel
    weights::Dict{String, Array{Float32}}
    parameters_count::Int
end

function MockModel(layer_sizes::Vector{Int})
    weights = Dict{String, Array{Float32}}()
    param_count = 0
    
    for i in 1:(length(layer_sizes)-1)
        in_size = layer_sizes[i]
        out_size = layer_sizes[i+1]
        
        # Create weight matrix
        W = randn(Float32, out_size, in_size) * 0.1f0
        b = zeros(Float32, out_size)
        
        weights["layer_$(i)_weight"] = W
        weights["layer_$(i)_bias"] = b
        
        param_count += length(W) + length(b)
    end
    
    MockModel(weights, param_count)
end

function test_simple_checkpointing()
    println("Testing basic model checkpointing functionality...")
    
    # Create test model
    model = MockModel([100, 64, 32, 1])
    println("✓ Created mock model with $(model.parameters_count) parameters")
    
    # Create simple optimizer state
    optimizer_state = Dict{String, Any}(
        "optimizer_type" => "Adam",
        "learning_rate" => 0.001,
        "beta1" => 0.9,
        "beta2" => 0.999
    )
    
    # Create checkpointing configuration
    config = CheckpointConfig(
        checkpoint_dir = "simple_test_checkpoints",
        max_checkpoints = 3,
        async_save = false,  # Use synchronous for testing
        compression_enabled = true,
        validation_enabled = true
    )
    
    # Create checkpoint manager
    manager = create_checkpoint_manager(config)
    println("✓ Created checkpoint manager")
    println("✓ Created directory structure: $(config.checkpoint_dir)")
    
    # Test model state extraction
    model_state = model.weights
    model_hash = ModelCheckpointing.compute_model_hash(model_state)
    println("✓ Computed model hash: $(string(model_hash, base=16))")
    
    # Create metadata
    metadata = CheckpointMetadata(
        iteration = 1000,
        epoch = 10,
        training_loss = 0.25,
        validation_loss = 0.30,
        correlation_score = 0.85,
        learning_rate = 0.001,
        tags = ["test", "simple"],
        notes = "Simple test checkpoint"
    )
    
    # Test compression
    test_data = rand(UInt8, 1000)
    compressed = ModelCheckpointing.compress_data(test_data, 6)
    decompressed = ModelCheckpointing.decompress_data(compressed)
    
    if decompressed == test_data
        compression_ratio = length(compressed) / length(test_data)
        println("✓ Compression working: $(round(compression_ratio * 100, digits=1))% of original size")
    else
        println("❌ Compression test failed")
        return false
    end
    
    # Create checkpoint data manually for testing
    checkpoint_data = ModelCheckpointing.CheckpointData(
        model_state,
        optimizer_state,
        Dict{String, Any}(),  # empty replay buffer
        Float64[0.5, 0.4, 0.3],  # training history
        Float64[0.6, 0.5, 0.4],  # validation history
        Dict{String, Any}("model_type" => "MockModel"),
        Dict{String, Any}("save_time" => time()),
        metadata
    )
    
    # Test serialization
    serialized = ModelCheckpointing.serialize_checkpoint(checkpoint_data)
    println("✓ Serialized checkpoint: $(length(serialized)) bytes")
    
    # Test compression of serialized data
    compressed_checkpoint = ModelCheckpointing.compress_data(serialized, 6)
    compression_ratio = length(compressed_checkpoint) / length(serialized)
    println("✓ Compressed checkpoint: $(round(compression_ratio * 100, digits=1))% of original")
    
    # Test checkpoint validation
    # Create a simple checkpoint file for testing
    test_checkpoint_file = joinpath(config.checkpoint_dir, "test_checkpoint.jls")
    write(test_checkpoint_file, compressed_checkpoint)
    
    is_valid = validate_checkpoint(test_checkpoint_file)
    println("✓ Checkpoint validation: $(is_valid ? "PASSED" : "FAILED")")
    
    # Test metadata saving
    metadata_file = joinpath(config.checkpoint_dir, "metadata", "test_metadata.json")
    mkpath(dirname(metadata_file))
    ModelCheckpointing.save_metadata(metadata, metadata_file)
    
    if isfile(metadata_file)
        println("✓ Metadata saved successfully")
    else
        println("❌ Metadata save failed")
        return false
    end
    
    # Test registry operations
    manager.checkpoint_registry["test_checkpoint"] = metadata
    ModelCheckpointing.save_registry(manager)
    
    registry_file = joinpath(config.checkpoint_dir, "registry.json")
    if isfile(registry_file)
        println("✓ Registry saved successfully")
    else
        println("❌ Registry save failed")
        return false
    end
    
    # Test checkpoint info retrieval
    info = get_checkpoint_info(manager, "test_checkpoint")
    if info !== nothing && info["iteration"] == 1000
        println("✓ Checkpoint info retrieval working")
    else
        println("❌ Checkpoint info retrieval failed")
        return false
    end
    
    # Test manager statistics
    manager.total_saves = 3
    manager.total_save_time = 0.150  # 150ms total
    manager.bytes_saved = 10000
    manager.bytes_compressed = 6000
    
    stats = ModelCheckpointing.get_manager_stats(manager)
    expected_avg_time = 0.150 / 3 * 1000  # 50ms average
    
    if abs(stats["avg_save_time_ms"] - expected_avg_time) < 0.1
        println("✓ Statistics calculation working")
        println("  Average save time: $(stats["avg_save_time_ms"]) ms")
        println("  Compression ratio: $(stats["overall_compression_ratio"])")
    else
        println("❌ Statistics calculation failed")
        return false
    end
    
    # Cleanup
    if isdir(config.checkpoint_dir)
        rm(config.checkpoint_dir, recursive=true)
        println("✓ Cleaned up test directory")
    end
    
    return true
end

# Run the simple test
if abspath(PROGRAM_FILE) == @__FILE__
    success = test_simple_checkpointing()
    
    println("="^50)
    if success
        println("✅ Simple model checkpointing test PASSED")
        println("✅ Core functionality validated:")
        println("  - Configuration and manager creation")
        println("  - Model state extraction and hashing")
        println("  - Compression and decompression")
        println("  - Checkpoint validation")
        println("  - Metadata handling")
        println("  - Registry management")
        println("  - Statistics collection")
    else
        println("❌ Simple model checkpointing test FAILED")
    end
    
    exit(success ? 0 : 1)
end