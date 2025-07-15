#!/usr/bin/env julia

# Dataset Storage Demo
# Demonstrates efficient dataset replication across multiple GPUs

using CUDA
using Printf
using Random
using DataFrames

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.DatasetStorage

function demo_dataset_storage()
    println("Dataset Storage Demo")
    println("=" ^ 60)
    
    # Create dataset manager
    num_gpus = CUDA.functional() ? min(2, length(CUDA.devices())) : 2
    manager = create_dataset_manager(
        num_gpus = num_gpus,
        memory_limit_per_gpu = 8 * 1024^3,  # 8GB per GPU
        enable_compression = false
    )
    
    println("\nDataset Manager Configuration:")
    println("  Number of GPUs: $num_gpus")
    println("  Memory limit per GPU: $(manager.memory_limit_per_gpu ÷ 1024^3) GB")
    println("  Compression: $(manager.enable_compression)")
    
    # Demo 1: Create and Replicate Dataset
    println("\n1. Dataset Creation and Replication:")
    println("-" ^ 40)
    
    # Generate synthetic dataset
    n_samples = 10000
    n_features = 100
    Random.seed!(42)
    
    println("Generating dataset: $n_samples samples × $n_features features")
    data = rand(Float32, n_samples, n_features)
    feature_names = ["feature_$i" for i in 1:n_features]
    
    # Add some patterns to make it interesting
    for i in 1:10
        data[:, i] = sin.(1:n_samples) .* rand() .+ randn(n_samples) .* 0.1
    end
    
    # Check memory requirements
    data_size_mb = sizeof(data) / 1024^2
    println("Dataset size: $(round(data_size_mb, digits=2)) MB")
    
    # Check if it fits
    for gpu_id in 0:(num_gpus-1)
        can_fit = has_sufficient_memory(manager, gpu_id, n_features, n_samples)
        println("  GPU $gpu_id: $(can_fit ? "✓ Sufficient memory" : "✗ Insufficient memory")")
    end
    
    # Replicate dataset
    println("\nReplicating dataset to all GPUs...")
    start_time = time()
    success_count = replicate_dataset!(manager, data, feature_names)
    replication_time = time() - start_time
    
    println("Replication completed in $(round(replication_time, digits=3))s")
    println("Successfully replicated to $success_count GPU(s)")
    
    # Demo 2: Memory Usage
    println("\n2. Memory Usage Information:")
    println("-" ^ 40)
    
    info = get_dataset_info(manager)
    println("Dataset version: $(info["version"])")
    
    for (gpu_name, gpu_info) in info["gpus"]
        println("\n$gpu_name:")
        println("  Loaded: $(gpu_info["loaded"])")
        if gpu_info["loaded"]
            println("  Samples: $(gpu_info["n_samples"])")
            println("  Features: $(gpu_info["n_features"])")
            println("  Memory used: $(gpu_info["memory_mb"]) MB")
        end
        
        if haskey(gpu_info, "total_memory_gb")
            println("  Total GPU memory: $(gpu_info["total_memory_gb"]) GB")
            println("  Free GPU memory: $(gpu_info["free_memory_gb"]) GB")
            println("  Utilization: $(gpu_info["utilization"])%")
        end
    end
    
    # Demo 3: Feature Access
    println("\n3. Feature Column Access:")
    println("-" ^ 40)
    
    if success_count > 0
        # Get replica from first GPU
        replica = get_dataset_replica(manager, 0)
        
        if replica.is_loaded
            # Access specific features
            feature_to_access = "feature_5"
            println("Accessing column '$feature_to_access' from GPU 0...")
            
            col_data = get_feature_column(replica, feature_to_access, as_host=true)
            println("  Column shape: $(size(col_data))")
            println("  First 5 values: $(col_data[1:5])")
            println("  Mean: $(round(mean(col_data), digits=4))")
            println("  Std: $(round(std(col_data), digits=4))")
            
            # Access by index
            col_data2 = get_feature_column(replica, 5, as_host=true)
            println("  Verified: access by name == access by index: $(col_data ≈ col_data2)")
        end
    end
    
    # Demo 4: Batch Access
    println("\n4. Sample Batch Access:")
    println("-" ^ 40)
    
    if success_count > 0 && get_dataset_replica(manager, 0).is_loaded
        replica = get_dataset_replica(manager, 0)
        
        # Get random batch
        batch_size = 32
        sample_indices = randperm(n_samples)[1:batch_size]
        feature_indices = [1, 5, 10, 20, 50]
        
        println("Accessing batch: $batch_size samples, $(length(feature_indices)) features")
        batch_data = get_sample_batch(
            replica, 
            sample_indices,
            feature_indices = feature_indices,
            as_host = true
        )
        
        println("  Batch shape: $(size(batch_data))")
        println("  Batch mean: $(round(mean(batch_data), digits=4))")
        
        # Verify correctness
        expected = data[sample_indices, feature_indices]
        println("  Correctness check: $(batch_data ≈ expected ? "✓ Passed" : "✗ Failed")")
    end
    
    # Demo 5: Dataset Updates
    println("\n5. Dataset Updates:")
    println("-" ^ 40)
    
    if success_count > 0 && get_dataset_replica(manager, 0).is_loaded
        # Update a feature on GPU 0
        feature_to_update = "feature_10"
        new_values = Float32.(cos.(1:n_samples) .* 2)
        
        println("Updating '$feature_to_update' on GPU 0...")
        old_version = manager.current_version.version_id
        
        update_dataset!(manager, 0, Dict(feature_to_update => new_values))
        
        new_version = manager.current_version.version_id
        println("  Version updated: $old_version → $new_version")
        
        # Verify update
        updated_col = get_feature_column(
            get_dataset_replica(manager, 0), 
            feature_to_update, 
            as_host = true
        )
        println("  Update verified: $(updated_col ≈ new_values ? "✓ Success" : "✗ Failed")")
    end
    
    # Demo 6: Multi-GPU Synchronization
    println("\n6. Multi-GPU Synchronization:")
    println("-" ^ 40)
    
    if num_gpus > 1 && success_count > 1
        println("GPUs before sync:")
        for gpu_id in 0:(num_gpus-1)
            replica = manager.replicas[gpu_id]
            if replica.is_loaded
                println("  GPU $gpu_id: version $(replica.version.version_id)")
            end
        end
        
        # Force sync
        println("\nSynchronizing datasets...")
        sync_datasets!(manager, force=true)
        
        println("\nGPUs after sync:")
        for gpu_id in 0:(num_gpus-1)
            replica = manager.replicas[gpu_id]
            if replica.is_loaded
                println("  GPU $gpu_id: version $(replica.version.version_id)")
            end
        end
    else
        println("  Skipped: Requires multiple GPUs")
    end
    
    # Demo 7: Memory Cleanup
    println("\n7. Memory Cleanup:")
    println("-" ^ 40)
    
    if success_count > 0
        # Get memory before cleanup
        initial_info = get_dataset_info(manager)
        
        # Clear GPU 0
        println("Clearing dataset from GPU 0...")
        clear_dataset!(manager, 0)
        
        # Check memory after cleanup
        final_info = get_dataset_info(manager)
        
        println("GPU 0 status:")
        println("  Before: loaded = $(initial_info["gpus"]["GPU0"]["loaded"]), " *
                "memory = $(initial_info["gpus"]["GPU0"]["memory_mb"]) MB")
        println("  After: loaded = $(final_info["gpus"]["GPU0"]["loaded"]), " *
                "memory = $(final_info["gpus"]["GPU0"]["memory_mb"]) MB")
    end
    
    # Demo 8: Performance Considerations
    println("\n8. Performance Tips:")
    println("-" ^ 40)
    println("• Column-wise storage optimizes GPU memory access patterns")
    println("• Replication eliminates PCIe transfers during computation")
    println("• Version tracking ensures consistency across GPUs")
    println("• Memory monitoring prevents OOM errors")
    println("• Batch access minimizes kernel launch overhead")
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

# Helper to calculate statistics
using Statistics

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_dataset_storage()
end