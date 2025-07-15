using Test
using CUDA
using DataFrames
using Random

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.DatasetStorage

@testset "Dataset Storage Tests" begin
    
    @testset "DatasetManager Creation" begin
        # Test with auto-detection
        manager = create_dataset_manager()
        @test isa(manager, DatasetManager)
        @test manager.memory_limit_per_gpu == 8 * 1024^3  # 8GB default
        @test !manager.enable_compression
        
        # Test with explicit parameters
        manager2 = create_dataset_manager(
            num_gpus = 2,
            memory_limit_per_gpu = 4 * 1024^3,
            enable_compression = true
        )
        @test length(manager2.replicas) == 2
        @test manager2.memory_limit_per_gpu == 4 * 1024^3
        @test manager2.enable_compression
        
        # Check replica initialization
        for gpu_id in 0:1
            @test haskey(manager2.replicas, gpu_id)
            replica = manager2.replicas[gpu_id]
            @test replica.gpu_id == gpu_id
            @test !replica.is_loaded
            @test replica.allocated_memory == 0
        end
    end
    
    @testset "Memory Management" begin
        manager = create_dataset_manager(num_gpus=1, memory_limit_per_gpu=1024^3)  # 1GB
        
        # Test memory calculation
        n_features = 1000
        n_samples = 10000
        
        # Should fit (1000 * 10000 * 4 bytes = 40MB < 1GB)
        @test has_sufficient_memory(manager, 0, n_features, n_samples)
        
        # Should not fit (10000 * 100000 * 4 bytes = 4GB > 1GB)
        @test !has_sufficient_memory(manager, 0, 10000, 100000)
        
        # Test memory stats
        if CUDA.functional()
            stats = get_memory_usage(0)
            @test stats.gpu_id == 0
            @test stats.total_memory > 0
            @test stats.free_memory > 0
            @test 0.0 <= stats.utilization <= 1.0
        end
    end
    
    @testset "Dataset Loading" begin
        manager = create_dataset_manager(num_gpus=1)
        
        # Create test data
        n_samples = 1000
        n_features = 50
        Random.seed!(123)
        data = rand(Float32, n_samples, n_features)
        feature_names = ["feature_$i" for i in 1:n_features]
        sample_ids = Int32.(1:n_samples)
        
        # Get replica
        replica = manager.replicas[0]
        
        # Load data
        load_dataset_to_gpu!(replica, data, feature_names, sample_ids=sample_ids)
        
        if CUDA.functional()
            @test replica.is_loaded
            @test replica.n_samples == n_samples
            @test replica.n_features == n_features
            @test replica.allocated_memory > 0
            @test length(replica.feature_names) == n_features
            @test replica.feature_indices["feature_1"] == 1
            @test replica.feature_indices["feature_50"] == 50
        else
            @test !replica.is_loaded  # Reference only without CUDA
            @test replica.n_samples == n_samples
            @test replica.n_features == n_features
        end
    end
    
    @testset "Dataset Replication" begin
        num_gpus = CUDA.functional() ? min(2, length(CUDA.devices())) : 2
        manager = create_dataset_manager(num_gpus=num_gpus)
        
        # Create test data
        n_samples = 500
        n_features = 20
        data = rand(Float32, n_samples, n_features)
        feature_names = ["feat_$i" for i in 1:n_features]
        
        # Replicate to all GPUs
        success_count = replicate_dataset!(manager, data, feature_names)
        
        if CUDA.functional()
            @test success_count > 0
            
            # Check each replica
            for gpu_id in 0:(num_gpus-1)
                if haskey(manager.replicas, gpu_id)
                    replica = manager.replicas[gpu_id]
                    if replica.is_loaded
                        @test replica.n_samples == n_samples
                        @test replica.n_features == n_features
                        @test replica.version.version_id == manager.current_version.version_id
                    end
                end
            end
        else
            # Without CUDA, no successful replications
            @test success_count == 0
        end
    end
    
    @testset "Feature Column Access" begin
        manager = create_dataset_manager(num_gpus=1)
        
        # Create and load data
        n_samples = 100
        n_features = 10
        data = rand(Float32, n_samples, n_features)
        feature_names = ["col_$i" for i in 1:n_features]
        
        replica = manager.replicas[0]
        load_dataset_to_gpu!(replica, data, feature_names)
        
        if CUDA.functional() && replica.is_loaded
            # Test column access by name
            col_data = get_feature_column(replica, "col_5", as_host=true)
            @test length(col_data) == n_samples
            @test col_data ≈ data[:, 5]
            
            # Test column access by index
            col_data2 = get_feature_column(replica, 5, as_host=true)
            @test col_data2 ≈ col_data
            
            # Test GPU array return
            gpu_col = get_feature_column(replica, "col_1", as_host=false)
            @test isa(gpu_col, AbstractArray{Float32, 1})
            
            # Test invalid column
            @test_throws ErrorException get_feature_column(replica, "invalid_col")
            @test_throws ErrorException get_feature_column(replica, 100)
        end
    end
    
    @testset "Sample Batch Access" begin
        manager = create_dataset_manager(num_gpus=1)
        
        # Create and load data
        n_samples = 200
        n_features = 15
        data = rand(Float32, n_samples, n_features)
        feature_names = ["f$i" for i in 1:n_features]
        
        replica = manager.replicas[0]
        load_dataset_to_gpu!(replica, data, feature_names)
        
        if CUDA.functional() && replica.is_loaded
            # Test sample batch
            sample_indices = [1, 10, 20, 50, 100]
            batch = get_sample_batch(replica, sample_indices, as_host=true)
            @test size(batch) == (length(sample_indices), n_features)
            @test batch ≈ data[sample_indices, :]
            
            # Test with feature subset
            feature_indices = [2, 5, 8]
            batch2 = get_sample_batch(replica, sample_indices, 
                                    feature_indices=feature_indices, as_host=true)
            @test size(batch2) == (length(sample_indices), length(feature_indices))
            @test batch2 ≈ data[sample_indices, feature_indices]
        end
    end
    
    @testset "Dataset Updates" begin
        manager = create_dataset_manager(num_gpus=1)
        
        # Create and load data
        n_samples = 50
        n_features = 5
        data = ones(Float32, n_samples, n_features)
        feature_names = ["var_$i" for i in 1:n_features]
        
        replica = manager.replicas[0]
        load_dataset_to_gpu!(replica, data, feature_names)
        
        if CUDA.functional() && replica.is_loaded
            # Update a feature
            new_values = Float32.(2:51)  # 2.0 to 51.0
            update_dataset!(manager, 0, Dict("var_3" => new_values))
            
            # Check update
            updated_col = get_feature_column(replica, "var_3", as_host=true)
            @test updated_col ≈ new_values
            
            # Check version increment
            @test replica.version.version_id == 2
            
            # Test invalid update size
            @test_throws ErrorException update_dataset!(
                manager, 0, Dict("var_1" => Float32[1, 2, 3])
            )
        end
    end
    
    @testset "Dataset Synchronization" begin
        num_gpus = CUDA.functional() ? min(2, length(CUDA.devices())) : 2
        manager = create_dataset_manager(num_gpus=num_gpus)
        
        if num_gpus >= 2
            # Load different data on different GPUs
            data1 = ones(Float32, 100, 10)
            data2 = ones(Float32, 100, 10) * 2
            feature_names = ["f$i" for i in 1:10]
            
            # Load to GPU 0
            load_dataset_to_gpu!(manager.replicas[0], data1, feature_names)
            manager.replicas[0].version = DatasetVersion(1, 10, 100)
            
            # Load to GPU 1 with older version
            load_dataset_to_gpu!(manager.replicas[1], data2, feature_names)
            manager.replicas[1].version = DatasetVersion(0, 10, 100)
            
            if CUDA.functional()
                # Sync datasets
                sync_datasets!(manager, force=true)
                
                # Check if GPU 1 was updated
                if manager.replicas[1].is_loaded
                    @test manager.replicas[1].version.version_id == 1
                end
            end
        end
    end
    
    @testset "Dataset Info" begin
        manager = create_dataset_manager(num_gpus=1)
        
        # Get info before loading
        info = get_dataset_info(manager)
        @test haskey(info, "version")
        @test haskey(info, "gpus")
        @test haskey(info["gpus"], "GPU0")
        @test !info["gpus"]["GPU0"]["loaded"]
        
        # Load data and get info again
        data = rand(Float32, 100, 20)
        feature_names = ["f$i" for i in 1:20]
        replicate_dataset!(manager, data, feature_names)
        
        info2 = get_dataset_info(manager)
        if CUDA.functional()
            @test info2["gpus"]["GPU0"]["loaded"]
            @test info2["gpus"]["GPU0"]["n_samples"] == 100
            @test info2["gpus"]["GPU0"]["n_features"] == 20
            @test info2["gpus"]["GPU0"]["memory_mb"] > 0
        end
    end
    
    @testset "Memory Cleanup" begin
        manager = create_dataset_manager(num_gpus=1)
        
        # Load data
        data = rand(Float32, 1000, 50)
        feature_names = ["f$i" for i in 1:50]
        replicate_dataset!(manager, data, feature_names)
        
        # Clear dataset
        clear_dataset!(manager, 0)
        
        replica = manager.replicas[0]
        @test !replica.is_loaded
        @test replica.allocated_memory == 0
        @test isnothing(replica.feature_data)
        @test isnothing(replica.sample_ids)
    end
    
    @testset "Edge Cases" begin
        manager = create_dataset_manager(num_gpus=1)
        
        # Empty dataset (need to pass a matrix)
        @test_throws ErrorException replicate_dataset!(
            manager, reshape(Float32[], 0, 0), String[]
        )
        
        # Mismatched dimensions
        data = rand(Float32, 10, 5)
        wrong_names = ["f1", "f2", "f3"]  # Only 3 names for 5 features
        replica = manager.replicas[0]
        @test_throws ErrorException load_dataset_to_gpu!(
            replica, data, wrong_names
        )
        
        # Access unloaded dataset
        @test_throws ErrorException get_feature_column(replica, "f1")
        @test_throws ErrorException get_sample_batch(replica, [1, 2, 3])
    end
    
end

# Print summary
println("\nDataset Storage Test Summary:")
println("=============================")
if CUDA.functional()
    num_gpus = length(CUDA.devices())
    println("✓ CUDA functional - GPU tests executed")
    println("  GPUs detected: $num_gpus")
    println("  Dataset replication tested")
    println("  GPU memory management validated")
else
    println("⚠ CUDA not functional - CPU reference tests only")
end
println("\nAll dataset storage tests completed!")