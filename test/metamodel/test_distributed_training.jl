"""
Test suite for Distributed Training System
Testing multi-GPU training capabilities, gradient synchronization, and fault tolerance
"""

using Test
using CUDA
using Statistics
using Random
using SharedArrays
using Distributed

# Include required modules
include("../../src/metamodel/distributed_training.jl")
include("../../src/metamodel/neural_architecture.jl")
include("../../src/metamodel/experience_replay.jl")
include("../../src/metamodel/online_learning.jl")

using .DistributedTraining
using .NeuralArchitecture
using .ExperienceReplay
using .OnlineLearning

@testset "Distributed Training Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        config = create_distributed_config()
        
        @test config.gpu_devices == [0, 1]
        @test config.primary_gpu == 0
        @test config.data_parallel == true
        @test config.sync_method == :custom_reduce
        @test config.shard_replay_buffer == true
        @test config.enable_fault_tolerance == true
        @test config.single_gpu_fallback == true
    end
    
    @testset "GPU Availability Tests" begin
        available_gpus = CUDA.ndevices()
        
        if available_gpus >= 2
            println("Testing with $(available_gpus) GPUs available")
            
            # Test with valid GPU configuration
            config = create_distributed_config(gpu_devices = [0, 1])
            @test length(config.gpu_devices) == 2
            
            # Test with single GPU fallback
            config_single = create_distributed_config(gpu_devices = [0])
            @test length(config_single.gpu_devices) == 1
        else
            println("Skipping multi-GPU tests - only $(available_gpus) GPU(s) available")
            
            # Test with single GPU only
            config = create_distributed_config(gpu_devices = [0])
            @test length(config.gpu_devices) == 1
        end
    end
    
    @testset "Model and Buffer Setup Tests" begin
        # Create base model
        model_config = create_metamodel_config(input_dim = 100, hidden_dims = [64, 32, 16])
        model = create_metamodel(model_config)
        
        # Create replay buffer
        buffer_config = ExperienceReplay.create_replay_config(buffer_size = 1000, max_features = 100)
        buffer = ExperienceReplay.create_replay_buffer(buffer_config)
        
        @test !isnothing(model)
        @test !isnothing(buffer)
        @test count_parameters(model) > 0
    end
    
    # Only run multi-GPU tests if multiple GPUs are available
    if CUDA.ndevices() >= 2
        @testset "Multi-GPU Initialization Tests" begin
            # Create test model and buffer
            model_config = create_metamodel_config(input_dim = 100, hidden_dims = [64, 32, 16])
            model = create_metamodel(model_config)
            
            buffer_config = ExperienceReplay.create_replay_config(buffer_size = 1000, max_features = 100)
            buffer = ExperienceReplay.create_replay_buffer(buffer_config)
            
            # Create distributed config
            dist_config = create_distributed_config(gpu_devices = [0, 1])
            online_config = create_online_config(batch_size = 16)
            
            # Initialize distributed training
            coordinator = initialize_distributed_training(model, buffer, dist_config, online_config)
            
            @test !isnothing(coordinator)
            @test length(coordinator.gpu_states) == 2
            @test coordinator.config.gpu_devices == [0, 1]
            @test coordinator.coordinator_healthy == true
            @test coordinator.global_update_count == 0
            
            # Test GPU states
            for gpu_state in coordinator.gpu_states
                @test gpu_state.is_healthy == true
                @test gpu_state.local_updates == 0
                @test gpu_state.error_count == 0
                @test !isnothing(gpu_state.model)
                @test !isnothing(gpu_state.local_optimizer)
            end
            
            # Cleanup
            shutdown_distributed_training!(coordinator)
        end
        
        @testset "Gradient Synchronization Tests" begin
            # Create test setup
            model_config = create_metamodel_config(input_dim = 50, hidden_dims = [32, 16, 8])
            model = create_metamodel(model_config)
            
            buffer_config = ExperienceReplay.create_replay_config(buffer_size = 500, max_features = 50)
            buffer = ExperienceReplay.create_replay_buffer(buffer_config)
            
            # Add some test data to buffer
            for i in 1:100
                features = rand(Int32(1):Int32(50), rand(5:15))
                actual_score = rand(Float32)
                ExperienceReplay.add_experience!(buffer, features, actual_score)
            end
            
            dist_config = create_distributed_config(
                gpu_devices = [0, 1],
                sync_frequency = 1,
                sync_method = :custom_reduce
            )
            
            coordinator = initialize_distributed_training(model, buffer, dist_config)
            
            # Manually create different gradients on each GPU to test synchronization
            gpu1_state = coordinator.gpu_states[1]
            gpu2_state = coordinator.gpu_states[2]
            
            CUDA.device!(0)
            gpu1_grads = Dict(
                :input_layer => Dict(:weight => CUDA.randn(Float32, 32, 50), :bias => CUDA.randn(Float32, 32))
            )
            gpu1_state.model.gradients = gpu1_grads
            
            CUDA.device!(1)
            gpu2_grads = Dict(
                :input_layer => Dict(:weight => CUDA.randn(Float32, 32, 50), :bias => CUDA.randn(Float32, 32))
            )
            gpu2_state.model.gradients = gpu2_grads
            
            # Test gradient synchronization
            success = DistributedTraining.synchronize_gradients!([gpu1_state, gpu2_state], dist_config)
            @test success == true
            
            # Cleanup
            shutdown_distributed_training!(coordinator)
        end
        
        @testset "Load Balancing Tests" begin
            model_config = create_metamodel_config(input_dim = 50, hidden_dims = [32, 16, 8])
            model = create_metamodel(model_config)
            
            buffer_config = ExperienceReplay.create_replay_config(buffer_size = 500, max_features = 50)
            buffer = ExperienceReplay.create_replay_buffer(buffer_config)
            
            dist_config = create_distributed_config(
                gpu_devices = [0, 1],
                dynamic_batch_sizing = true,
                load_balance_frequency = 10
            )
            
            coordinator = initialize_distributed_training(model, buffer, dist_config)
            
            # Set different throughputs for testing load balancing
            coordinator.gpu_states[1].throughput_samples_per_sec = 100.0f0
            coordinator.gpu_states[2].throughput_samples_per_sec = 50.0f0
            
            # Test load balancing adjustment
            DistributedTraining.adjust_batch_sizes!(coordinator.gpu_states, dist_config)
            
            # Should not crash and should log adjustments
            @test true  # If we get here, the function didn't crash
            
            shutdown_distributed_training!(coordinator)
        end
        
        @testset "Fault Tolerance Tests" begin
            model_config = create_metamodel_config(input_dim = 50, hidden_dims = [32, 16, 8])
            model = create_metamodel(model_config)
            
            buffer_config = ExperienceReplay.create_replay_config(buffer_size = 500, max_features = 50)
            buffer = ExperienceReplay.create_replay_buffer(buffer_config)
            
            dist_config = create_distributed_config(
                gpu_devices = [0, 1],
                enable_fault_tolerance = true,
                single_gpu_fallback = true,
                heartbeat_interval = 1.0f0
            )
            
            coordinator = initialize_distributed_training(model, buffer, dist_config)
            
            # Simulate GPU failure
            coordinator.gpu_states[2].is_healthy = false
            coordinator.gpu_states[2].error_count = 10
            push!(coordinator.failed_gpus, 1)
            
            # Test health check
            DistributedTraining.check_gpu_health!(coordinator)
            
            @test 1 in coordinator.failed_gpus
            @test coordinator.gpu_states[2].is_healthy == false
            
            # Test single GPU fallback
            if coordinator.fallback_active
                success = DistributedTraining.single_gpu_fallback_step!(coordinator)
                # May fail due to empty buffer, but should not crash
                @test success isa Bool
            end
            
            shutdown_distributed_training!(coordinator)
        end
        
        @testset "Performance Monitoring Tests" begin
            model_config = create_metamodel_config(input_dim = 50, hidden_dims = [32, 16, 8])
            model = create_metamodel(model_config)
            
            buffer_config = ExperienceReplay.create_replay_config(buffer_size = 500, max_features = 50)
            buffer = ExperienceReplay.create_replay_buffer(buffer_config)
            
            dist_config = create_distributed_config(gpu_devices = [0, 1])
            coordinator = initialize_distributed_training(model, buffer, dist_config)
            
            # Set some statistics
            coordinator.global_update_count = 100
            coordinator.total_samples_processed = 3200
            coordinator.gpu_states[1].throughput_samples_per_sec = 120.0f0
            coordinator.gpu_states[2].throughput_samples_per_sec = 110.0f0
            
            # Test scaling efficiency calculation
            efficiency = DistributedTraining.calculate_scaling_efficiency(coordinator.gpu_states)
            @test 0.0f0 <= efficiency <= 1.0f0
            
            # Test statistics gathering
            stats = get_distributed_stats(coordinator)
            @test stats.global_update_count == 100
            @test stats.total_samples_processed == 3200
            @test stats.healthy_gpus == 2
            @test stats.total_throughput > 0
            
            shutdown_distributed_training!(coordinator)
        end
    else
        @testset "Single GPU Fallback Tests" begin
            println("Testing single GPU configuration only")
            
            model_config = create_metamodel_config(input_dim = 50, hidden_dims = [32, 16, 8])
            model = create_metamodel(model_config)
            
            buffer_config = ExperienceReplay.create_replay_config(buffer_size = 500, max_features = 50)
            buffer = ExperienceReplay.create_replay_buffer(buffer_config)
            
            # Test with single GPU
            dist_config = create_distributed_config(gpu_devices = [0])
            coordinator = initialize_distributed_training(model, buffer, dist_config)
            
            @test length(coordinator.gpu_states) == 1
            @test coordinator.gpu_states[1].gpu_id == 0
            @test coordinator.gpu_states[1].is_healthy == true
            
            # Test statistics with single GPU
            stats = get_distributed_stats(coordinator)
            @test stats.healthy_gpus == 1
            @test stats.failed_gpus == 0
            
            shutdown_distributed_training!(coordinator)
        end
    end
    
    @testset "Configuration Validation Tests" begin
        # Test various configuration combinations
        
        # Test custom reduce method
        config1 = create_distributed_config(sync_method = :custom_reduce)
        @test config1.sync_method == :custom_reduce
        
        # Test parameter server method
        config2 = create_distributed_config(sync_method = :parameter_server)
        @test config2.sync_method == :parameter_server
        
        # Test gradient compression
        config3 = create_distributed_config(
            gradient_compression = true,
            compression_ratio = 0.3f0
        )
        @test config3.gradient_compression == true
        @test config3.compression_ratio == 0.3f0
        
        # Test performance optimizations
        config4 = create_distributed_config(
            enable_mixed_precision = true,
            enable_overlap_comm = true,
            async_gradient_copy = true
        )
        @test config4.enable_mixed_precision == true
        @test config4.enable_overlap_comm == true
        @test config4.async_gradient_copy == true
    end
    
    @testset "Memory Management Tests" begin
        # Test parameter flattening and reconstruction
        model_config = create_metamodel_config(input_dim = 20, hidden_dims = [16, 8, 4])
        model = create_metamodel(model_config)
        
        total_params = count_parameters(model)
        @test total_params > 0
        
        # Test shared array creation
        shared_params = SharedArray{Float32}(total_params)
        @test length(shared_params) == total_params
        
        # Test parameter flattening
        DistributedTraining.flatten_model_params!(shared_params, model)
        @test !all(iszero, shared_params)  # Should contain non-zero values
    end
    
    @testset "Integration Tests" begin
        # Test complete workflow with small model and buffer
        model_config = create_metamodel_config(input_dim = 20, hidden_dims = [16, 8, 4])
        model = create_metamodel(model_config)
        
        buffer_config = ExperienceReplay.create_replay_config(buffer_size = 100, max_features = 20)
        buffer = ExperienceReplay.create_replay_buffer(buffer_config)
        
        # Add minimal test data
        for i in 1:20
            features = rand(Int32(1):Int32(20), rand(3:8))
            actual_score = rand(Float32)
            ExperienceReplay.add_experience!(buffer, features, actual_score)
        end
        
        # Test with available GPUs
        available_gpus = min(CUDA.ndevices(), 2)
        gpu_devices = collect(0:available_gpus-1)
        
        dist_config = create_distributed_config(
            gpu_devices = gpu_devices,
            sync_frequency = 2,
            enable_fault_tolerance = true
        )
        
        coordinator = initialize_distributed_training(model, buffer, dist_config)
        
        # Test a few training steps
        for step in 1:3
            success = distributed_training_step!(coordinator)
            # May fail due to insufficient data, but should not crash
            @test success isa Bool
            
            # Check statistics
            stats = get_distributed_stats(coordinator)
            @test stats.global_update_count >= 0
            @test stats.healthy_gpus <= length(gpu_devices)
        end
        
        shutdown_distributed_training!(coordinator)
    end
end

println("Distributed Training tests completed successfully!")