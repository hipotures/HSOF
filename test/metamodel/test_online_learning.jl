using Test
using CUDA
using Flux
using Statistics

# Include modules
include("../../src/metamodel/neural_architecture.jl")
include("../../src/metamodel/experience_replay.jl")
include("../../src/metamodel/online_learning.jl")

using .NeuralArchitecture
using .ExperienceReplay
using .OnlineLearning

@testset "Online Learning System Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU online learning tests"
        return
    end
    
    @testset "OnlineLearningConfig Creation" begin
        # Test default configuration
        config = create_online_config()
        
        @test config.batch_size == 32
        @test config.update_frequency == 100
        @test config.learning_rate == 1f-4
        @test config.weight_decay == 1f-5
        @test config.gradient_clip == 1.0f0
        @test config.use_double_buffering == true
        @test config.lr_scheduler == :adaptive
        @test config.accumulation_steps == 4
        
        # Test custom configuration
        custom_config = create_online_config(
            batch_size = 64,
            update_frequency = 50,
            learning_rate = 1f-3,
            lr_scheduler = :exponential
        )
        
        @test custom_config.batch_size == 64
        @test custom_config.update_frequency == 50
        @test custom_config.learning_rate == 1f-3
        @test custom_config.lr_scheduler == :exponential
    end
    
    @testset "Online Learning Initialization" begin
        # Create model and config
        model_config = create_model_config(input_dim=100)
        model = create_metamodel(model_config)
        ol_config = create_online_config()
        
        # Initialize online learning
        state = initialize_online_learning(model, ol_config)
        
        @test isa(state, OnlineLearningState)
        @test state.current_lr == ol_config.learning_rate
        @test state.iteration_count == 0
        @test state.update_count == 0
        @test length(state.correlation_history) == 0
        
        # Check double buffering
        if ol_config.use_double_buffering
            @test !isnothing(state.inference_model)
            @test state.inference_model !== state.primary_model
        end
        
        # Check CUDA streams
        @test isa(state.training_stream, CuStream)
        @test isa(state.inference_stream, CuStream)
    end
    
    @testset "Batch Input Preparation" begin
        model_config = create_model_config(input_dim=20)
        
        # Create sparse feature indices
        feature_indices = CuArray(Int32[1 5 10;
                                       3 7 15;
                                       0 12 0;
                                       0 0 0])
        n_features = CuArray(Int32[2, 3, 2])
        
        # Prepare batch inputs
        inputs = prepare_batch_inputs(feature_indices, n_features, model_config)
        
        @test size(inputs) == (20, 3)
        
        # Check sparse to dense conversion
        inputs_cpu = Array(inputs)
        @test inputs_cpu[1, 1] == 1.0f0
        @test inputs_cpu[3, 1] == 1.0f0
        @test inputs_cpu[5, 2] == 1.0f0
        @test inputs_cpu[7, 2] == 1.0f0
        @test inputs_cpu[12, 2] == 1.0f0
        @test inputs_cpu[10, 3] == 1.0f0
        @test inputs_cpu[15, 3] == 1.0f0
        
        # Check zeros elsewhere
        @test sum(inputs_cpu[:, 1]) == 2.0f0
        @test sum(inputs_cpu[:, 2]) == 3.0f0
        @test sum(inputs_cpu[:, 3]) == 2.0f0
    end
    
    @testset "Online Update" begin
        # Setup
        model_config = create_model_config(input_dim=50, hidden_dim=64)
        model = create_metamodel(model_config)
        ol_config = create_online_config(
            batch_size = 4,
            update_frequency = 2,
            accumulation_steps = 1
        )
        state = initialize_online_learning(model, ol_config)
        
        # Create replay buffer and add experiences
        replay_config = create_replay_config(buffer_size=100, max_features=10)
        buffer = create_replay_buffer(replay_config)
        
        # Add some experiences
        for i in 1:10
            features = Int32[i, i+1, i+2]
            predicted = 0.5f0 + 0.1f0 * randn(Float32)
            actual = 0.5f0 + 0.05f0 * randn(Float32)
            insert_experience!(buffer, features, predicted, actual)
        end
        update_sum_tree!(buffer)
        
        # Test that no update happens before frequency
        state.iteration_count = 1
        updated = online_update!(state, buffer, ol_config)
        @test !updated
        @test state.update_count == 0
        
        # Test update at frequency
        state.iteration_count = 1  # Reset for proper modulo
        updated = online_update!(state, buffer, ol_config)
        @test updated
        @test state.update_count == 1
        @test state.iteration_count == 2
    end
    
    @testset "Learning Rate Scheduling" begin
        model_config = create_model_config(input_dim=10)
        model = create_metamodel(model_config)
        
        # Test exponential decay
        exp_config = create_online_config(lr_scheduler=:exponential, lr_decay_rate=0.9f0)
        exp_state = initialize_online_learning(model, exp_config)
        exp_state.update_count = 1000
        update_learning_rate!(exp_state, exp_config, 0.1f0)
        @test exp_state.current_lr < exp_config.learning_rate
        
        # Test constant
        const_config = create_online_config(lr_scheduler=:constant)
        const_state = initialize_online_learning(model, const_config)
        const_state.update_count = 1000
        initial_lr = const_state.current_lr
        update_learning_rate!(const_state, const_config, 0.1f0)
        @test const_state.current_lr == initial_lr
    end
    
    @testset "Correlation Tracking" begin
        model_config = create_model_config(input_dim=20)
        model = create_metamodel(model_config)
        ol_config = create_online_config(correlation_window=100)
        state = initialize_online_learning(model, ol_config)
        
        # Create test inputs
        inputs = CUDA.rand(Float32, 20, 5)
        actuals = CUDA.rand(Float32, 5) * 0.5f0 .+ 0.25f0
        
        # Update tracking
        state.update_count = 9  # Will trigger correlation computation on next update
        update_correlation_tracking!(state, inputs, actuals, ol_config)
        
        # Check buffers updated
        @test state.recent_idx > 1
        
        # Force correlation computation
        state.update_count = 10
        state.iteration_count = 50
        update_correlation_tracking!(state, inputs, actuals, ol_config)
        
        # Should have computed correlation
        @test length(state.correlation_history) >= 1
    end
    
    @testset "Model Synchronization" begin
        model_config = create_model_config(input_dim=10)
        model = create_metamodel(model_config)
        ol_config = create_online_config(use_double_buffering=true)
        state = initialize_online_learning(model, ol_config)
        
        # Modify primary model weights
        primary_params = Flux.params(state.primary_model)
        inference_params = Flux.params(state.inference_model)
        
        # Get initial weight from first dense layer
        primary_weight = primary_params[1]
        inference_weight = inference_params[1]
        
        # Modify primary
        primary_weight .+= 0.1f0
        
        # Weights should differ
        @test !all(primary_weight .≈ inference_weight)
        
        # Sync models
        sync_inference_model!(state)
        CUDA.synchronize()
        
        # Weights should match after sync
        @test all(Array(primary_weight) .≈ Array(inference_weight))
    end
    
    @testset "Gradient Accumulation" begin
        model_config = create_model_config(input_dim=30)
        model = create_metamodel(model_config)
        ol_config = create_online_config(
            batch_size = 2,
            update_frequency = 1,
            accumulation_steps = 3
        )
        state = initialize_online_learning(model, ol_config)
        
        # Create replay buffer with experiences
        replay_config = create_replay_config(buffer_size=50, max_features=5)
        buffer = create_replay_buffer(replay_config)
        
        for i in 1:20
            features = Int32[i, i+1]
            insert_experience!(buffer, features, 0.5f0, 0.6f0)
        end
        update_sum_tree!(buffer)
        
        # First update - should accumulate
        online_update!(state, buffer, ol_config)
        @test state.accumulation_count == 1
        @test state.update_count == 0
        
        # Second update - should accumulate
        state.iteration_count = 0  # Reset for modulo
        online_update!(state, buffer, ol_config)
        @test state.accumulation_count == 2
        @test state.update_count == 0
        
        # Third update - should apply
        state.iteration_count = 0  # Reset for modulo
        online_update!(state, buffer, ol_config)
        @test state.accumulation_count == 0
        @test state.update_count == 1
    end
    
    @testset "Online Statistics" begin
        model_config = create_model_config(input_dim=10)
        model = create_metamodel(model_config)
        ol_config = create_online_config()
        state = initialize_online_learning(model, ol_config)
        
        # Set some state
        state.iteration_count = 1000
        state.update_count = 10
        state.current_lr = 5f-5
        state.correlation_history = [0.85f0, 0.87f0, 0.89f0, 0.91f0, 0.93f0]
        state.total_training_time = 100.0
        
        stats = get_online_stats(state)
        
        @test stats.iteration_count == 1000
        @test stats.update_count == 10
        @test stats.current_lr == 5f-5
        @test stats.avg_correlation ≈ mean([0.85f0, 0.87f0, 0.89f0, 0.91f0, 0.93f0])
        @test stats.updates_per_second ≈ 0.1
    end
    
    @testset "Checkpoint Save/Load" begin
        model_config = create_model_config(input_dim=10)
        model = create_metamodel(model_config)
        ol_config = create_online_config()
        state = initialize_online_learning(model, ol_config)
        
        # Set some state
        state.iteration_count = 500
        state.update_count = 5
        state.correlation_history = [0.9f0, 0.92f0]
        state.total_training_time = 50.0
        
        # Save checkpoint
        temp_file = tempname() * ".jld2"
        save_checkpoint(state, temp_file)
        
        @test isfile(temp_file)
        
        # Create new state and load
        new_state = initialize_online_learning(model, ol_config)
        load_checkpoint!(new_state, temp_file)
        
        @test new_state.iteration_count == 500
        @test new_state.update_count == 5
        @test new_state.correlation_history == [0.9f0, 0.92f0]
        @test new_state.total_training_time == 50.0
        
        # Cleanup
        rm(temp_file)
    end
end

println("\n✅ Online learning system tests completed!")