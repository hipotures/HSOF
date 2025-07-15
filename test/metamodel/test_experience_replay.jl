using Test
using CUDA
using Statistics
using Random

# Include the experience replay module
include("../../src/metamodel/experience_replay.jl")

using .ExperienceReplay

@testset "Experience Replay Buffer Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU experience replay tests"
        return
    end
    
    @testset "ReplayConfig Creation" begin
        # Test default configuration
        config = create_replay_config()
        
        @test config.buffer_size == 1000
        @test config.max_features == 100
        @test config.priority_alpha == 0.6f0
        @test config.priority_beta == 0.4f0
        @test config.priority_epsilon == 1f-6
        @test config.batch_size == 32
        @test config.use_memory_pool == true
        
        # Test custom configuration
        custom_config = create_replay_config(
            buffer_size = 500,
            max_features = 50,
            priority_alpha = 0.8f0,
            batch_size = 64
        )
        
        @test custom_config.buffer_size == 500
        @test custom_config.max_features == 50
        @test custom_config.priority_alpha == 0.8f0
        @test custom_config.batch_size == 64
    end
    
    @testset "ReplayBuffer Creation" begin
        config = create_replay_config(buffer_size=100)
        buffer = create_replay_buffer(config)
        
        @test isa(buffer, ReplayBuffer)
        @test buffer.config == config
        
        # Check array dimensions
        @test size(buffer.experience.feature_indices) == (100, 100)
        @test length(buffer.experience.n_features) == 100
        @test length(buffer.experience.priorities) == 100
        @test length(buffer.experience.valid) == 100
        
        # Check initial state
        @test CUDA.@allowscalar(buffer.current_idx[]) == 0
        @test CUDA.@allowscalar(buffer.total_count[]) == 0
        @test CUDA.@allowscalar(buffer.max_priority[]) == 1.0f0
        
        # Check sum tree size
        @test length(buffer.sum_tree) == 200  # 2 * buffer_size
    end
    
    @testset "Single Experience Insertion" begin
        config = create_replay_config(buffer_size=10, max_features=5)
        buffer = create_replay_buffer(config)
        
        # Insert single experience
        features = Int32[1, 3, 5]
        predicted = 0.8f0
        actual = 0.75f0
        
        insert_experience!(buffer, features, predicted, actual)
        
        # Check state after insertion
        stats = get_buffer_stats(buffer)
        @test stats.n_valid == 1
        @test stats.total_count == 1
        
        # Check stored data
        @test CUDA.@allowscalar(buffer.experience.n_features[1]) == 3
        @test CUDA.@allowscalar(buffer.experience.predicted_scores[1]) == predicted
        @test CUDA.@allowscalar(buffer.experience.actual_scores[1]) == actual
        @test CUDA.@allowscalar(buffer.experience.valid[1]) == true
        
        # Check TD error
        td_error = abs(predicted - actual)
        @test CUDA.@allowscalar(buffer.experience.td_errors[1]) ≈ td_error
        
        # Check priority calculation
        expected_priority = (td_error + config.priority_epsilon)^config.priority_alpha
        @test CUDA.@allowscalar(buffer.experience.priorities[1]) ≈ expected_priority
    end
    
    @testset "Batch Insertion" begin
        config = create_replay_config(buffer_size=20, max_features=5)
        buffer = create_replay_buffer(config)
        
        # Prepare batch data
        batch_size = 5
        features_batch = Int32[1 2 3 4 5;
                              2 3 4 5 6;
                              3 4 5 6 7;
                              0 0 0 0 0;
                              0 0 0 0 0]
        predicted_batch = Float32[0.7, 0.8, 0.6, 0.9, 0.75]
        actual_batch = Float32[0.65, 0.85, 0.55, 0.88, 0.8]
        
        batch_insert!(buffer, features_batch, predicted_batch, actual_batch)
        
        # Check state
        stats = get_buffer_stats(buffer)
        @test stats.n_valid == batch_size
        @test stats.total_count == batch_size
        
        # Verify data
        for i in 1:batch_size
            @test CUDA.@allowscalar(buffer.experience.n_features[i]) == 3
            @test CUDA.@allowscalar(buffer.experience.predicted_scores[i]) == predicted_batch[i]
            @test CUDA.@allowscalar(buffer.experience.actual_scores[i]) == actual_batch[i]
            @test CUDA.@allowscalar(buffer.experience.valid[i]) == true
        end
    end
    
    @testset "Circular Buffer Behavior" begin
        config = create_replay_config(buffer_size=5, max_features=3)
        buffer = create_replay_buffer(config)
        
        # Fill buffer beyond capacity
        for i in 1:8
            features = Int32[i, i+1, i+2]
            insert_experience!(buffer, features, Float32(i)/10, Float32(i)/10 + 0.05f0)
        end
        
        # Check that buffer wraps around
        stats = get_buffer_stats(buffer)
        @test stats.n_valid == 5  # Buffer size
        @test stats.total_count == 8  # Total inserted
        
        # Last 5 insertions should be present
        current_idx = CUDA.@allowscalar(buffer.current_idx[]) % 5
        for i in 1:5
            idx = mod1(current_idx - 5 + i, 5)
            @test CUDA.@allowscalar(buffer.experience.valid[idx]) == true
        end
    end
    
    @testset "Priority Sampling" begin
        config = create_replay_config(buffer_size=10, max_features=3)
        buffer = create_replay_buffer(config)
        
        # Insert experiences with different TD errors
        for i in 1:5
            features = Int32[i, i+1, i+2]
            # Larger error = higher priority
            predicted = Float32(i) / 10
            actual = predicted + Float32(i) * 0.1f0  # Increasing errors
            insert_experience!(buffer, features, predicted, actual)
        end
        
        # Update sum tree
        update_sum_tree!(buffer)
        
        # Sample batch
        batch = sample_batch(buffer, 3)
        
        @test !isnothing(batch)
        @test length(batch.indices) == 3
        @test size(batch.features) == (3, 3)
        @test length(batch.weights) == 3
        
        # Weights should be normalized
        @test maximum(Array(batch.weights)) ≈ 1.0f0
        @test all(Array(batch.weights) .> 0)
    end
    
    @testset "Priority Updates" begin
        config = create_replay_config(buffer_size=10, max_features=3)
        buffer = create_replay_buffer(config)
        
        # Insert some experiences
        for i in 1:5
            features = Int32[i, i+1, i+2]
            insert_experience!(buffer, features, 0.5f0, 0.5f0)
        end
        
        # Update sum tree before sampling
        update_sum_tree!(buffer)
        
        # Sample and update priorities
        batch = sample_batch(buffer, 3)
        new_td_errors = CUDA.ones(Float32, 3) * 0.1f0
        
        update_priorities!(buffer, batch.indices, new_td_errors)
        
        # Check priorities were updated
        for (i, idx) in enumerate(Array(batch.indices))
            new_priority = (0.1f0 + config.priority_epsilon)^config.priority_alpha
            @test CUDA.@allowscalar(buffer.experience.priorities[idx]) ≈ new_priority
        end
    end
    
    @testset "Buffer Statistics" begin
        config = create_replay_config(buffer_size=20, max_features=5)
        buffer = create_replay_buffer(config)
        
        # Insert varied experiences
        for i in 1:15
            features = Int32[i, i+1]
            predicted = rand(Float32) * 0.5f0 + 0.5f0
            actual = predicted + randn(Float32) * 0.1f0
            insert_experience!(buffer, features, predicted, actual)
        end
        
        # Update sum tree
        update_sum_tree!(buffer)
        
        stats = get_buffer_stats(buffer)
        
        @test stats.n_valid == 15
        @test stats.total_count == 15
        @test stats.buffer_utilization == 0.75
        @test stats.avg_td_error > 0
        @test stats.avg_priority > 0
        @test stats.max_priority >= stats.avg_priority  # max should be at least avg
    end
    
    @testset "Clear Buffer" begin
        config = create_replay_config(buffer_size=10)
        buffer = create_replay_buffer(config)
        
        # Add some experiences
        for i in 1:5
            insert_experience!(buffer, Int32[i, i+1], 0.5f0, 0.6f0)
        end
        
        # Clear
        clear_buffer!(buffer)
        
        stats = get_buffer_stats(buffer)
        @test stats.n_valid == 0
        @test stats.total_count == 0
        @test CUDA.@allowscalar(buffer.current_idx[]) == 0
        @test all(.!buffer.experience.valid)
    end
    
    @testset "Concurrent Operations" begin
        config = create_replay_config(buffer_size=100, max_features=10)
        buffer = create_replay_buffer(config)
        
        # Simulate concurrent insertions and sampling
        for _ in 1:10
            # Insert batch
            features = rand(Int32(1):Int32(100), 10, 5)
            predicted = rand(Float32, 5)
            actual = predicted .+ randn(Float32, 5) * 0.1f0
            
            batch_insert!(buffer, features, predicted, actual)
            
            # Sample if enough data
            if CUDA.@allowscalar(sum(buffer.experience.valid)) >= 10
                batch = sample_batch(buffer, 8)
                @test !isnothing(batch)
                @test length(batch.indices) == 8
            end
        end
        
        # Buffer should handle concurrent ops without issues
        stats = get_buffer_stats(buffer)
        @test stats.n_valid > 0
        @test stats.total_count >= stats.n_valid
    end
end

println("\n✅ Experience replay buffer tests completed!")