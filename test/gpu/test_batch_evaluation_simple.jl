using Test
using CUDA

# Include modules
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/batch_evaluation.jl")

using .MCTSTypes
using .BatchEvaluation

@testset "Simple Batch Evaluation Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping batch evaluation tests"
        return
    end
    
    @testset "Batch Configuration" begin
        config = BatchEvalConfig(
            Int32(256),   # batch_size
            true,         # double_buffer
            Int32(128),   # eval_threads
            0.7f0         # coalesce_threshold
        )
        
        @test config.batch_size == 256
        @test config.double_buffer == true
        @test config.eval_threads == 128
        @test config.coalesce_threshold ≈ 0.7f0
    end
    
    @testset "Buffer Creation" begin
        batch_size = Int32(64)
        num_features = Int32(100)
        max_actions = Int32(16)
        
        buffer = EvalBatchBuffer(batch_size, num_features, max_actions)
        
        # Check that buffers are allocated
        @test size(buffer.features_a) == (num_features, batch_size)
        @test size(buffer.features_b) == (num_features, batch_size)
        @test length(buffer.scores_a) == batch_size
        @test length(buffer.scores_b) == batch_size
        
        # Check initial state
        CUDA.@allowscalar begin
            @test buffer.batch_size_a[1] == 0
            @test buffer.batch_size_b[1] == 0
            @test buffer.active_buffer[1] == 0
        end
    end
    
    @testset "Dynamic Batch Size Calculation" begin
        config = BatchEvalConfig(Int32(1024), true, Int32(256), 0.7f0)
        
        # Low occupancy - should use full batch
        size1 = compute_dynamic_batch_size(Int32(800), 0.5f0, config)
        @test size1 == 800
        
        # High occupancy - should use half max
        size2 = compute_dynamic_batch_size(Int32(2000), 0.8f0, config)
        @test size2 == 512
        
        # Limited by available nodes
        size3 = compute_dynamic_batch_size(Int32(100), 0.8f0, config)
        @test size3 == 100
    end
    
    @testset "Feature Mask Helpers" begin
        # Test has_feature function
        masks = CUDA.zeros(UInt64, FEATURE_CHUNKS, 10)
        
        # Set feature 5 for node 1
        CUDA.@allowscalar begin
            masks[1, 1] = UInt64(1) << 4  # Feature 5 (0-indexed)
        end
        
        # Test on CPU first
        masks_cpu = Array(masks)
        @test MCTSTypes.has_feature(masks_cpu, 1, 5) == true
        @test MCTSTypes.has_feature(masks_cpu, 1, 6) == false
        
        # Test feature counting
        CUDA.@allowscalar begin
            # Set multiple features
            masks[1, 2] = UInt64(0b1111)  # First 4 features
        end
        
        count = CUDA.@allowscalar MCTSTypes.count_features(masks, Int32(2))
        @test count == 4
    end
    
    @testset "Dispatch Logic" begin
        buffer = EvalBatchBuffer(Int32(32), Int32(50), Int32(8))
        
        # Test buffer switching
        CUDA.@allowscalar begin
            # Initially on buffer A
            @test buffer.active_buffer[1] == 0
            
            # Simulate dispatch
            buffer.batch_size_a[1] = 10
            buffer.ready_flag[1] = true
            buffer.active_buffer[1] = 1  # Switch to B
            
            @test buffer.active_buffer[1] == 1
            @test buffer.ready_flag[1] == true
        end
    end
end

println("\n✅ Simple batch evaluation tests passed!")