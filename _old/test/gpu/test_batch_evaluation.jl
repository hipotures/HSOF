using Test
using CUDA
using Statistics

# Include modules
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/batch_evaluation.jl")

using .MCTSTypes
using .BatchEvaluation

@testset "Batch Evaluation Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping batch evaluation tests"
        return
    end
    
    @testset "Batch Buffer Creation" begin
        batch_size = Int32(256)
        num_features = Int32(100)
        max_actions = Int32(16)
        
        buffer = EvalBatchBuffer(batch_size, num_features, max_actions)
        
        # Check dimensions
        @test size(buffer.features_a) == (num_features, batch_size)
        @test size(buffer.features_b) == (num_features, batch_size)
        @test size(buffer.scores_a) == (batch_size,)
        @test size(buffer.priors_a) == (max_actions, batch_size)
        
        # Check initial state
        CUDA.@allowscalar begin
            @test buffer.batch_size_a[1] == 0
            @test buffer.batch_size_b[1] == 0
            @test buffer.active_buffer[1] == 0
            @test all(buffer.ready_flag .== false)
        end
    end
    
    @testset "Feature Conversion" begin
        # Create mock feature masks
        feature_masks = CUDA.zeros(UInt64, FEATURE_CHUNKS, 10)
        dense_features = CUDA.zeros(Float32, 100, 10)
        
        # Set some features for node 1
        CUDA.@allowscalar begin
            # Set features 1, 10, 50
            feature_masks[1, 1] = UInt64(1) | (UInt64(1) << 9) | (UInt64(1) << 49)
        end
        
        # Test kernel that converts features
        function test_conversion_kernel!(feature_masks, dense_features)
            tid = threadIdx().x
            
            convert_features_to_dense!(
                feature_masks,
                dense_features,
                Int32(1),  # node_idx
                Int32(1)   # batch_pos
            )
            
            return nothing
        end
        
        @cuda threads=32 test_conversion_kernel!(feature_masks, dense_features)
        
        # Check results
        dense_host = Array(dense_features[:, 1])
        @test dense_host[1] ≈ 1.0f0
        @test dense_host[10] ≈ 1.0f0
        @test dense_host[50] ≈ 1.0f0
        @test dense_host[2] ≈ 0.0f0
        @test dense_host[51] ≈ 0.0f0
    end
    
    @testset "Dynamic Batch Size" begin
        config = BatchEvalConfig(
            Int32(1024),  # max batch size
            true,
            Int32(256),
            0.7f0
        )
        
        # Test low occupancy
        batch_size = compute_dynamic_batch_size(Int32(500), 0.3f0, config)
        @test batch_size == 500  # Should use all available
        
        # Test high occupancy
        batch_size = compute_dynamic_batch_size(Int32(2000), 0.9f0, config)
        @test batch_size == 512  # Should use half of max
        
        # Test edge case
        batch_size = compute_dynamic_batch_size(Int32(100), 0.9f0, config)
        @test batch_size == 100  # Limited by available nodes
    end
    
    @testset "Double Buffering" begin
        batch_size = Int32(64)
        buffer = EvalBatchBuffer(batch_size, Int32(100), Int32(16))
        
        # Simulate dispatch
        function test_dispatch!(buffer)
            tid = threadIdx().x
            
            if tid == 1
                # Add some nodes to buffer A
                buffer.batch_size_a[1] = 32
                buffer.active_buffer[1] = 0
                
                # Dispatch buffer A
                dispatch_eval_batch!(buffer, Int32(0), Int32(32))
            end
            
            return nothing
        end
        
        @cuda threads=1 test_dispatch!(buffer)
        
        CUDA.@allowscalar begin
            # Check that buffer switched
            @test buffer.active_buffer[1] == 1
            @test buffer.ready_flag[1] == true
            @test buffer.ready_flag[2] == false
            @test buffer.batch_size_b[1] == 0
        end
    end
    
    @testset "Batch Collection and Scatter" begin
        # Create minimal tree and work queue
        tree = MCTSTreeSoA(CUDA.device())
        
        # Initialize some nodes
        CUDA.@allowscalar begin
            tree.total_nodes[1] = 5
            tree.next_free_node[1] = 6
            for i in 1:5
                tree.node_ids[i] = i
                tree.node_states[i] = NODE_ACTIVE
                tree.prior_scores[i] = 0.0f0
            end
        end
        
        # Create mock work queue
        work_items = CUDA.zeros(Int32, 4, 10)
        work_size = CUDA.ones(Int32, 1) .* 5
        
        # Add evaluation work items
        CUDA.@allowscalar begin
            for i in 1:5
                work_items[1, i] = Int32(WORK_EVALUATE)  # operation
                work_items[2, i] = i  # node_idx
            end
        end
        
        work_queue = WorkQueue(work_items, CUDA.zeros(Int32, 1), CUDA.zeros(Int32, 1), work_size, Int32(10))
        
        # Create batch buffer and config
        buffer = EvalBatchBuffer(Int32(64), MAX_FEATURES, Int32(16))
        config = BatchEvalConfig(Int32(64), true, Int32(256), 0.7f0)
        
        # Test collection kernel
        function test_collect!(tree, work_queue, buffer, config)
            gid = threadIdx().x
            stride = blockDim().x
            
            collect_eval_batch!(tree, work_queue, buffer, config, gid, stride)
            
            return nothing
        end
        
        @cuda threads=32 test_collect!(tree, work_queue, buffer, config)
        
        # Check results
        CUDA.@allowscalar begin
            batch_size = buffer.batch_size_a[1]
            @test batch_size == 5
            
            # Check node indices
            for i in 1:5
                @test buffer.node_indices_a[i] > 0
            end
        end
        
        # Test scatter with dummy scores
        buffer.scores_a .= 0.8f0
        
        function test_scatter!(tree, buffer)
            gid = threadIdx().x
            stride = blockDim().x
            
            scatter_eval_results!(tree, buffer, Int32(0), gid, stride)
            
            return nothing
        end
        
        @cuda threads=32 test_scatter!(tree, buffer)
        
        # Check that scores were updated
        CUDA.@allowscalar begin
            for i in 1:5
                @test tree.prior_scores[i] ≈ 0.8f0
            end
        end
    end
    
    @testset "Batch Evaluation Manager" begin
        manager = BatchEvalManager(
            batch_size = 128,
            num_features = 100,
            max_actions = 16
        )
        
        @test !isnothing(manager.buffer)
        @test !isnothing(manager.config)
        @test !isnothing(manager.eval_stream)
        @test !isnothing(manager.eval_event)
        
        # Test evaluation function
        eval_called = Ref(false)
        
        # Simulate ready batch
        CUDA.@allowscalar begin
            manager.buffer.ready_flag[1] = true
            manager.buffer.batch_size_a[1] = 10
        end
        
        process_eval_batches!(manager) do features, scores, batch_size
            eval_called[] = true
            @test batch_size == 10
            scores .= 0.9f0
        end
        
        # Wait for evaluation
        CUDA.synchronize()
        
        @test eval_called[]
    end
end

println("\n✅ Batch evaluation tests passed!")