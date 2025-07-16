using Test
using CUDA
using Random

# Include the backup propagation module
include("../../src/gpu/kernels/backup_propagation.jl")
include("../../src/gpu/kernels/mcts_types.jl")

using .BackupPropagation
using .MCTSTypes

# Helper function to create a simple tree structure
function create_test_tree(num_nodes::Int)
    parent_ids = CUDA.zeros(Int32, MCTSTypes.MAX_NODES)
    total_scores = CUDA.zeros(Float32, MCTSTypes.MAX_NODES)
    visit_counts = CUDA.zeros(Int32, MCTSTypes.MAX_NODES)
    
    # Create a simple tree structure
    CUDA.@allowscalar begin
        # Root node
        parent_ids[1] = -1
        visit_counts[1] = 100
        total_scores[1] = 50.0f0
        
        # Create additional nodes
        for i in 2:num_nodes
            # Simple parent assignment (binary tree-like)
            parent_ids[i] = Int32(i ÷ 2)
            visit_counts[i] = max(1, 100 - i * 10)
            total_scores[i] = Float32(visit_counts[i]) * 0.5f0
        end
    end
    
    return parent_ids, total_scores, visit_counts
end

@testset "Backup Propagation Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU backup propagation tests"
        return
    end
    
    @testset "BackupConfig Creation" begin
        config = BackupConfig(
            1.0f0,      # virtual_loss
            32,         # backup_batch_size
            100,        # max_path_length
            0.001f0,    # convergence_threshold
            0.9f0,      # damping_factor
            true        # use_path_compression
        )
        
        @test config.virtual_loss == 1.0f0
        @test config.backup_batch_size == 32
        @test config.max_path_length == 100
        @test config.convergence_threshold == 0.001f0
        @test config.damping_factor == 0.9f0
        @test config.use_path_compression == true
    end
    
    @testset "BackupBuffer Creation" begin
        config = BackupConfig(1.0f0, 16, 50, 0.001f0, 0.9f0, false)
        buffer = BackupBuffer(config)
        
        @test size(buffer.paths) == (50, 16)
        @test length(buffer.path_lengths) == 16
        @test length(buffer.leaf_values) == 16
        @test length(buffer.virtual_losses) == MCTSTypes.MAX_NODES
        
        # Check initial state
        CUDA.@allowscalar begin
            @test buffer.total_backups[1] == 0
            @test buffer.avg_path_length[1] == 0.0f0
            @test buffer.convergence_rate[1] == 0.0f0
        end
    end
    
    @testset "Path Tracing" begin
        config = BackupConfig(1.0f0, 8, 20, 0.001f0, 0.9f0, false)
        buffer = BackupBuffer(config)
        
        # Create simple tree
        parent_ids, _, _ = create_test_tree(10)
        
        # Test tracing paths from various leaves
        leaf_indices = CuArray(Int32[5, 7, 9])  # Some leaf nodes
        batch_size = Int32(3)
        
        @cuda threads=32 BackupPropagation.trace_path_kernel!(
            parent_ids,
            buffer.paths,
            buffer.path_lengths,
            leaf_indices,
            batch_size,
            config.max_path_length
        )
        
        # Verify paths
        CUDA.@allowscalar begin
            # Check path from node 5: 5 -> 2 -> 1
            @test buffer.path_lengths[1] == 3
            @test buffer.paths[1, 1] == 5
            @test buffer.paths[2, 1] == 2
            @test buffer.paths[3, 1] == 1
            
            # Check path from node 7: 7 -> 3 -> 1
            @test buffer.path_lengths[2] == 3
            @test buffer.paths[1, 2] == 7
            @test buffer.paths[2, 2] == 3
            @test buffer.paths[3, 2] == 1
        end
    end
    
    @testset "Virtual Loss Application" begin
        config = BackupConfig(2.0f0, 4, 10, 0.001f0, 0.9f0, false)
        buffer = BackupBuffer(config)
        
        parent_ids, _, visit_counts = create_test_tree(5)
        
        # Create paths manually for testing
        CUDA.@allowscalar begin
            # Path 1: 4 -> 2 -> 1
            buffer.paths[1, 1] = 4
            buffer.paths[2, 1] = 2
            buffer.paths[3, 1] = 1
            buffer.path_lengths[1] = 3
            
            # Path 2: 5 -> 2 -> 1
            buffer.paths[1, 2] = 5
            buffer.paths[2, 2] = 2
            buffer.paths[3, 2] = 1
            buffer.path_lengths[2] = 3
        end
        
        # Store original visit counts
        original_visits = Array(visit_counts)
        
        # Apply virtual loss
        @cuda threads=32 BackupPropagation.apply_virtual_loss_kernel!(
            buffer.virtual_losses,
            visit_counts,
            buffer.paths,
            buffer.path_lengths,
            config.virtual_loss,
            Int32(2)
        )
        
        # Check virtual losses and visit counts
        CUDA.@allowscalar begin
            # Node 1 should have virtual loss from both paths
            @test buffer.virtual_losses[1] == 4.0f0  # 2 paths * 2.0 virtual loss
            # Node 2 should have virtual loss from both paths
            @test buffer.virtual_losses[2] == 4.0f0
            # Nodes 4 and 5 should have virtual loss from one path each
            @test buffer.virtual_losses[4] == 2.0f0
            @test buffer.virtual_losses[5] == 2.0f0
            
            # Visit counts should be incremented
            @test visit_counts[1] == original_visits[1] + 2
            @test visit_counts[2] == original_visits[2] + 2
            @test visit_counts[4] == original_visits[4] + 1
            @test visit_counts[5] == original_visits[5] + 1
        end
        
        # Remove virtual loss
        @cuda threads=32 BackupPropagation.remove_virtual_loss_kernel!(
            buffer.virtual_losses,
            visit_counts,
            buffer.paths,
            buffer.path_lengths,
            config.virtual_loss,
            Int32(2)
        )
        
        # Check virtual losses are removed
        CUDA.@allowscalar begin
            @test buffer.virtual_losses[1] == 0.0f0
            @test buffer.virtual_losses[2] == 0.0f0
            @test buffer.virtual_losses[4] == 0.0f0
            @test buffer.virtual_losses[5] == 0.0f0
            
            # Visit counts should remain incremented
            @test visit_counts[1] == original_visits[1] + 2
        end
    end
    
    @testset "Value Backup" begin
        config = BackupConfig(1.0f0, 4, 10, 0.001f0, 0.8f0, false)
        buffer = BackupBuffer(config)
        
        parent_ids, total_scores, visit_counts = create_test_tree(5)
        
        # Setup paths and leaf values
        CUDA.@allowscalar begin
            # Path: 4 -> 2 -> 1
            buffer.paths[1, 1] = 4
            buffer.paths[2, 1] = 2
            buffer.paths[3, 1] = 1
            buffer.path_lengths[1] = 3
            buffer.leaf_values[1] = 0.9f0  # High value at leaf
        end
        
        # Store original scores
        original_scores = Array(total_scores)
        
        # Backup values
        @cuda threads=32 BackupPropagation.backup_values_kernel!(
            total_scores,
            visit_counts,
            buffer.value_deltas,
            buffer.paths,
            buffer.path_lengths,
            buffer.leaf_values,
            config.damping_factor,
            Int32(1)
        )
        
        # Check that values were propagated
        CUDA.@allowscalar begin
            # All nodes in path should have increased scores
            @test total_scores[4] > original_scores[4]
            @test total_scores[2] > original_scores[2]
            @test total_scores[1] > original_scores[1]
            
            # Value deltas should be recorded
            @test buffer.value_deltas[4] > 0.0f0
            @test buffer.value_deltas[2] > 0.0f0
            @test buffer.value_deltas[1] > 0.0f0
        end
    end
    
    @testset "Path Compression" begin
        config = BackupConfig(1.0f0, 2, 10, 0.001f0, 0.9f0, true)
        buffer = BackupBuffer(config)
        
        parent_ids, _, visit_counts = create_test_tree(10)
        
        # Create a long path with varying visit counts
        CUDA.@allowscalar begin
            # Path: 8 -> 4 -> 2 -> 1
            buffer.paths[1, 1] = 8
            buffer.paths[2, 1] = 4
            buffer.paths[3, 1] = 2
            buffer.paths[4, 1] = 1
            buffer.path_lengths[1] = 4
            
            # Set visit counts: make node 4 have low visits
            visit_counts[8] = 100
            visit_counts[4] = 5   # Below threshold
            visit_counts[2] = 50
            visit_counts[1] = 100
        end
        
        # Apply path compression
        compression_threshold = Int32(10)
        @cuda threads=32 BackupPropagation.compress_paths_kernel!(
            parent_ids,
            visit_counts,
            buffer.paths,
            buffer.path_lengths,
            Int32(1),
            compression_threshold
        )
        
        # Check compressed path
        CUDA.@allowscalar begin
            # Path should skip node 4 (low visits)
            @test buffer.path_lengths[1] == 3
            @test buffer.paths[1, 1] == 8  # Leaf kept
            @test buffer.paths[2, 1] == 2  # Node 4 skipped
            @test buffer.paths[3, 1] == 1  # Root kept
        end
    end
    
    @testset "Convergence Detection" begin
        config = BackupConfig(1.0f0, 4, 10, 0.01f0, 0.9f0, false)
        buffer = BackupBuffer(config)
        
        _, _, visit_counts = create_test_tree(5)
        
        # Set up value deltas and visit counts
        CUDA.@allowscalar begin
            # Node 1: high visits, small delta -> should converge
            visit_counts[1] = 200
            buffer.value_deltas[1] = 0.005f0
            
            # Node 2: high visits, large delta -> should not converge
            visit_counts[2] = 150
            buffer.value_deltas[2] = 0.05f0
            
            # Node 3: low visits, small delta -> should not converge
            visit_counts[3] = 50
            buffer.value_deltas[3] = 0.005f0
        end
        
        # Detect convergence
        min_visits = Int32(100)
        @cuda threads=32 BackupPropagation.detect_convergence_kernel!(
            buffer.convergence_flags,
            buffer.value_deltas,
            visit_counts,
            config.convergence_threshold,
            min_visits
        )
        
        # Check convergence flags
        CUDA.@allowscalar begin
            @test buffer.convergence_flags[1] == true   # Converged
            @test buffer.convergence_flags[2] == false  # Large delta
            @test buffer.convergence_flags[3] == false  # Low visits
            
            # Deltas should be reset
            @test buffer.value_deltas[1] == 0.0f0
            @test buffer.value_deltas[2] == 0.0f0
            @test buffer.value_deltas[3] == 0.0f0
        end
    end
    
    @testset "Batch Backup Integration" begin
        config = BackupConfig(1.0f0, 8, 20, 0.001f0, 0.9f0, false)
        buffer = BackupBuffer(config)
        
        parent_ids, total_scores, visit_counts = create_test_tree(10)
        
        # Prepare batch of leaves
        leaf_indices = CuArray(Int32[5, 7, 8, 9])
        leaf_values = CuArray([0.8f0, 0.7f0, 0.9f0, 0.6f0])
        batch_size = Int32(4)
        
        # Perform batch backup
        backup_batch!(
            buffer, config,
            parent_ids, total_scores, visit_counts,
            leaf_indices, leaf_values, batch_size
        )
        
        # Check statistics
        stats = get_backup_stats(buffer)
        @test stats["total_backups"] == 4
        @test stats["avg_path_length"] > 0.0
        @test haskey(stats, "converged_nodes")
        @test haskey(stats, "max_virtual_loss")
        
        # Verify virtual losses were cleaned up
        CUDA.@allowscalar begin
            @test maximum(buffer.virtual_losses) == 0.0f0
        end
    end
    
    @testset "Statistics and Reset" begin
        config = BackupConfig(1.0f0, 4, 10, 0.001f0, 0.9f0, false)
        buffer = BackupBuffer(config)
        
        # Set some convergence data
        CUDA.@allowscalar begin
            buffer.convergence_flags[1] = true
            buffer.convergence_flags[2] = true
            buffer.convergence_rate[1] = 0.5f0
            buffer.value_deltas[1] = 0.1f0
        end
        
        # Reset convergence
        reset_convergence!(buffer)
        
        # Check reset
        CUDA.@allowscalar begin
            @test all(.!buffer.convergence_flags)
            @test all(buffer.value_deltas .== 0.0f0)
            @test buffer.convergence_rate[1] == 0.0f0
        end
    end
end

println("\n✅ Backup propagation tests completed!")