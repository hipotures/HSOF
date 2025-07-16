using Test
using CUDA

# Include modules
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/warp_optimization.jl")

using .MCTSTypes
using .WarpOptimization

@testset "Warp Optimization Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping warp optimization tests"
        return
    end
    
    @testset "Warp Efficiency Computation" begin
        # Test full warp
        efficiency1 = compute_warp_efficiency(0xFFFFFFFF, Int32(32))
        @test efficiency1 ≈ 1.0f0
        
        # Test half warp
        efficiency2 = compute_warp_efficiency(0x0000FFFF, Int32(32))
        @test efficiency2 ≈ 0.5f0
        
        # Test single thread
        efficiency3 = compute_warp_efficiency(0x00000001, Int32(32))
        @test efficiency3 ≈ 1.0f0/32.0f0
    end
    
    @testset "Predicated Selection" begin
        # Test predicated select
        val1 = predicated_select(true, 10, 20)
        @test val1 == 10
        
        val2 = predicated_select(false, 10, 20)
        @test val2 == 20
    end
    
    @testset "Warp Scheduler Creation" begin
        num_warps = Int32(4)
        scheduler = WarpScheduler(
            CUDA.zeros(Int32, WARP_SIZE, num_warps),
            CUDA.zeros(Int32, num_warps),
            CUDA.zeros(Float32, num_warps),
            num_warps
        )
        
        @test scheduler.num_warps == 4
        @test size(scheduler.warp_work_assignments) == (32, 4)
        
        # Check initial state
        CUDA.@allowscalar begin
            @test all(scheduler.warp_depths .== 0)
            @test all(scheduler.warp_occupancy .== 0)
        end
    end
    
    @testset "Node Sorter" begin
        max_depth = Int32(10)
        num_nodes = Int32(100)
        
        sorter = NodeSorter(
            CUDA.zeros(Int32, num_nodes),
            CUDA.zeros(Int32, num_nodes),
            CUDA.zeros(Int32, max_depth),
            CUDA.zeros(Int32, max_depth),
            max_depth
        )
        
        @test length(sorter.sorted_indices) == num_nodes
        @test length(sorter.bucket_starts) == max_depth
    end
    
    @testset "Divergence Tracker" begin
        num_warps = Int32(8)
        num_branches = Int32(5)
        
        tracker = DivergenceTracker(
            CUDA.zeros(Int32, num_warps),
            CUDA.zeros(Int32, num_branches, num_warps),
            CUDA.zeros(UInt32, num_warps)
        )
        
        @test size(tracker.divergence_counters) == (num_warps,)
        @test size(tracker.branch_counters) == (num_branches, num_warps)
        
        # Test divergence tracking kernel
        function test_divergence_kernel!(tracker)
            tid = threadIdx().x
            wid = (tid - 1) ÷ WARP_SIZE + 1
            
            # Create divergence: even threads take branch 0, odd take branch 1
            branch = tid % 2
            track_divergence!(tracker, wid, branch, true)
            
            return nothing
        end
        
        # Run with 64 threads (2 warps)
        @cuda threads=64 test_divergence_kernel!(tracker)
        
        # Check that divergence was detected
        divergence_counts = Array(tracker.divergence_counters)
        @test divergence_counts[1] > 0  # First warp had divergence
        @test divergence_counts[2] > 0  # Second warp had divergence
    end
    
    @testset "Warp Convergence Check" begin
        function test_convergence_kernel!(results)
            tid = threadIdx().x
            
            # Test 1: All threads have same predicate
            converged1 = warp_converged_loop(true, 1)
            
            # Test 2: Threads have different predicates
            converged2 = warp_converged_loop(tid % 2 == 0, 1)
            
            if tid == 1
                results[1] = converged1 ? 1 : 0
                results[2] = converged2 ? 1 : 0
            end
            
            return nothing
        end
        
        results = CUDA.zeros(Int32, 2)
        @cuda threads=32 test_convergence_kernel!(results)
        
        results_host = Array(results)
        @test results_host[1] == 1  # All threads converged
        @test results_host[2] == 0  # Threads diverged
    end
    
    @testset "Process Node Coherent" begin
        # Create simple tree
        tree = MCTSTreeSoA(CUDA.device())
        config = PersistentKernelConfig()
        
        # Setup test node with children
        CUDA.@allowscalar begin
            # Node 1 with 4 children
            tree.node_ids[1] = 1
            tree.node_states[1] = NODE_EXPANDED
            tree.num_children[1] = 4
            tree.first_child_idx[1] = 2
            tree.visit_counts[1] = 100
            
            # Children nodes
            for i in 2:5
                tree.node_ids[i] = i
                tree.node_states[i] = NODE_ACTIVE
                tree.visit_counts[i] = 10 + i
                tree.total_scores[i] = Float32(i * 0.5)
                tree.prior_scores[i] = 0.1f0
            end
        end
        
        # Test coherent processing
        function test_coherent_kernel!(tree, config, results)
            tid = threadIdx().x
            lane = (tid - 1) % WARP_SIZE + 1
            
            best_child = process_node_coherent!(tree, Int32(1), lane, config)
            
            # All threads should get same result
            results[tid] = best_child
            
            return nothing
        end
        
        results = CUDA.fill(Int32(-1), 32)
        @cuda threads=32 test_coherent_kernel!(tree, config, results)
        
        results_host = Array(results)
        
        # All threads should agree on best child
        @test all(results_host .== results_host[1])
        @test results_host[1] > 0  # Should find a valid child
    end
end

println("\n✅ Warp optimization tests passed!")