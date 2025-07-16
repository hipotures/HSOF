using Test
using CUDA

# Include synchronization module
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/synchronization.jl")

using .MCTSTypes
using .Synchronization

@testset "Simple Synchronization Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping synchronization tests"
        return
    end
    
    @testset "Warp-level Synchronization" begin
        function test_warp_primitives!(results)
            tid = threadIdx().x
            lane_id = (tid - 1) % 32 + 1
            
            # Test ballot
            predicate = lane_id <= 16
            ballot_result = sync_warp_ballot(predicate)
            
            if tid == 1
                results[1] = ballot_result
            end
            
            # Test any
            any_result = sync_warp_any(lane_id == 5)
            if tid == 1
                results[2] = any_result ? 1 : 0
            end
            
            # Test all
            all_result = sync_warp_all(lane_id <= 32)
            if tid == 1
                results[3] = all_result ? 1 : 0
            end
            
            return nothing
        end
        
        results = CUDA.zeros(Int32, 3)
        @cuda threads=32 test_warp_primitives!(results)
        
        results_host = Array(results)
        
        # Ballot should have lower 16 bits set
        @test results_host[1] == 0x0000FFFF
        
        # Any should be true (lane 5 exists)
        @test results_host[2] == 1
        
        # All should be true (all lanes <= 32)
        @test results_host[3] == 1
    end
    
    @testset "Basic Lock Operations" begin
        max_nodes = Int32(10)
        tree_sync = TreeSynchronizer(max_nodes)
        
        # Test direct array access from host
        CUDA.@allowscalar begin
            @test tree_sync.read_locks[1] == 0
            @test tree_sync.write_locks[1] == 0
            @test tree_sync.global_lock[1] == 0
        end
        
        # Simple kernel to test lock functionality
        function test_simple_lock!(read_locks, write_locks, result)
            tid = threadIdx().x
            
            if tid == 1
                # Try to set write lock
                old_val = CUDA.atomic_cas!(pointer(write_locks, 1), Int32(0), Int32(1))
                result[1] = old_val  # Should be 0 (successful)
                
                # Try again - should fail
                old_val2 = CUDA.atomic_cas!(pointer(write_locks, 1), Int32(0), Int32(1))
                result[2] = old_val2  # Should be 1 (failed)
                
                # Release lock
                write_locks[1] = 0
            end
            
            return nothing
        end
        
        result = CUDA.zeros(Int32, 2)
        @cuda threads=32 test_simple_lock!(
            tree_sync.read_locks,
            tree_sync.write_locks,
            result
        )
        
        result_host = Array(result)
        @test result_host[1] == 0  # First CAS succeeded
        @test result_host[2] == 1  # Second CAS failed
    end
    
    @testset "Barrier State Initialization" begin
        num_blocks = Int32(4)
        barrier = GridBarrier(num_blocks)
        
        CUDA.@allowscalar begin
            @test barrier.arrival_count[1] == 0
            @test barrier.generation[1] == 0
            @test barrier.target_count == num_blocks
        end
    end
end

println("\n✅ Simple synchronization tests passed!")