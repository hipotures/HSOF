using Test
using CUDA

# Include synchronization module
include("../../src/gpu/kernels/mcts_types.jl")
include("../../src/gpu/kernels/synchronization.jl")

using .MCTSTypes
using .Synchronization

@testset "Synchronization Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping synchronization tests"
        return
    end
    
    @testset "Grid Barrier" begin
        num_blocks = Int32(32)
        barrier = GridBarrier(num_blocks)
        
        # Test initial state
        CUDA.@allowscalar begin
            @test barrier.arrival_count[1] == 0
            @test barrier.release_count[1] == 0
            @test barrier.generation[1] == 0
        end
        
        # Test barrier sync kernel
        function test_grid_barrier_kernel!(barrier, results)
            bid = blockIdx().x
            tid = threadIdx().x
            
            # Do some work
            if tid == 1
                results[bid] = bid
            end
            
            # Synchronize
            grid_barrier_sync!(barrier, bid, tid)
            
            # Check all blocks have written
            if tid == 1
                sum = 0
                for i in 1:gridDim().x
                    sum += results[i]
                end
                results[bid] = sum
            end
            
            return nothing
        end
        
        results = CUDA.zeros(Int32, num_blocks)
        @cuda threads=32 blocks=num_blocks test_grid_barrier_kernel!(barrier, results)
        
        # All blocks should see the same sum
        expected_sum = sum(1:num_blocks)
        results_host = Array(results)
        @test all(r -> r == expected_sum, results_host)
    end
    
    @testset "Phase Synchronizer" begin
        num_blocks = Int32(16)
        phase_sync = PhaseSynchronizer(num_blocks)
        
        # Test initial state
        CUDA.@allowscalar begin
            @test phase_sync.current_phase[1] == 0
            @test all(phase_sync.phase_counters .== 0)
        end
        
        # Test phase transitions
        function test_phase_sync_kernel!(phase_sync, phase_log)
            bid = blockIdx().x
            tid = threadIdx().x
            
            # Log initial phase
            if tid == 1
                phase_log[1, bid] = phase_sync.current_phase[1]
            end
            
            # Phase 0 -> 1
            phase_barrier_sync!(phase_sync, UInt8(0), bid, tid)
            
            if tid == 1
                phase_log[2, bid] = phase_sync.current_phase[1]
            end
            
            # Phase 1 -> 2
            phase_barrier_sync!(phase_sync, UInt8(1), bid, tid)
            
            if tid == 1
                phase_log[3, bid] = phase_sync.current_phase[1]
            end
            
            return nothing
        end
        
        phase_log = CUDA.fill(UInt8(255), 3, num_blocks)
        @cuda threads=32 blocks=num_blocks test_phase_sync_kernel!(phase_sync, phase_log)
        
        phase_log_host = Array(phase_log)
        
        # All blocks should see phase 0 initially
        @test all(phase_log_host[1, :] .== 0)
        
        # All blocks should transition to phase 1
        @test all(phase_log_host[2, :] .== 1)
        
        # All blocks should transition to phase 2
        @test all(phase_log_host[3, :] .== 2)
    end
    
    @testset "Tree Synchronizer - Read/Write Locks" begin
        max_nodes = Int32(100)
        tree_sync = TreeSynchronizer(max_nodes)
        
        # Test concurrent reads
        function test_concurrent_reads!(tree_sync, read_counts)
            tid = threadIdx().x
            node_idx = Int32(1)
            
            # Multiple threads acquire read lock
            acquire_read_lock!(tree_sync, node_idx)
            
            # Increment counter (should work concurrently)
            CUDA.atomic_add!(pointer(read_counts), Int32(1))
            
            # Small delay
            for i in 1:100
                CUDA.sync_warp()
            end
            
            release_read_lock!(tree_sync, node_idx)
            
            return nothing
        end
        
        read_counts = CUDA.zeros(Int32, 1)
        @cuda threads=256 test_concurrent_reads!(tree_sync, read_counts)
        
        CUDA.@allowscalar @test read_counts[1] == 256
        
        # Test write lock exclusivity
        function test_write_exclusivity!(tree_sync, write_log, write_counter)
            tid = threadIdx().x
            node_idx = Int32(1)
            
            # Try to acquire write lock
            acquire_write_lock!(tree_sync, node_idx)
            
            # Only one thread should be here at a time
            old_val = CUDA.atomic_add!(pointer(write_counter), Int32(1))
            write_log[tid] = old_val
            
            # Delay to ensure overlap attempts
            for i in 1:1000
                CUDA.sync_warp()
            end
            
            CUDA.atomic_sub!(pointer(write_counter), Int32(1))
            
            release_write_lock!(tree_sync, node_idx)
            
            return nothing
        end
        
        write_log = CUDA.fill(Int32(-1), 32)
        write_counter = CUDA.zeros(Int32, 1)
        @cuda threads=32 test_write_exclusivity!(tree_sync, write_log, write_counter)
        
        write_log_host = Array(write_log)
        
        # Each thread should see counter value of 0 (exclusive access)
        @test all(val -> val == 0 || val == -1, write_log_host)
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
end

println("\n✅ Synchronization tests passed!")