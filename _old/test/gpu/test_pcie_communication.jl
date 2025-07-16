using Test
using CUDA
using Dates

# Include required modules
include("../../src/gpu/GPU.jl")
using .GPU

# Import PCIeCommunication from GPU module (already included there)
using .GPU.PCIeCommunication

@testset "PCIe Communication Tests" begin
    
    @testset "CandidateData Structure" begin
        indices = [1, 5, 10, 15, 20]
        scores = [0.9, 0.85, 0.8, 0.75, 0.7]
        
        candidates = CandidateData(indices, scores, 0, 100)
        
        @test candidates.feature_indices == Int32.(indices)
        @test candidates.scores == Float32.(scores)
        @test candidates.source_gpu == 0
        @test candidates.iteration == 100
        @test isa(candidates.timestamp, DateTime)
        
        # Test type conversion
        indices64 = Int64[100, 200, 300]
        scores64 = Float64[0.99, 0.98, 0.97]
        candidates2 = CandidateData(indices64, scores64, 1, 200)
        @test eltype(candidates2.feature_indices) == Int32
        @test eltype(candidates2.scores) == Float32
    end
    
    @testset "TransferBuffer" begin
        buffer = TransferBuffer(transfer_interval=500, max_candidates=5)
        
        @test buffer.transfer_interval == 500
        @test buffer.max_candidates == 5
        @test isempty(buffer.candidates)
        @test buffer.last_transfer_iteration == 0
    end
    
    @testset "PCIeTransferManager Creation" begin
        # Test with auto-detection
        manager = create_transfer_manager()
        
        num_gpus = CUDA.functional() ? length(CUDA.devices()) : 0
        expected_pairs = num_gpus * (num_gpus - 1)  # Bidirectional pairs
        
        @test length(manager.gpu_pairs) == expected_pairs
        @test manager.use_compression == true
        
        # Test with explicit GPU count
        manager2 = PCIeTransferManager(num_gpus=2, use_compression=false)
        @test length(manager2.gpu_pairs) == 2  # (0,1) and (1,0)
        @test manager2.use_compression == false
        
        # Check buffer initialization
        for pair in manager2.gpu_pairs
            @test haskey(manager2.buffers, pair)
            @test haskey(manager2.stats, pair)
        end
    end
    
    @testset "Candidate Selection" begin
        # Create feature scores
        feature_scores = Dict{Int, Float64}(
            1 => 0.95,
            2 => 0.90,
            3 => 0.85,
            4 => 0.80,
            5 => 0.75,
            6 => 0.70,
            7 => 0.65,
            8 => 0.60,
            9 => 0.55,
            10 => 0.50,
            11 => 0.45,
            12 => 0.40
        )
        
        # Select top 10
        indices, scores = select_top_candidates(feature_scores, 10)
        
        @test length(indices) == 10
        @test indices[1] == 1  # Highest score
        @test scores[1] ≈ 0.95f0
        @test issorted(scores, rev=true)  # Descending order
        
        # Test with threshold
        indices2, scores2 = select_top_candidates(feature_scores, 10, min_score_threshold=0.7)
        @test length(indices2) == 6  # Only 6 features have score >= 0.7
        @test all(s >= 0.7f0 for s in scores2)
        
        # Test with fewer candidates than requested
        small_scores = Dict(1 => 0.9, 2 => 0.8, 3 => 0.7)
        indices3, scores3 = select_top_candidates(small_scores, 10)
        @test length(indices3) == 3
    end
    
    @testset "Buffer Management" begin
        manager = create_transfer_manager()
        
        if length(manager.gpu_pairs) > 0
            src, dst = manager.gpu_pairs[1]
            
            # Add candidates
            candidates = CandidateData([1, 2, 3], [0.9, 0.8, 0.7], src, 100)
            add_candidates!(manager, src, dst, candidates)
            
            buffer = manager.buffers[(src, dst)]
            @test length(buffer.candidates) == 1
            @test buffer.candidates[1].feature_indices == Int32[1, 2, 3]
            
            # Add more candidates
            candidates2 = CandidateData([4, 5, 6], [0.95, 0.85, 0.75], src, 200)
            add_candidates!(manager, src, dst, candidates2)
            @test length(buffer.candidates) == 2
            
            # Test buffer overflow handling
            buffer.max_candidates = 5  # Limit to 5 candidates total
            candidates3 = CandidateData([7, 8, 9, 10], [0.99, 0.89, 0.79, 0.69], src, 300)
            add_candidates!(manager, src, dst, candidates3)
            
            # Should consolidate to top 5
            @test length(buffer.candidates) == 1
            consolidated = buffer.candidates[1]
            @test length(consolidated.feature_indices) == 5
            @test consolidated.scores[1] ≈ 0.99f0  # Highest score
        end
    end
    
    @testset "Transfer Timing" begin
        manager = PCIeTransferManager(num_gpus=2, transfer_interval=1000)
        
        if length(manager.gpu_pairs) > 0
            src, dst = manager.gpu_pairs[1]
            
            # Initially should not transfer
            @test !should_transfer(manager, src, dst, 500)
            
            # Should transfer after interval
            @test should_transfer(manager, src, dst, 1000)
            @test should_transfer(manager, src, dst, 1500)
            
            # After transfer, update last iteration
            buffer = manager.buffers[(src, dst)]
            buffer.last_transfer_iteration = 1000
            
            @test !should_transfer(manager, src, dst, 1500)
            @test should_transfer(manager, src, dst, 2000)
        end
    end
    
    @testset "Compression" begin
        # Test with duplicates
        indices = Int32[1, 2, 3, 1, 2, 4]
        scores = Float32[0.8, 0.9, 0.7, 0.85, 0.95, 0.6]
        data = CandidateData(indices, scores, 0, 100)
        
        compressed = compress_candidates(data)
        
        # Should remove duplicates, keeping highest scores
        @test length(compressed.feature_indices) == 4  # Unique features
        @test 2 in compressed.feature_indices
        
        # Find score for feature 2 (should be 0.95, the higher one)
        idx2_pos = findfirst(x -> x == 2, compressed.feature_indices)
        @test compressed.scores[idx2_pos] ≈ 0.95f0
        
        # Test decompression (currently no-op)
        decompressed = decompress_candidates(compressed)
        @test decompressed.feature_indices == compressed.feature_indices
        @test decompressed.scores == compressed.scores
    end
    
    @testset "Transfer Operations" begin
        manager = PCIeTransferManager(num_gpus=2, transfer_interval=100)
        
        if length(manager.gpu_pairs) > 0
            src, dst = manager.gpu_pairs[1]
            
            # Add candidates
            candidates = CandidateData([10, 20, 30], [0.9, 0.8, 0.7], src, 50)
            add_candidates!(manager, src, dst, candidates)
            
            # Perform transfer
            received = transfer_candidates(manager, src, dst, 100)
            
            @test !isnothing(received)
            @test received.feature_indices == candidates.feature_indices
            @test received.scores == candidates.scores
            @test received.source_gpu == src
            
            # Check stats updated
            stats = manager.stats[(src, dst)]
            @test stats.total_transfers == 1
            @test stats.total_bytes > 0
            @test stats.total_time_ms > 0
            
            # Buffer should be cleared
            buffer = manager.buffers[(src, dst)]
            @test isempty(buffer.candidates)
            @test buffer.last_transfer_iteration == 100
        end
    end
    
    @testset "Peer Access Detection" begin
        manager = create_transfer_manager()
        
        if CUDA.functional() && length(CUDA.devices()) > 1
            # Test P2P availability between GPU 0 and 1
            p2p_available = PCIeCommunication.can_access_peer(manager, 0, 1)
            @test isa(p2p_available, Bool)
            
            # Same GPU should return false
            @test !PCIeCommunication.can_access_peer(manager, 0, 0)
        else
            # No P2P without multiple GPUs
            @test !PCIeCommunication.can_access_peer(manager, 0, 1)
        end
    end
    
    @testset "Transfer Statistics" begin
        manager = PCIeTransferManager(num_gpus=2, transfer_interval=50)
        
        if length(manager.gpu_pairs) > 0
            # Simulate some transfers with more data
            for i in 1:3
                src, dst = manager.gpu_pairs[1]
                # Create more features to ensure measurable transfer size
                indices = collect(1:10000) .+ (i-1)*10000
                scores = rand(Float32, 10000) .* 0.5 .+ 0.5  # Random scores between 0.5 and 1.0
                candidates = CandidateData(indices, scores, src, i*100)
                add_candidates!(manager, src, dst, candidates)
                transfer_candidates(manager, src, dst, i*100)
            end
            
            # Get statistics
            stats = get_transfer_stats(manager)
            
            @test stats["total_transfers"] == 3
            @test stats["total_MB"] > 0
            @test stats["avg_transfer_size_KB"] > 0
            @test haskey(stats, "pair_stats")
            
            # Check pair-specific stats
            pair_key = "GPU0→GPU1"
            if haskey(stats["pair_stats"], pair_key)
                pair_stat = stats["pair_stats"][pair_key]
                @test pair_stat["transfers"] == 3
                @test pair_stat["total_MB"] > 0
                @test pair_stat["avg_time_ms"] > 0
            end
        end
    end
    
    @testset "Broadcast Operations" begin
        manager = PCIeTransferManager(num_gpus=3)  # 3 GPUs for testing broadcast
        
        if length(manager.gpu_pairs) >= 2
            src_gpu = 0
            candidates = CandidateData([100, 200, 300], [0.95, 0.90, 0.85], src_gpu, 1000)
            
            # Broadcast from GPU 0 to others
            received = broadcast_candidates(manager, src_gpu, candidates, 1000)
            
            # Should have transfers to other GPUs
            expected_dsts = [1, 2]
            for dst in expected_dsts
                if haskey(received, dst)
                    @test received[dst].feature_indices == candidates.feature_indices
                    @test received[dst].source_gpu == src_gpu
                end
            end
        end
    end
    
    @testset "Edge Cases" begin
        manager = PCIeTransferManager(num_gpus=1)  # Single GPU
        
        # Should have no pairs
        @test isempty(manager.gpu_pairs)
        @test isempty(manager.buffers)
        @test isempty(manager.stats)
        
        # Test empty transfer
        manager2 = PCIeTransferManager(num_gpus=2)
        if length(manager2.gpu_pairs) > 0
            src, dst = manager2.gpu_pairs[1]
            
            # Transfer with empty buffer
            result = transfer_candidates(manager2, src, dst, 100)
            @test isnothing(result)
            
            # Stats should not change
            stats = manager2.stats[(src, dst)]
            @test stats.total_transfers == 0
        end
        
        # Test reset buffer
        if length(manager2.gpu_pairs) > 0
            src, dst = manager2.gpu_pairs[1]
            candidates = CandidateData([1, 2], [0.9, 0.8], src, 100)
            add_candidates!(manager2, src, dst, candidates)
            
            reset_buffer!(manager2, src, dst)
            buffer = manager2.buffers[(src, dst)]
            @test isempty(buffer.candidates)
            @test buffer.last_transfer_iteration == 0
        end
    end
    
end

# Print summary
println("\nPCIe Communication Test Summary:")
println("================================")
if CUDA.functional()
    num_gpus = length(CUDA.devices())
    println("✓ CUDA functional - GPU tests executed")
    println("  GPUs detected: $num_gpus")
    if num_gpus > 1
        println("  Multi-GPU transfers tested")
    else
        println("  Single GPU - P2P tests skipped")
    end
else
    println("⚠ CUDA not functional - GPU tests limited")
end
println("\nAll PCIe communication tests completed!")