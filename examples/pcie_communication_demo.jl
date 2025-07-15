#!/usr/bin/env julia

# PCIe Communication Demo
# Demonstrates GPU-to-GPU candidate transfer patterns

using CUDA
using Printf
using Dates

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU
using .GPU.PCIeCommunication

function demo_pcie_communication()
    println("PCIe Communication Demo")
    println("=" ^ 60)
    
    # Create transfer manager
    manager = create_transfer_manager(
        num_gpus = -1,  # Auto-detect
        use_compression = true,
        transfer_interval = 1000
    )
    
    println("\nTransfer Manager Configuration:")
    println("  Number of GPUs: $(size(manager.peer_access_matrix, 1))")
    println("  GPU pairs configured: $(length(manager.gpu_pairs))")
    println("  Compression enabled: $(manager.use_compression)")
    
    # Demo 1: Basic Candidate Selection
    println("\n1. Candidate Selection Demo:")
    println("-" ^ 40)
    
    # Simulate feature scores from MCTS
    feature_scores = Dict{Int, Float64}()
    for i in 1:100
        feature_scores[i] = rand() * 0.5 + 0.5  # Scores between 0.5 and 1.0
    end
    
    # Select top candidates
    indices, scores = select_top_candidates(feature_scores, 10, min_score_threshold=0.7)
    println("Selected $(length(indices)) top candidates above threshold 0.7:")
    for (idx, score) in zip(indices[1:min(5, length(indices))], scores[1:min(5, length(scores))])
        println("  Feature $idx: score = $(round(score, digits=3))")
    end
    if length(indices) > 5
        println("  ... and $(length(indices) - 5) more")
    end
    
    # Demo 2: P2P Access Detection
    println("\n2. Peer-to-Peer Access Detection:")
    println("-" ^ 40)
    
    if CUDA.functional()
        for (src, dst) in manager.gpu_pairs
            has_p2p = can_access_peer(manager, src, dst)
            println("  GPU $src → GPU $dst: P2P = $has_p2p")
        end
    else
        println("  CUDA not functional - P2P detection skipped")
    end
    
    # Demo 3: Simulated Transfer Workflow
    println("\n3. Simulated Transfer Workflow:")
    println("-" ^ 40)
    
    if length(manager.gpu_pairs) > 0
        # Simulate multiple iterations
        for iteration in [500, 1000, 1500, 2000, 2500]
            println("\nIteration $iteration:")
            
            # GPU 0 finds candidates
            if iteration % 500 == 0
                # Generate new candidates every 500 iterations
                n_candidates = rand(5:15)
                candidate_indices = rand(1:1000, n_candidates)
                candidate_scores = rand(Float32, n_candidates) .* 0.4 .+ 0.6
                
                candidates = CandidateData(candidate_indices, candidate_scores, 0, iteration)
                
                # Add to transfer buffers for all other GPUs
                for (src, dst) in manager.gpu_pairs
                    if src == 0
                        add_candidates!(manager, src, dst, candidates)
                        
                        # Check if should transfer
                        if should_transfer(manager, src, dst, iteration)
                            println("  Transferring candidates from GPU $src to GPU $dst...")
                            
                            start_time = time()
                            received = transfer_candidates(manager, src, dst, iteration)
                            transfer_time = (time() - start_time) * 1000
                            
                            if !isnothing(received)
                                println("    Transferred $(length(received.feature_indices)) features in $(round(transfer_time, digits=2))ms")
                            end
                        else
                            buffer = manager.buffers[(src, dst)]
                            next_transfer = buffer.last_transfer_iteration + buffer.transfer_interval
                            println("  Buffering candidates for GPU $src→$dst (next transfer at iteration $next_transfer)")
                        end
                    end
                end
            end
        end
    end
    
    # Demo 4: Transfer Statistics
    println("\n4. Transfer Statistics:")
    println("-" ^ 40)
    
    stats = get_transfer_stats(manager)
    println("Summary:")
    println("  Total transfers: $(stats["total_transfers"])")
    println("  Total data transferred: $(stats["total_MB"]) MB")
    if stats["total_transfers"] > 0
        println("  Average transfer size: $(stats["avg_transfer_size_KB"]) KB")
        println("  Average bandwidth: $(stats["avg_bandwidth_MB/s"]) MB/s")
    end
    
    println("\nPer-GPU-pair statistics:")
    for (pair, pair_stats) in stats["pair_stats"]
        if pair_stats["transfers"] > 0
            println("  $pair:")
            println("    Transfers: $(pair_stats["transfers"])")
            println("    Total: $(pair_stats["total_MB"]) MB")
            println("    Avg time: $(pair_stats["avg_time_ms"]) ms")
            println("    Bandwidth: $(pair_stats["bandwidth_MB/s"]) MB/s")
            println("    P2P enabled: $(pair_stats["p2p_enabled"])")
        end
    end
    
    # Demo 5: Broadcast Operation
    println("\n5. Broadcast Operation Demo:")
    println("-" ^ 40)
    
    if size(manager.peer_access_matrix, 1) > 1
        # Create important candidates to broadcast
        important_indices = [10, 20, 30, 40, 50]
        important_scores = Float32[0.99, 0.98, 0.97, 0.96, 0.95]
        broadcast_data = CandidateData(important_indices, important_scores, 0, 3000)
        
        println("Broadcasting high-priority candidates from GPU 0 to all other GPUs...")
        received_map = broadcast_candidates(manager, 0, broadcast_data, 3000)
        
        for (dst_gpu, received) in received_map
            println("  GPU $dst_gpu received $(length(received.feature_indices)) candidates")
        end
    else
        println("  Single GPU system - broadcast demo skipped")
    end
    
    # Demo 6: Compression Efficiency
    println("\n6. Compression Efficiency Demo:")
    println("-" ^ 40)
    
    # Create data with duplicates
    dup_indices = Int32[1, 2, 3, 1, 2, 4, 5, 3, 6, 7]
    dup_scores = Float32[0.9, 0.8, 0.7, 0.85, 0.95, 0.6, 0.5, 0.75, 0.4, 0.3]
    dup_data = CandidateData(dup_indices, dup_scores, 0, 4000)
    
    println("Original data: $(length(dup_indices)) entries")
    compressed = compress_candidates(dup_data)
    println("Compressed data: $(length(compressed.feature_indices)) unique entries")
    compression_ratio = length(compressed.feature_indices) / length(dup_indices)
    println("Compression ratio: $(round(compression_ratio * 100, digits=1))%")
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

# Helper function to display peer access matrix
function display_peer_matrix(manager::PCIeTransferManager)
    n = size(manager.peer_access_matrix, 1)
    println("\nPeer Access Matrix:")
    print("     ")
    for j in 0:n-1
        print("GPU$j ")
    end
    println()
    
    for i in 0:n-1
        print("GPU$i ")
        for j in 0:n-1
            if i == j
                print("  -  ")
            else
                print(manager.peer_access_matrix[i+1, j+1] ? "  ✓  " : "  ✗  ")
            end
        end
        println()
    end
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_pcie_communication()
end