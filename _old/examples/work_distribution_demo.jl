#!/usr/bin/env julia

# Work Distribution Demo
# Demonstrates GPU work distribution patterns for dual-GPU setup

using CUDA
using Printf
using Dates

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules directly
include("../src/gpu/GPU.jl")
using .GPU

function demo_work_distribution()
    println("GPU Work Distribution Demo")
    println("=" ^ 60)
    
    # Create work distributor
    distributor = create_work_distributor(
        total_trees = 100,
        rebalance_threshold = 0.05
    )
    
    # Display initial configuration
    println("\nInitial Work Distribution:")
    summary = get_work_summary(distributor)
    display_summary(summary)
    
    # Demo 1: Tree Assignment
    println("\n1. Tree Assignment Demo:")
    println("-" ^ 40)
    
    # Single tree queries
    for tree_id in [1, 25, 50, 51, 75, 100]
        gpu_id = get_gpu_for_tree(distributor, tree_id)
        println("Tree $tree_id → GPU $gpu_id")
    end
    
    # Batch assignment
    println("\nBatch assignment for trees 45-55:")
    batch_trees = collect(45:55)
    assignments = assign_tree_work(distributor, batch_trees)
    for (gpu_id, trees) in assignments
        println("  GPU $gpu_id: trees $(minimum(trees))-$(maximum(trees)) ($(length(trees)) trees)")
    end
    
    # Demo 2: Metamodel Work Assignment
    println("\n2. Metamodel Work Assignment:")
    println("-" ^ 40)
    
    training_gpu = assign_metamodel_work(distributor, :training)
    inference_gpu = assign_metamodel_work(distributor, :inference)
    
    println("Metamodel training → GPU $training_gpu")
    println("Metamodel inference → GPU $inference_gpu")
    
    # Demo 3: Simulated Workload
    println("\n3. Simulated Workload Execution:")
    println("-" ^ 40)
    
    # Simulate processing on each GPU
    for epoch in 1:3
        println("\nEpoch $epoch:")
        
        # Process trees on each GPU
        for (gpu_id, assignment) in distributor.gpu_assignments
            trees_to_process = rand(assignment.tree_range, 10)  # Random subset
            
            # Simulate work with timing
            start_time = time()
            
            # Execute on assigned GPU
            result = execute_on_gpu(gpu_id, simulate_tree_processing, trees_to_process)
            
            elapsed_ms = (time() - start_time) * 1000
            
            # Update metrics
            update_metrics!(distributor, gpu_id,
                trees_processed = length(trees_to_process),
                time_ms = elapsed_ms,
                utilization = 50.0 + rand() * 40.0,  # 50-90%
                memory_mb = 2000.0 + rand() * 2000.0  # 2-4GB
            )
            
            println("  GPU $gpu_id: processed $(length(trees_to_process)) trees in $(round(elapsed_ms, digits=1))ms")
        end
        
        # Simulate metamodel operations
        if distributor.num_gpus > 1
            # Training on GPU 0
            execute_on_gpu(0, simulate_metamodel_training)
            update_metrics!(distributor, 0, metamodel_operations = 1)
            
            # Inference on GPU 1
            execute_on_gpu(1, simulate_metamodel_inference)
            update_metrics!(distributor, 1, metamodel_operations = 1)
        end
    end
    
    # Display updated metrics
    println("\nUpdated Work Distribution:")
    summary = get_work_summary(distributor)
    display_summary(summary)
    
    # Demo 4: Load Balancing
    println("\n4. Load Balancing Demo:")
    println("-" ^ 40)
    
    balance_ratio = get_load_balance_ratio(distributor)
    println("Current load balance ratio: $(round(balance_ratio, digits=3))")
    
    if distributor.num_gpus > 1
        # Create imbalance
        println("\nCreating workload imbalance...")
        update_metrics!(distributor, 0, 
            trees_processed = 100,
            total_time_ms = 1000.0
        )
        update_metrics!(distributor, 1, 
            trees_processed = 50,
            total_time_ms = 1000.0
        )
        
        new_ratio = get_load_balance_ratio(distributor)
        println("Load balance ratio after imbalance: $(round(new_ratio, digits=3))")
        
        # Check if rebalancing is needed
        if rebalance_if_needed!(distributor)
            println("Rebalancing triggered!")
            println("\nAfter rebalancing:")
            summary = get_work_summary(distributor)
            display_summary(summary)
        else
            println("No rebalancing needed (threshold: $(distributor.rebalance_threshold))")
        end
    end
    
    # Demo 5: GPU Affinity
    println("\n5. GPU Affinity Demo:")
    println("-" ^ 40)
    
    if CUDA.functional()
        original_device = CUDA.device()
        println("Original GPU: $(CUDA.name(original_device))")
        
        # Switch to different GPUs
        for gpu_id in 0:(distributor.num_gpus-1)
            set_gpu_affinity(gpu_id)
            current = CUDA.device()
            println("After set_gpu_affinity($gpu_id): $(CUDA.name(current))")
        end
        
        # Restore original
        CUDA.device!(original_device)
    else
        println("CUDA not functional - skipping affinity demo")
    end
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

# Helper function to display work summary
function display_summary(summary)
    println("  Total trees: $(summary["total_trees"])")
    println("  Number of GPUs: $(summary["num_gpus"])")
    println("  Load balance ratio: $(summary["load_balance_ratio"])")
    
    for (gpu_id, gpu_info) in summary["assignments"]
        println("\n  GPU $gpu_id:")
        println("    Tree range: $(gpu_info["tree_range"]) ($(gpu_info["tree_count"]) trees)")
        println("    Metamodel role: $(gpu_info["metamodel_role"])")
        println("    Primary GPU: $(gpu_info["is_primary"])")
        
        if haskey(gpu_info, "trees_processed")
            println("    Trees processed: $(gpu_info["trees_processed"])")
            println("    Metamodel ops: $(gpu_info["metamodel_ops"])")
            println("    Utilization: $(gpu_info["utilization"])%")
            println("    Memory: $(gpu_info["memory_mb"]) MB")
        end
    end
end

# Simulation functions
function simulate_tree_processing(tree_indices)
    # Simulate MCTS tree processing
    if CUDA.functional()
        # Allocate some GPU memory
        data = CUDA.randn(1000, length(tree_indices))
        result = sum(data, dims=1)
        return Array(result)
    else
        # CPU fallback
        return randn(length(tree_indices))
    end
end

function simulate_metamodel_training()
    # Simulate neural network training
    if CUDA.functional()
        # Small matrix multiplication
        A = CUDA.randn(100, 100)
        B = CUDA.randn(100, 100)
        C = A * B
        return sum(C)
    else
        return 0.0
    end
end

function simulate_metamodel_inference()
    # Simulate neural network inference
    if CUDA.functional()
        # Smaller computation for inference
        x = CUDA.randn(100)
        W = CUDA.randn(50, 100)
        y = W * x
        return sum(y)
    else
        return 0.0
    end
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_work_distribution()
end