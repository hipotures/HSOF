#!/usr/bin/env julia

# Dynamic Rebalancing Demo
# Demonstrates automatic GPU workload rebalancing for multi-GPU systems

using CUDA
using Printf
using Random
using Dates
using Statistics

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU

# Import from GPU module
using .GPU.DynamicRebalancing
using .GPU.DynamicRebalancing: BALANCED, MONITORING, PLANNING, MIGRATING, STABILIZING
using .GPU.DynamicRebalancing: execute_rebalancing!

"""
Simulate dynamic workload changes
"""
function simulate_workload_imbalance!(manager::RebalancingManager)
    gpu_id = rand(0:manager.num_gpus-1)
    
    # Create imbalance by making one GPU heavily loaded
    if rand() < 0.7  # 70% chance of imbalance
        # Heavy load on one GPU
        heavy_gpu = gpu_id
        light_gpu = (gpu_id + 1) % manager.num_gpus
        
        update_workload_metrics!(manager, heavy_gpu,
            utilization = Float32(80.0 + rand() * 15.0),
            memory_usage = Float32(50.0 + rand() * 30.0) / 100.0f0,
            throughput = Float32(8.0 + rand() * 2.0),
            avg_tree_time = 6.0 + rand() * 2.0
        )
        
        update_workload_metrics!(manager, light_gpu,
            utilization = Float32(30.0 + rand() * 20.0),
            memory_usage = Float32(20.0 + rand() * 20.0) / 100.0f0,
            throughput = Float32(12.0 + rand() * 3.0),
            avg_tree_time = 3.0 + rand() * 1.0
        )
    else
        # Balanced load
        for i in 0:manager.num_gpus-1
            update_workload_metrics!(manager, i,
                utilization = Float32(50.0 + rand() * 20.0),
                memory_usage = Float32(40.0 + rand() * 20.0) / 100.0f0,
                throughput = Float32(10.0 + rand() * 2.0),
                avg_tree_time = 4.0 + rand() * 1.0
            )
        end
    end
end

"""
Display current workload distribution
"""
function display_workload_status(manager::RebalancingManager)
    println("\nCurrent Workload Distribution:")
    println("=" ^ 50)
    
    total_util = 0.0f0
    for gpu_id in sort(collect(keys(manager.gpu_workloads)))
        workload = manager.gpu_workloads[gpu_id]
        metrics = workload.current_utilization
        
        # Create visual representation
        util_bar = "█" ^ Int(round(metrics / 5))
        mem_bar = "█" ^ Int(round(workload.memory_usage / 5))
        
        println(@sprintf("GPU %d: Trees=%3d | Util: %-20s %.1f%% | Mem: %-20s %.1f%%",
            gpu_id, workload.total_trees,
            util_bar, metrics,
            mem_bar, workload.memory_usage * 100))
        
        total_util += metrics
    end
    
    avg_util = total_util / manager.num_gpus
    imbalance = check_imbalance(manager)
    
    println("\nAverage Utilization: $(round(avg_util, digits=1))%")
    println("Imbalance Ratio: $(round(imbalance * 100, digits=1))%")
    println("State: $(manager.current_state)")
    
    if imbalance > manager.imbalance_threshold
        println("⚠️  Imbalance detected! (threshold: $(round(manager.imbalance_threshold * 100))%)")
    else
        println("✅ System is balanced")
    end
end

"""
Demo dynamic rebalancing with various scenarios
"""
function demo_dynamic_rebalancing()
    println("GPU Dynamic Rebalancing Demo")
    println("=" ^ 60)
    
    # Create rebalancing manager
    num_gpus = 2  # Simulate 2 GPUs even on single GPU systems
    manager = create_rebalancing_manager(
        num_gpus = num_gpus,
        total_trees = 200,
        imbalance_threshold = 0.15f0,  # 15% threshold
        check_interval = 2.0,
        auto_rebalancing = false,  # Manual control for demo
        hysteresis_factor = 0.8f0,
        cooldown_period = 10.0
    )
    
    println("\nRebalancing Manager Configuration:")
    println("  Number of GPUs: $num_gpus")
    println("  Total trees: $(manager.total_trees)")
    println("  Imbalance threshold: $(manager.imbalance_threshold * 100)%")
    println("  Hysteresis factor: $(manager.hysteresis_factor)")
    println("  Cooldown period: $(manager.cooldown_period)s")
    
    # Demo 1: Initial Distribution
    println("\n1. Initial Even Distribution:")
    println("-" ^ 40)
    display_workload_status(manager)
    
    # Demo 2: Create Imbalance
    println("\n2. Creating Workload Imbalance:")
    println("-" ^ 40)
    
    # Set heavy load on GPU 0
    update_workload_metrics!(manager, 0,
        utilization = 85.0f0,
        memory_usage = 0.65f0,
        throughput = 7.0f0,
        avg_tree_time = 8.0
    )
    
    # Set light load on GPU 1
    update_workload_metrics!(manager, 1,
        utilization = 35.0f0,
        memory_usage = 0.30f0,
        throughput = 13.0f0,
        avg_tree_time = 3.0
    )
    
    display_workload_status(manager)
    
    # Demo 3: Rebalancing Decision
    println("\n3. Analyzing Rebalancing Decision:")
    println("-" ^ 40)
    
    decision = DynamicRebalancing.create_rebalancing_decision(manager)
    
    println("Should rebalance: $(decision.should_rebalance)")
    println("Decision reason: $(decision.decision_reason)")
    
    if decision.should_rebalance
        println("\nMigration Plans:")
        for (i, plan) in enumerate(decision.migration_plans)
            println("  Plan $i: Move $(length(plan.tree_ids)) trees from GPU $(plan.source_gpu) to GPU $(plan.target_gpu)")
            println("    Estimated cost: $(round(plan.estimated_cost, digits=1))ms")
            println("    Expected improvement: $(round(plan.expected_improvement, digits=1))%")
            println("    Reason: $(plan.reason)")
        end
        
        println("\nTotal trees to migrate: $(decision.total_trees_to_migrate)")
        println("Estimated total cost: $(round(decision.estimated_total_cost, digits=1))ms")
    end
    
    # Demo 4: Execute Rebalancing
    if decision.should_rebalance
        println("\n4. Executing Rebalancing:")
        println("-" ^ 40)
        
        # Register callbacks to track progress
        push!(manager.pre_migration_callbacks, (d) -> println("🔄 Starting migration..."))
        push!(manager.post_migration_callbacks, (m) -> 
            println("✅ Migration completed in $(round(m.migration_time, digits=1))ms"))
        
        # Execute rebalancing
        execute_rebalancing!(manager, decision)
        
        # Show new distribution
        println("\nNew Distribution After Rebalancing:")
        display_workload_status(manager)
    end
    
    # Demo 5: Automatic Monitoring
    println("\n5. Automatic Rebalancing Monitor:")
    println("-" ^ 40)
    println("Starting automatic monitoring (10 seconds)...")
    
    # Enable auto-rebalancing
    enable_auto_rebalancing!(manager, true)
    start_rebalancing!(manager)
    
    # Simulate workload changes
    start_time = time()
    while time() - start_time < 10
        # Create dynamic workloads
        simulate_workload_imbalance!(manager)
        
        # Display status every 2 seconds
        if mod(round(time() - start_time), 2) == 0
            print("\r")
            print("Time: $(round(time() - start_time, digits=1))s | ")
            print("State: $(rpad(string(manager.current_state), 12)) | ")
            print("Imbalance: $(round(check_imbalance(manager) * 100, digits=1))%   ")
        end
        
        sleep(0.5)
    end
    
    println("\n\nStopping automatic monitoring...")
    stop_rebalancing!(manager)
    
    # Demo 6: Rebalancing History
    println("\n6. Rebalancing History:")
    println("-" ^ 40)
    
    history = get_rebalancing_history(manager)
    
    if !isempty(history)
        println("Total rebalancing events: $(length(history))")
        println("\nRecent rebalancing operations:")
        
        for (i, event) in enumerate(history[max(1, end-2):end])
            println("\n  Event $i:")
            println("    Time: $(Dates.format(event.timestamp, "HH:MM:SS"))")
            println("    Imbalance before: $(round(event.imbalance_before * 100, digits=1))%")
            println("    Imbalance after: $(round(event.imbalance_after * 100, digits=1))%")
            println("    Trees migrated: $(event.trees_migrated)")
            println("    Migration time: $(round(event.migration_time, digits=1))ms")
            println("    Success: $(event.success ? "✅" : "❌")")
        end
    else
        println("No rebalancing events occurred during monitoring")
    end
    
    # Demo 7: Statistics
    println("\n7. Rebalancing Statistics:")
    println("-" ^ 40)
    
    total_rebalancings = manager.total_rebalancings[]
    total_migrations = manager.total_migrations[]
    total_time = manager.total_migration_time[]
    
    println("Total rebalancing operations: $total_rebalancings")
    println("Total trees migrated: $total_migrations")
    
    if total_rebalancings > 0
        avg_trees = total_migrations / total_rebalancings
        avg_time = total_time / total_rebalancings
        println("Average trees per rebalancing: $(round(avg_trees, digits=1))")
        println("Average migration time: $(round(avg_time, digits=1))ms")
    end
    
    # Demo 8: Advanced Features
    println("\n8. Advanced Rebalancing Features:")
    println("-" ^ 40)
    
    println("• Hysteresis prevents oscillation when imbalance is near threshold")
    println("• Cooldown period prevents too frequent rebalancing")
    println("• Cost model considers migration overhead vs. benefit")
    println("• Gradual migration (max 30% of trees at once)")
    println("• Memory pressure detection prevents OOM during migration")
    println("• State machine ensures consistent operation")
    println("• Callbacks allow custom actions during rebalancing")
    
    # Demo 9: Manual Control
    println("\n9. Manual Migration Example:")
    println("-" ^ 40)
    
    # Create manual migration plan
    trees_to_move = collect(1:10)  # Move first 10 trees
    manual_plan = create_migration_plan(0, 1, trees_to_move, "Manual load adjustment")
    
    println("Manual plan: Move trees $(trees_to_move[1])-$(trees_to_move[end]) from GPU 0 to GPU 1")
    
    # Get current distribution
    current_dist = get_current_distribution(manager)
    println("\nCurrent tree distribution:")
    for (gpu_id, trees) in sort(collect(current_dist))
        println("  GPU $gpu_id: $(length(trees)) trees")
    end
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

# Interactive monitoring function
function interactive_monitor(duration::Int = 30)
    println("\nInteractive Rebalancing Monitor")
    println("=" ^ 40)
    println("Press 'i' to create imbalance, 'b' to balance, 'q' to quit\n")
    
    manager = create_rebalancing_manager(
        num_gpus = 2,
        total_trees = 100,
        imbalance_threshold = 0.1f0,
        auto_rebalancing = true
    )
    
    start_rebalancing!(manager)
    
    # Initial balanced state
    for i in 0:1
        update_workload_metrics!(manager, i,
            utilization = 50.0f0,
            memory_usage = 0.4f0,
            throughput = 10.0f0,
            avg_tree_time = 5.0
        )
    end
    
    start_time = time()
    
    try
        while time() - start_time < duration
            # Update display
            print("\033[H\033[2J")  # Clear screen
            println("Interactive Rebalancing Monitor ($(round(duration - (time() - start_time), digits=0))s remaining)")
            println("=" ^ 60)
            
            display_workload_status(manager)
            
            println("\nRebalancing events: $(manager.total_rebalancings[])")
            println("Total migrations: $(manager.total_migrations[])")
            
            println("\nCommands: [i]mbalance, [b]alance, [q]uit")
            
            # Simulate continuous minor variations
            for i in 0:1
                current = manager.gpu_workloads[i].avg_utilization
                variation = (rand() - 0.5) * 5.0f0
                new_util = clamp(current + variation, 20.0f0, 95.0f0)
                
                update_workload_metrics!(manager, i,
                    utilization = new_util,
                    memory_usage = manager.gpu_workloads[i].memory_usage,
                    throughput = manager.gpu_workloads[i].throughput,
                    avg_tree_time = manager.gpu_workloads[i].avg_tree_time
                )
            end
            
            sleep(1)
        end
    finally
        stop_rebalancing!(manager)
    end
    
    println("\nMonitoring stopped.")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_dynamic_rebalancing()
    
    # Uncomment to run interactive monitor
    # interactive_monitor(30)
end