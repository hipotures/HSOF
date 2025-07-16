#!/usr/bin/env julia

# GPU Synchronization Demo
# Demonstrates CPU-based synchronization patterns for multi-GPU coordination

using CUDA
using Printf
using Dates
using Base.Threads

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Load GPU modules
include("../src/gpu/GPU.jl")
using .GPU
using .GPU.GPUSynchronization

function demo_gpu_synchronization()
    println("GPU Synchronization Demo")
    println("=" ^ 60)
    
    # Create synchronization manager
    num_gpus = CUDA.functional() ? min(2, length(CUDA.devices())) : 2
    manager = create_sync_manager(num_gpus=num_gpus, timeout_ms=5000)
    
    println("\nSynchronization Manager Configuration:")
    println("  Number of GPUs: $num_gpus")
    println("  Timeout: $(manager.timeout_ms)ms")
    println("  Active GPUs: $(get_active_gpus(manager))")
    
    # Demo 1: Phase-based Synchronization
    println("\n1. Phase-based Synchronization Demo:")
    println("-" ^ 40)
    
    # Simulate GPU initialization
    gpu_tasks = []
    for gpu_id in 0:(num_gpus-1)
        task = @async begin
            println("  GPU $gpu_id: Starting initialization...")
            
            # Simulate initialization work
            sleep(0.1 * (gpu_id + 1))
            
            # Signal ready
            set_phase!(manager, gpu_id, PHASE_READY)
            println("  GPU $gpu_id: Ready")
            
            # Wait for all GPUs to be ready
            if wait_for_phase(manager, PHASE_READY, timeout_ms=2000)
                println("  GPU $gpu_id: All GPUs ready, starting computation")
                set_phase!(manager, gpu_id, PHASE_RUNNING)
            else
                println("  GPU $gpu_id: Timeout waiting for other GPUs")
            end
        end
        push!(gpu_tasks, task)
    end
    
    # Wait for all tasks
    for task in gpu_tasks
        wait(task)
    end
    
    # Demo 2: Barrier Synchronization
    println("\n2. Barrier Synchronization Demo:")
    println("-" ^ 40)
    
    # Create custom barrier
    compute_barrier = SyncBarrier(num_gpus)
    manager.barriers[:custom_compute] = compute_barrier
    
    gpu_tasks = []
    for gpu_id in 0:(num_gpus-1)
        task = @async begin
            # Phase 1: Different computation times
            compute_time = 0.2 + 0.1 * gpu_id
            println("  GPU $gpu_id: Computing for $(round(compute_time, digits=2))s...")
            sleep(compute_time)
            
            # Store result
            result = gpu_id * 100 + rand(1:50)
            set_gpu_result!(manager, gpu_id, result)
            println("  GPU $gpu_id: Computed result = $result")
            
            # Wait at barrier
            println("  GPU $gpu_id: Waiting at barrier...")
            if enter_barrier!(compute_barrier, timeout_ms=3000)
                println("  GPU $gpu_id: Barrier passed, all GPUs synchronized")
            else
                println("  GPU $gpu_id: Barrier timeout!")
            end
        end
        push!(gpu_tasks, task)
    end
    
    # Wait for completion
    for task in gpu_tasks
        wait(task)
    end
    
    # Show collected results
    println("\nCollected Results:")
    all_results = get_all_results(manager)
    for (gpu_id, result) in all_results
        println("  GPU $gpu_id: $result")
    end
    
    # Demo 3: Event-based Synchronization
    println("\n3. Event-based Synchronization Demo:")
    println("-" ^ 40)
    
    data_ready_event = SyncEvent()
    process_complete_event = SyncEvent()
    
    # Producer task
    producer = @async begin
        println("  Producer: Preparing data...")
        sleep(0.5)
        
        # Data is ready
        println("  Producer: Data ready, signaling consumers")
        signal_event!(data_ready_event)
        
        # Wait for processing to complete
        if wait_for_event(process_complete_event, timeout_ms=2000)
            println("  Producer: Processing complete")
        else
            println("  Producer: Timeout waiting for processing")
        end
    end
    
    # Consumer tasks
    consumers = []
    for i in 1:2
        consumer = @async begin
            println("  Consumer $i: Waiting for data...")
            
            if wait_for_event(data_ready_event, timeout_ms=1000)
                println("  Consumer $i: Data received, processing...")
                sleep(0.2)
                println("  Consumer $i: Processing done")
                
                # Last consumer signals completion
                if i == 2
                    signal_event!(process_complete_event)
                end
            else
                println("  Consumer $i: Timeout waiting for data")
            end
        end
        push!(consumers, consumer)
    end
    
    # Wait for all
    wait(producer)
    for consumer in consumers
        wait(consumer)
    end
    
    # Demo 4: Error Handling
    println("\n4. Error Handling Demo:")
    println("-" ^ 40)
    
    # Reset manager for error demo
    clear_results!(manager)
    
    # Simulate GPU with error
    error_tasks = []
    for gpu_id in 0:(num_gpus-1)
        task = @async begin
            try
                if gpu_id == 1 && num_gpus > 1
                    # Simulate error on GPU 1
                    sleep(0.2)
                    error("Simulated GPU memory error")
                else
                    # Normal operation
                    set_phase!(manager, gpu_id, PHASE_RUNNING)
                    sleep(0.3)
                    set_phase!(manager, gpu_id, PHASE_DONE)
                    println("  GPU $gpu_id: Completed successfully")
                end
            catch e
                set_phase!(manager, gpu_id, PHASE_ERROR, 
                          error_msg="Error: $(sprint(showerror, e))")
                println("  GPU $gpu_id: Failed with error")
            end
        end
        push!(error_tasks, task)
    end
    
    # Wait and check status
    for task in error_tasks
        try
            wait(task)
        catch
            # Expected for error simulation
        end
    end
    
    # Show final state
    println("\nFinal Synchronization State:")
    state = get_sync_state(manager)
    for (gpu_name, gpu_info) in state["gpu_states"]
        println("  $gpu_name:")
        println("    Phase: $(gpu_info["phase"])")
        println("    Active: $(gpu_info["active"])")
        if !isnothing(gpu_info["error"])
            println("    Error: $(gpu_info["error"])")
        end
    end
    
    # Demo 5: Lock-based Resource Access
    println("\n5. Lock-based Resource Access Demo:")
    println("-" ^ 40)
    
    shared_resource = Ref(0)
    resource_lock = ReentrantLock()
    
    access_tasks = []
    for i in 1:3
        task = @async begin
            # Try to access shared resource
            success = with_lock(resource_lock, timeout_ms=1000) do
                current = shared_resource[]
                println("  Task $i: Reading value = $current")
                sleep(0.1)  # Simulate work
                shared_resource[] = current + i
                println("  Task $i: Updated value to $(shared_resource[])")
                return true
            end
            
            if !success
                println("  Task $i: Failed to acquire lock")
            end
        end
        push!(access_tasks, task)
    end
    
    # Wait for all
    for task in access_tasks
        wait(task)
    end
    
    println("  Final shared resource value: $(shared_resource[])")
    
    println("\n" * "=" ^ 60)
    println("Demo completed!")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    demo_gpu_synchronization()
end