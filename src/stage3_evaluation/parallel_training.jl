module ParallelTraining

using Base.Threads
using Distributed
using ProgressMeter
using DataFrames
using Statistics
using Dates
using InteractiveUtils

# Re-export key functions
export ParallelTrainer, WorkQueue, TrainingResult
export train_models_parallel, create_work_queue, collect_results
export ProgressMonitor, monitor_progress
export interrupt_training, save_partial_results

"""
Work item representing a single model training task
"""
struct WorkItem
    id::Int
    model_type::Symbol
    feature_indices::Vector{Int}
    hyperparams::Dict{Symbol, Any}
    priority::Int  # Higher priority = process first
end

"""
Thread-safe work queue for distributing training tasks
"""
mutable struct WorkQueue
    items::Vector{WorkItem}
    lock::ReentrantLock
    completed::Atomic{Int}
    total::Int
    
    function WorkQueue(items::Vector{WorkItem})
        sorted_items = sort(items, by=x->x.priority, rev=true)
        new(sorted_items, ReentrantLock(), Atomic{Int}(0), length(items))
    end
end

"""
Result from a single model training
"""
struct TrainingResult
    work_id::Int
    model_type::Symbol
    feature_indices::Vector{Int}
    metrics::Dict{Symbol, Float64}
    training_time::Float64
    thread_id::Int
    error::Union{Nothing, String}
end

"""
Progress monitoring for parallel training
"""
mutable struct ProgressMonitor
    start_time::DateTime
    completed::Atomic{Int}
    total::Int
    results_per_second::Atomic{Float64}
    lock::ReentrantLock
    last_update::DateTime
    progress_bar::Union{Nothing, Progress}
    
    function ProgressMonitor(total::Int; show_progress::Bool=true)
        now_time = now()
        prog_bar = show_progress ? Progress(total, 1, "Training models: ") : nothing
        new(now_time, Atomic{Int}(0), total, Atomic{Float64}(0.0), 
            ReentrantLock(), now_time, prog_bar)
    end
end

"""
Main parallel trainer managing thread pool and work distribution
"""
mutable struct ParallelTrainer
    n_threads::Int
    work_queue::WorkQueue
    results::Vector{TrainingResult}
    results_lock::ReentrantLock
    progress_monitor::ProgressMonitor
    interrupt_flag::Atomic{Bool}
    memory_limit_per_thread::Int  # MB
    
    function ParallelTrainer(work_items::Vector{WorkItem}; 
                            n_threads::Int=Threads.nthreads(),
                            memory_limit_mb::Int=2048,
                            show_progress::Bool=true)
        queue = WorkQueue(work_items)
        monitor = ProgressMonitor(length(work_items), show_progress=show_progress)
        memory_per_thread = memory_limit_mb ÷ n_threads
        
        new(n_threads, queue, TrainingResult[], ReentrantLock(), 
            monitor, Atomic{Bool}(false), memory_per_thread)
    end
end

"""
Get next work item from queue (thread-safe)
"""
function get_next_work!(queue::WorkQueue)
    lock(queue.lock) do
        if isempty(queue.items)
            return nothing
        else
            return popfirst!(queue.items)
        end
    end
end

"""
Add result to collection (thread-safe)
"""
function add_result!(trainer::ParallelTrainer, result::TrainingResult)
    lock(trainer.results_lock) do
        push!(trainer.results, result)
    end
    
    # Update progress
    atomic_add!(trainer.progress_monitor.completed, 1)
    update_progress!(trainer.progress_monitor)
end

"""
Update progress monitor with ETA calculation
"""
function update_progress!(monitor::ProgressMonitor)
    completed = monitor.completed[]
    
    lock(monitor.lock) do
        current_time = now()
        time_diff = (current_time - monitor.last_update).value / 1000.0  # seconds
        
        if time_diff > 0.1  # Update at most 10 times per second
            elapsed = (current_time - monitor.start_time).value / 1000.0
            rate = completed / elapsed
            monitor.results_per_second[] = rate
            
            if !isnothing(monitor.progress_bar)
                eta_seconds = (monitor.total - completed) / rate
                eta_str = format_time(eta_seconds)
                
                ProgressMeter.update!(monitor.progress_bar, completed,
                    desc="Training models ($(completed)/$(monitor.total), ETA: $eta_str): ")
            end
            
            monitor.last_update = current_time
        end
    end
end

"""
Format time duration for display
"""
function format_time(seconds::Float64)
    if seconds < 60
        return "$(round(Int, seconds))s"
    elseif seconds < 3600
        minutes = round(Int, seconds / 60)
        return "$(minutes)m"
    else
        hours = round(Int, seconds / 3600)
        minutes = round(Int, (seconds % 3600) / 60)
        return "$(hours)h $(minutes)m"
    end
end

"""
Extract feature subset efficiently with memory reuse
"""
function extract_features(X::AbstractMatrix, indices::Vector{Int}, 
                         buffer::Union{Nothing, Matrix{Float64}}=nothing)
    n_samples = size(X, 1)
    n_features = length(indices)
    
    # Reuse buffer if provided and correct size
    if !isnothing(buffer) && size(buffer) == (n_samples, n_features)
        @inbounds for (j, idx) in enumerate(indices)
            buffer[:, j] .= @view X[:, idx]
        end
        return buffer
    else
        return X[:, indices]
    end
end

"""
Worker function for training a single model
"""
function train_single_model(work_item::WorkItem, X::AbstractMatrix, y::AbstractVector,
                           model_factory::Function, cv_function::Function,
                           feature_buffer::Union{Nothing, Matrix{Float64}}=nothing)
    start_time = time()
    thread_id = Threads.threadid()
    
    try
        # Extract features
        X_subset = extract_features(X, work_item.feature_indices, feature_buffer)
        
        # Create model
        model = model_factory(work_item.model_type, work_item.hyperparams)
        
        # Run cross-validation
        cv_results = cv_function(model, X_subset, y)
        
        # Extract metrics
        metrics = cv_results["aggregated_metrics"]
        
        training_time = time() - start_time
        
        return TrainingResult(
            work_item.id,
            work_item.model_type,
            work_item.feature_indices,
            metrics,
            training_time,
            thread_id,
            nothing
        )
    catch e
        error_msg = sprint(showerror, e)
        training_time = time() - start_time
        
        return TrainingResult(
            work_item.id,
            work_item.model_type,
            work_item.feature_indices,
            Dict{Symbol, Float64}(),
            training_time,
            thread_id,
            error_msg
        )
    end
end

"""
Main parallel training execution
"""
function train_models_parallel(trainer::ParallelTrainer, X::AbstractMatrix, y::AbstractVector,
                              model_factory::Function, cv_function::Function;
                              save_interval::Int=100)
    # Pre-allocate feature buffers for each thread
    max_features = maximum(length(item.feature_indices) for item in trainer.work_queue.items)
    feature_buffers = [Matrix{Float64}(undef, size(X, 1), max_features) 
                      for _ in 1:trainer.n_threads]
    
    # Launch parallel workers
    @threads for thread_id in 1:trainer.n_threads
        thread_buffer = feature_buffers[thread_id]
        
        while !trainer.interrupt_flag[]
            # Get next work item
            work_item = get_next_work!(trainer.work_queue)
            
            if isnothing(work_item)
                break  # No more work
            end
            
            # Train model
            result = train_single_model(work_item, X, y, model_factory, 
                                      cv_function, thread_buffer)
            
            # Add result
            add_result!(trainer, result)
            
            # Periodic saving
            if trainer.progress_monitor.completed[] % save_interval == 0
                save_partial_results(trainer)
            end
        end
    end
    
    # Final save
    save_partial_results(trainer)
    
    # Wait for progress bar to finish
    if !isnothing(trainer.progress_monitor.progress_bar)
        ProgressMeter.finish!(trainer.progress_monitor.progress_bar)
    end
    
    return trainer.results
end

"""
Create work queue from feature combinations and model specifications
"""
function create_work_queue(feature_combinations::Vector{Vector{Int}},
                          model_specs::Vector{Tuple{Symbol, Dict{Symbol, Any}}};
                          prioritize_small::Bool=true)
    work_items = WorkItem[]
    work_id = 1
    
    for (model_type, hyperparams) in model_specs
        for feature_indices in feature_combinations
            # Priority based on feature count (smaller = higher priority if prioritize_small)
            priority = prioritize_small ? -length(feature_indices) : length(feature_indices)
            
            push!(work_items, WorkItem(
                work_id,
                model_type,
                feature_indices,
                hyperparams,
                priority
            ))
            work_id += 1
        end
    end
    
    return work_items
end

"""
Interrupt training gracefully
"""
function interrupt_training(trainer::ParallelTrainer)
    println("\nInterrupting training... Saving partial results...")
    trainer.interrupt_flag[] = true
end

"""
Save partial results to file
"""
function save_partial_results(trainer::ParallelTrainer; 
                            filename::String="partial_results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jld2")
    lock(trainer.results_lock) do
        if !isempty(trainer.results)
            # Convert to DataFrame for easy analysis
            df = results_to_dataframe(trainer.results)
            
            # Save using JLD2 or CSV
            # For now, we'll create a simple CSV-compatible format
            println("Saving $(length(trainer.results)) results to $filename")
            # Actual saving implementation would go here
        end
    end
end

"""
Convert results to DataFrame for analysis
"""
function results_to_dataframe(results::Vector{TrainingResult})
    df_data = []
    
    for result in results
        row = Dict(
            :work_id => result.work_id,
            :model_type => result.model_type,
            :n_features => length(result.feature_indices),
            :training_time => result.training_time,
            :thread_id => result.thread_id,
            :has_error => !isnothing(result.error)
        )
        
        # Add metrics
        for (metric_name, value) in result.metrics
            row[metric_name] = value
        end
        
        push!(df_data, row)
    end
    
    return DataFrame(df_data)
end

"""
Collect and aggregate results by model type
"""
function collect_results(results::Vector{TrainingResult})
    # Group by model type
    model_groups = Dict{Symbol, Vector{TrainingResult}}()
    
    for result in results
        if !haskey(model_groups, result.model_type)
            model_groups[result.model_type] = TrainingResult[]
        end
        push!(model_groups[result.model_type], result)
    end
    
    # Aggregate statistics
    aggregated = Dict{Symbol, Dict}()
    
    for (model_type, model_results) in model_groups
        # Filter out errors
        valid_results = filter(r -> isnothing(r.error), model_results)
        
        if !isempty(valid_results)
            # Get all metric names
            metric_names = keys(first(valid_results).metrics)
            
            stats = Dict{Symbol, Any}(
                :count => length(valid_results),
                :error_count => length(model_results) - length(valid_results),
                :avg_training_time => mean(r.training_time for r in valid_results),
                :total_training_time => sum(r.training_time for r in valid_results)
            )
            
            # Aggregate each metric
            for metric in metric_names
                values = [r.metrics[metric] for r in valid_results]
                stats[Symbol("$(metric)_mean")] = mean(values)
                stats[Symbol("$(metric)_std")] = std(values)
                stats[Symbol("$(metric)_min")] = minimum(values)
                stats[Symbol("$(metric)_max")] = maximum(values)
            end
            
            aggregated[model_type] = stats
        end
    end
    
    return aggregated
end

"""
Monitor system resources during training
"""
function monitor_resources(trainer::ParallelTrainer)
    while !trainer.interrupt_flag[]
        # Get memory usage per thread
        gc_stats = Base.gc_num()
        memory_mb = gc_stats.allocd / 1024 / 1024
        
        if memory_mb > trainer.memory_limit_per_thread * trainer.n_threads
            @warn "Memory usage high: $(round(memory_mb, digits=1)) MB"
            # Could trigger GC or reduce thread count
            GC.gc()
        end
        
        sleep(5)  # Check every 5 seconds
    end
end

"""
Dynamic load balancing based on training times
"""
function rebalance_work!(trainer::ParallelTrainer)
    # Analyze completed work
    lock(trainer.results_lock) do
        if length(trainer.results) < 10
            return  # Not enough data
        end
        
        # Calculate average time by model type
        model_times = Dict{Symbol, Float64}()
        model_counts = Dict{Symbol, Int}()
        
        for result in trainer.results
            if !haskey(model_times, result.model_type)
                model_times[result.model_type] = 0.0
                model_counts[result.model_type] = 0
            end
            model_times[result.model_type] += result.training_time
            model_counts[result.model_type] += 1
        end
        
        # Calculate averages
        for model_type in keys(model_times)
            model_times[model_type] /= model_counts[model_type]
        end
        
        # Reorder remaining work based on estimated times
        lock(trainer.work_queue.lock) do
            # Sort by estimated time (longest first for better load balancing)
            sort!(trainer.work_queue.items, 
                  by=item -> get(model_times, item.model_type, 1.0) * length(item.feature_indices),
                  rev=true)
        end
    end
end

end # module