module PrometheusIntegration

using Logging
include("prometheus.jl")
using .Prometheus

export start_monitoring, stop_monitoring, report_stage_metrics, report_gpu_metrics
export report_model_metrics, report_mcts_metrics, report_database_metrics

# Monitoring state
mutable struct MonitoringState
    active::Bool
    task::Union{Task, Nothing}
    update_interval::Float64
    
    MonitoringState() = new(false, nothing, 30.0)  # 30 second updates
end

const MONITORING_STATE = MonitoringState()

"""
Start automatic metrics collection and reporting
"""
function start_monitoring(; update_interval::Float64=30.0, auto_update_gpu::Bool=true)
    if MONITORING_STATE.active
        @warn "Monitoring already active"
        return
    end
    
    MONITORING_STATE.update_interval = update_interval
    MONITORING_STATE.active = true
    
    # Initialize Prometheus metrics
    Prometheus.initialize_hsof_metrics()
    @info "Prometheus metrics initialized"
    
    if auto_update_gpu
        # Start background task for automatic GPU metrics updates
        MONITORING_STATE.task = @async begin
            while MONITORING_STATE.active
                try
                    Prometheus.update_gpu_metrics()
                    update_system_metrics()
                    sleep(update_interval)
                catch e
                    @error "Error in monitoring loop" exception=e
                    sleep(5.0)  # Brief pause before retry
                end
            end
        end
        @info "Automatic GPU and system metrics collection started (interval: $(update_interval)s)"
    end
end

"""
Stop automatic metrics collection
"""
function stop_monitoring()
    if !MONITORING_STATE.active
        return
    end
    
    MONITORING_STATE.active = false
    
    if MONITORING_STATE.task !== nothing
        try
            # Allow graceful shutdown
            sleep(0.1)
            if !istaskdone(MONITORING_STATE.task)
                @async Base.throwto(MONITORING_STATE.task, InterruptException())
            end
        catch e
            @debug "Error stopping monitoring task" exception=e
        end
        MONITORING_STATE.task = nothing
    end
    
    @info "Monitoring stopped"
end

"""
Update system-level metrics
"""
function update_system_metrics()
    try
        # Memory usage
        memory_info = Sys.free_memory()
        total_memory = Sys.total_memory()
        used_memory = total_memory - memory_info
        
        Prometheus.set_gauge!("hsof_system_memory_used_bytes", Float64(used_memory))
        
        # CPU utilization (simplified - would use more sophisticated method in production)
        # This is a placeholder - actual implementation would use system tools
        cpu_util = 0.0  # Would implement actual CPU monitoring
        Prometheus.set_gauge!("hsof_system_cpu_utilization", cpu_util)
        
        # GC metrics
        gc_stats = Base.gc_num()
        Prometheus.set_gauge!("hsof_system_gc_runs_total", Float64(gc_stats.total_time))
        
    catch e
        @debug "Error updating system metrics" exception=e
    end
end

"""
Report pipeline stage metrics with timing
"""
function report_stage_metrics(stage::Int, operation::String="process")
    start_time = time()
    
    # This would be called at the end of stage operations
    function finish_stage(features_processed::Int=0, error::Bool=false)
        duration = time() - start_time
        Prometheus.record_stage_operation(stage, duration, features_processed, error)
        
        @debug "Stage $stage metrics recorded" operation=operation duration=duration features=features_processed error=error
    end
    
    return finish_stage
end

"""
Convenience wrapper for reporting GPU metrics manually
"""
function report_gpu_metrics()
    try
        Prometheus.update_gpu_metrics()
        @debug "GPU metrics updated manually"
    catch e
        @error "Failed to update GPU metrics" exception=e
    end
end

"""
Report model inference metrics
"""
function report_model_metrics(inference_time_ms::Float64; accuracy::Float64=0.0, error::Bool=false)
    Prometheus.record_model_inference(inference_time_ms / 1000.0, accuracy, error)
    @debug "Model metrics recorded" inference_time=inference_time_ms accuracy=accuracy error=error
end

"""
Report MCTS exploration metrics
"""
function report_mcts_metrics(; nodes_explored::Int=0, tree_depth::Int=0, best_score::Float64=0.0, 
                            evaluation_time_ms::Float64=0.0)
    if nodes_explored > 0 || tree_depth > 0 || best_score > 0.0
        Prometheus.record_mcts_metrics(nodes_explored, tree_depth, best_score, evaluation_time_ms / 1000.0)
        @debug "MCTS metrics recorded" nodes=nodes_explored depth=tree_depth score=best_score eval_time=evaluation_time_ms
    end
end

"""
Report feature selection results
"""
function report_feature_selection_metrics(features_selected::Int, features_total::Int; 
                                         importance_scores::Vector{Float64}=Float64[])
    Prometheus.record_feature_selection(features_selected, features_total, importance_scores)
    @debug "Feature selection metrics recorded" selected=features_selected total=features_total scores_count=length(importance_scores)
end

"""
Report database operation metrics
"""
function report_database_metrics(operation_time_ms::Float64; error::Bool=false)
    Prometheus.record_database_operation(operation_time_ms / 1000.0, error)
    @debug "Database metrics recorded" operation_time=operation_time_ms error=error
end

"""
Helper function to time and report an operation
"""
macro timed_operation(metric_func, args...)
    quote
        start_time = time_ns()
        local result
        local error_occurred = false
        
        try
            result = $(esc(args[end]))
        catch e
            error_occurred = true
            rethrow(e)
        finally
            duration_ms = (time_ns() - start_time) / 1_000_000
            $(esc(metric_func))(duration_ms, error=error_occurred)
        end
        
        result
    end
end

"""
Create hooks for integration with main application components
"""
module Hooks
    using ..PrometheusIntegration
    
    # Stage operation hooks
    function on_stage_start(stage::Int)
        return PrometheusIntegration.report_stage_metrics(stage, "start")
    end
    
    function on_stage_complete(stage::Int, duration_seconds::Float64, features_processed::Int=0)
        PrometheusIntegration.Prometheus.record_stage_operation(stage, duration_seconds, features_processed, false)
    end
    
    function on_stage_error(stage::Int, exception::Exception)
        PrometheusIntegration.Prometheus.increment_counter!("hsof_stage_errors_total", 1.0, 
                                                           Dict("stage" => string(stage)))
        @warn "Stage $stage error recorded" exception=exception
    end
    
    # Model operation hooks
    function on_model_inference(inference_time_ms::Float64, accuracy::Float64=0.0)
        PrometheusIntegration.report_model_metrics(inference_time_ms, accuracy=accuracy)
    end
    
    function on_model_loaded(version::String)
        PrometheusIntegration.Prometheus.set_gauge!("hsof_model_loaded", 1.0, 
                                                   Dict("version" => version))
        @info "Model loaded" version=version
    end
    
    # MCTS operation hooks
    function on_mcts_iteration(nodes_explored::Int, best_score::Float64)
        PrometheusIntegration.report_mcts_metrics(nodes_explored=nodes_explored, best_score=best_score)
    end
    
    function on_feature_evaluation(features_selected::Int, features_total::Int, score::Float64)
        PrometheusIntegration.report_feature_selection_metrics(features_selected, features_total, 
                                                              importance_scores=[score])
    end
    
    # Database operation hooks
    function on_database_query(duration_ms::Float64, query_type::String="unknown")
        PrometheusIntegration.report_database_metrics(duration_ms)
        PrometheusIntegration.Prometheus.increment_counter!("hsof_database_queries_total", 1.0,
                                                          Dict("type" => query_type))
    end
    
    export on_stage_start, on_stage_complete, on_stage_error
    export on_model_inference, on_model_loaded
    export on_mcts_iteration, on_feature_evaluation
    export on_database_query
end

end  # module PrometheusIntegration