module Prometheus

using CUDA
using HTTP
using Dates
using Statistics
using Logging

export PrometheusExporter, register_metrics, export_metrics, start_metrics_server

# Metric types
abstract type AbstractMetric end

mutable struct Counter <: AbstractMetric
    name::String
    help::String
    labels::Dict{String, String}
    value::Float64
    
    Counter(name::String, help::String="", labels::Dict{String, String}=Dict{String, String}(), value::Float64=0.0) = 
        new(name, help, labels, value)
end

mutable struct Gauge <: AbstractMetric
    name::String
    help::String
    labels::Dict{String, String}
    value::Float64
    
    Gauge(name::String, help::String="", labels::Dict{String, String}=Dict{String, String}(), value::Float64=0.0) = 
        new(name, help, labels, value)
end

mutable struct Histogram <: AbstractMetric
    name::String
    help::String
    labels::Dict{String, String}
    buckets::Vector{Float64}
    counts::Vector{Int}
    sum::Float64
    count::Int
    
    function Histogram(name::String, help::String="", labels::Dict{String, String}=Dict{String, String}(), 
                      buckets::Vector{Float64}=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, Inf])
        new(name, help, labels, buckets, zeros(Int, length(buckets)), 0.0, 0)
    end
end

# Global metrics registry
const METRICS_REGISTRY = Dict{String, AbstractMetric}()

# Prometheus exporter
struct PrometheusExporter
    metrics::Dict{String, AbstractMetric}
    
    PrometheusExporter() = new(copy(METRICS_REGISTRY))
end

"""
Register a metric in the global registry
"""
function register_metric(metric::AbstractMetric)
    METRICS_REGISTRY[metric.name] = metric
    @debug "Registered metric: $(metric.name)"
    return metric
end

"""
Update counter value
"""
function increment_counter!(name::String, value::Float64=1.0, labels::Dict{String, String}=Dict{String, String}())
    key = labels_key(name, labels)
    if haskey(METRICS_REGISTRY, key)
        metric = METRICS_REGISTRY[key]
        if isa(metric, Counter)
            metric.value += value
        end
    else
        METRICS_REGISTRY[key] = Counter(name, "", labels, value)
    end
end

"""
Update gauge value
"""
function set_gauge!(name::String, value::Float64, labels::Dict{String, String}=Dict{String, String}())
    key = labels_key(name, labels)
    if haskey(METRICS_REGISTRY, key)
        metric = METRICS_REGISTRY[key]
        if isa(metric, Gauge)
            metric.value = value
        end
    else
        METRICS_REGISTRY[key] = Gauge(name, "", labels, value)
    end
end

"""
Observe histogram value
"""
function observe_histogram!(name::String, value::Float64, labels::Dict{String, String}=Dict{String, String}())
    key = labels_key(name, labels)
    if haskey(METRICS_REGISTRY, key)
        metric = METRICS_REGISTRY[key]
        if isa(metric, Histogram)
            for (i, bucket) in enumerate(metric.buckets)
                if value <= bucket
                    metric.counts[i] += 1
                end
            end
            metric.sum += value
            metric.count += 1
        end
    else
        histogram = Histogram(name, "", labels)
        METRICS_REGISTRY[key] = histogram
        observe_histogram!(name, value, labels)
    end
end

"""
Generate labels key for metric identification
"""
function labels_key(name::String, labels::Dict{String, String})
    if isempty(labels)
        return name
    end
    sorted_labels = sort(collect(labels))
    label_str = join(["$(k)=\"$(v)\"" for (k, v) in sorted_labels], ",")
    return "$(name){$(label_str)}"
end

"""
Format labels for Prometheus output
"""
function format_labels(labels::Dict{String, String})
    if isempty(labels)
        return ""
    end
    sorted_labels = sort(collect(labels))
    label_str = join(["$(k)=\"$(v)\"" for (k, v) in sorted_labels], ",")
    return "{$(label_str)}"
end

"""
Initialize all HSOF metrics
"""
function initialize_hsof_metrics()
    # GPU Metrics
    register_metric(Gauge("hsof_gpu_utilization", "GPU utilization percentage"))
    register_metric(Gauge("hsof_gpu_memory_used_bytes", "GPU memory usage in bytes"))
    register_metric(Gauge("hsof_gpu_memory_total_bytes", "GPU total memory in bytes"))
    register_metric(Gauge("hsof_gpu_temperature_celsius", "GPU temperature in Celsius"))
    register_metric(Gauge("hsof_gpu_power_watts", "GPU power consumption in watts"))
    register_metric(Counter("hsof_gpu_errors_total", "Total GPU errors"))
    
    # Pipeline Stage Metrics
    register_metric(Counter("hsof_stage_operations_total", "Total operations per pipeline stage"))
    register_metric(Histogram("hsof_stage_duration_seconds", "Duration of pipeline stage operations"))
    register_metric(Gauge("hsof_stage_features_processed", "Number of features processed by stage"))
    register_metric(Counter("hsof_stage_errors_total", "Total errors per pipeline stage"))
    
    # Model Metrics
    register_metric(Histogram("hsof_model_inference_duration_seconds", "Model inference latency"))
    register_metric(Counter("hsof_model_inference_total", "Total model inferences"))
    register_metric(Counter("hsof_model_inference_errors_total", "Total model inference errors"))
    register_metric(Gauge("hsof_model_accuracy", "Current model accuracy score"))
    register_metric(Gauge("hsof_metamodel_correlation", "Metamodel correlation with actual scores"))
    
    # MCTS Metrics
    register_metric(Counter("hsof_mcts_nodes_explored_total", "Total MCTS nodes explored"))
    register_metric(Gauge("hsof_mcts_tree_depth", "Current MCTS tree depth"))
    register_metric(Histogram("hsof_mcts_node_evaluation_seconds", "MCTS node evaluation time"))
    register_metric(Gauge("hsof_mcts_best_score", "Best score found by MCTS"))
    register_metric(Counter("hsof_mcts_simulations_total", "Total MCTS simulations"))
    
    # Feature Selection Metrics
    register_metric(Gauge("hsof_features_selected", "Number of features selected"))
    register_metric(Gauge("hsof_features_total", "Total number of input features"))
    register_metric(Histogram("hsof_feature_importance_score", "Feature importance scores"))
    register_metric(Counter("hsof_feature_evaluations_total", "Total feature evaluations"))
    
    # Database Metrics
    register_metric(Counter("hsof_database_queries_total", "Total database queries"))
    register_metric(Histogram("hsof_database_query_duration_seconds", "Database query duration"))
    register_metric(Counter("hsof_database_errors_total", "Total database errors"))
    register_metric(Gauge("hsof_database_connections_active", "Active database connections"))
    
    # System Metrics
    register_metric(Gauge("hsof_system_memory_used_bytes", "System memory usage in bytes"))
    register_metric(Gauge("hsof_system_cpu_utilization", "System CPU utilization percentage"))
    register_metric(Counter("hsof_system_gc_runs_total", "Total garbage collection runs"))
    register_metric(Histogram("hsof_system_gc_duration_seconds", "Garbage collection duration"))
    
    @info "Initialized $(length(METRICS_REGISTRY)) HSOF metrics"
end

"""
Update GPU metrics from CUDA
"""
function update_gpu_metrics()
    if !CUDA.functional()
        return
    end
    
    gpu_count = length(CUDA.devices())
    for i in 0:gpu_count-1
        try
            device = CuDevice(i)
            CUDA.device!(device)
            
            labels = Dict("gpu" => string(i))
            
            # Memory metrics
            free_mem = CUDA.available_memory()
            total_mem = CUDA.total_memory()
            used_mem = total_mem - free_mem
            
            set_gauge!("hsof_gpu_memory_used_bytes", Float64(used_mem), labels)
            set_gauge!("hsof_gpu_memory_total_bytes", Float64(total_mem), labels)
            
            # GPU utilization (placeholder - CUDA.jl doesn't have utilization function)
            # In production, this would use nvidia-ml-py or NVML
            util = 50.0  # Placeholder value
            set_gauge!("hsof_gpu_utilization", Float64(util), labels)
            
            # Temperature (placeholder - CUDA.jl doesn't have temperature function)
            temp = 65.0  # Placeholder value  
            set_gauge!("hsof_gpu_temperature_celsius", Float64(temp), labels)
            
            # Power consumption (placeholder)
            power = 200.0  # Placeholder value
            set_gauge!("hsof_gpu_power_watts", Float64(power), labels)
            
        catch e
            @warn "Failed to update GPU metrics for device $i" exception=e
            increment_counter!("hsof_gpu_errors_total", 1.0, Dict("gpu" => string(i)))
        end
    end
end

"""
Export metrics in Prometheus format
"""
function export_metrics()
    output = String[]
    
    # Group metrics by name (without labels)
    grouped_metrics = Dict{String, Vector{Pair{String, AbstractMetric}}}()
    for (key, metric) in METRICS_REGISTRY
        base_name = metric.name
        if !haskey(grouped_metrics, base_name)
            grouped_metrics[base_name] = []
        end
        push!(grouped_metrics[base_name], key => metric)
    end
    
    for (name, metrics) in grouped_metrics
        if isempty(metrics)
            continue
        end
        
        # Get help text from first metric
        help_text = metrics[1].second.help
        if !isempty(help_text)
            push!(output, "# HELP $name $help_text")
        else
            push!(output, "# HELP $name No description available")
        end
        
        # Determine metric type
        metric_type = if isa(metrics[1].second, Counter)
            "counter"
        elseif isa(metrics[1].second, Gauge)
            "gauge"
        elseif isa(metrics[1].second, Histogram)
            "histogram"
        else
            "untyped"
        end
        push!(output, "# TYPE $name $metric_type")
        
        # Export metric values
        for (_, metric) in metrics
            labels_str = format_labels(metric.labels)
            
            if isa(metric, Counter) || isa(metric, Gauge)
                push!(output, "$(metric.name)$labels_str $(metric.value)")
            elseif isa(metric, Histogram)
                # Export histogram buckets
                for (i, bucket) in enumerate(metric.buckets)
                    bucket_labels = copy(metric.labels)
                    bucket_labels["le"] = bucket == Inf ? "+Inf" : string(bucket)
                    bucket_str = format_labels(bucket_labels)
                    push!(output, "$(metric.name)_bucket$bucket_str $(metric.counts[i])")
                end
                
                # Export sum and count
                push!(output, "$(metric.name)_sum$labels_str $(metric.sum)")
                push!(output, "$(metric.name)_count$labels_str $(metric.count)")
            end
        end
        push!(output, "")  # Empty line between metric families
    end
    
    return join(output, "\n")
end

"""
HTTP handler for /metrics endpoint
"""
function metrics_handler(req::HTTP.Request)
    try
        # Update dynamic metrics before export
        update_gpu_metrics()
        
        # Export all metrics in Prometheus format
        metrics_text = export_metrics()
        
        headers = [
            "Content-Type" => "text/plain; version=0.0.4; charset=utf-8",
            "Cache-Control" => "no-cache"
        ]
        
        return HTTP.Response(200, headers, metrics_text)
    catch e
        @error "Failed to export metrics" exception=e
        return HTTP.Response(500, "Internal Server Error")
    end
end

"""
Start Prometheus metrics server
"""
function start_metrics_server(; host="0.0.0.0", port=9090)
    # Initialize metrics if not already done
    if isempty(METRICS_REGISTRY)
        initialize_hsof_metrics()
    end
    
    @info "Starting Prometheus metrics server on $host:$port"
    
    router = HTTP.Router()
    HTTP.register!(router, "/metrics", metrics_handler)
    
    # Add health endpoint
    HTTP.register!(router, "/") do req
        HTTP.Response(200, "Prometheus Metrics Exporter for HSOF")
    end
    
    try
        HTTP.serve(router, host, port)
    catch e
        @error "Failed to start metrics server" exception=e
        rethrow(e)
    end
end

# Convenience functions for common metrics updates
"""
Record stage operation
"""
function record_stage_operation(stage::Int, duration_seconds::Float64, features_processed::Int=0, error::Bool=false)
    labels = Dict("stage" => string(stage))
    
    increment_counter!("hsof_stage_operations_total", 1.0, labels)
    observe_histogram!("hsof_stage_duration_seconds", duration_seconds, labels)
    
    if features_processed > 0
        set_gauge!("hsof_stage_features_processed", Float64(features_processed), labels)
    end
    
    if error
        increment_counter!("hsof_stage_errors_total", 1.0, labels)
    end
end

"""
Record model inference
"""
function record_model_inference(duration_seconds::Float64, accuracy::Float64=0.0, error::Bool=false)
    observe_histogram!("hsof_model_inference_duration_seconds", duration_seconds)
    increment_counter!("hsof_model_inference_total", 1.0)
    
    if accuracy > 0.0
        set_gauge!("hsof_model_accuracy", accuracy)
    end
    
    if error
        increment_counter!("hsof_model_inference_errors_total", 1.0)
    end
end

"""
Record MCTS metrics
"""
function record_mcts_metrics(nodes_explored::Int, tree_depth::Int, best_score::Float64, 
                           evaluation_time::Float64=0.0)
    increment_counter!("hsof_mcts_nodes_explored_total", Float64(nodes_explored))
    set_gauge!("hsof_mcts_tree_depth", Float64(tree_depth))
    set_gauge!("hsof_mcts_best_score", best_score)
    
    if evaluation_time > 0.0
        observe_histogram!("hsof_mcts_node_evaluation_seconds", evaluation_time)
    end
end

"""
Record feature selection metrics
"""
function record_feature_selection(features_selected::Int, features_total::Int, importance_scores::Vector{Float64}=Float64[])
    set_gauge!("hsof_features_selected", Float64(features_selected))
    set_gauge!("hsof_features_total", Float64(features_total))
    
    for score in importance_scores
        observe_histogram!("hsof_feature_importance_score", score)
    end
    
    increment_counter!("hsof_feature_evaluations_total", Float64(length(importance_scores)))
end

"""
Record database operation
"""
function record_database_operation(duration_seconds::Float64, error::Bool=false)
    increment_counter!("hsof_database_queries_total", 1.0)
    observe_histogram!("hsof_database_query_duration_seconds", duration_seconds)
    
    if error
        increment_counter!("hsof_database_errors_total", 1.0)
    end
end

end  # module Prometheus