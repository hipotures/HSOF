module EnsembleDebugger

using CUDA
using JSON3
using Dates
using Printf
using Statistics
using Plots
using ColorSchemes

include("tree_state_visualizer.jl")
include("ensemble_timeline.jl")
include("profiling_hooks.jl")
include("feature_heatmap.jl")
include("debug_logger.jl")

using .TreeStateVisualizer
using .EnsembleTimeline
using .ProfilingHooks
using .FeatureHeatmap
using .DebugLogger

export EnsembleDebugSession, start_debug_session, stop_debug_session
export visualize_tree_state, generate_timeline, create_heatmap
export profile_component, get_profiling_report
export set_debug_level, log_debug

"""
Main debugging session for ensemble analysis
"""
mutable struct EnsembleDebugSession
    # Core components
    tree_visualizer::TreeVisualizer
    timeline::TimelineRecorder
    profiler::ComponentProfiler
    heatmap_gen::HeatmapGenerator
    logger::DebugLogManager
    
    # Session state
    is_active::Bool
    start_time::DateTime
    session_id::String
    output_dir::String
    
    # Configuration
    config::Dict{String, Any}
    
    function EnsembleDebugSession(;
        output_dir::String = "debug_output",
        config::Dict{String, Any} = Dict()
    )
        # Create output directory
        mkpath(output_dir)
        
        # Generate session ID
        session_id = string(now(), "_", rand(UInt32))
        
        # Initialize components
        tree_viz = TreeVisualizer(output_dir)
        timeline = TimelineRecorder()
        profiler = ComponentProfiler()
        heatmap_gen = HeatmapGenerator()
        logger = DebugLogManager(output_dir, config)
        
        new(
            tree_viz, timeline, profiler, heatmap_gen, logger,
            false, now(), session_id, output_dir, config
        )
    end
end

"""
Start a debugging session for ensemble analysis
"""
function start_debug_session(session::EnsembleDebugSession)
    if session.is_active
        @warn "Debug session already active"
        return
    end
    
    session.is_active = true
    session.start_time = now()
    
    # Initialize all components
    initialize!(session.tree_visualizer)
    start_recording!(session.timeline)
    reset!(session.profiler)
    clear!(session.heatmap_gen)
    
    # Log session start
    log_info(session.logger, "Debug session started", 
        session_id = session.session_id,
        output_dir = session.output_dir
    )
    
    @info "Ensemble debug session started" session_id=session.session_id
end

"""
Stop the debugging session and generate reports
"""
function stop_debug_session(session::EnsembleDebugSession)
    if !session.is_active
        @warn "Debug session not active"
        return
    end
    
    session.is_active = false
    elapsed = now() - session.start_time
    
    # Generate final reports
    @info "Generating debug reports..."
    
    # Save timeline
    timeline_path = joinpath(session.output_dir, "timeline_$(session.session_id).json")
    save_timeline(session.timeline, timeline_path)
    
    # Generate profiling report
    prof_report = generate_report(session.profiler)
    prof_path = joinpath(session.output_dir, "profiling_$(session.session_id).json")
    save_profiling_report(prof_report, prof_path)
    
    # Generate feature heatmap
    heatmap_path = joinpath(session.output_dir, "feature_heatmap_$(session.session_id).png")
    save_heatmap(session.heatmap_gen, heatmap_path)
    
    # Save debug logs
    flush_logs(session.logger)
    
    log_info(session.logger, "Debug session stopped",
        session_id = session.session_id,
        duration = elapsed,
        reports_generated = 4
    )
    
    @info "Debug session completed" session_id=session.session_id duration=elapsed
end

"""
Visualize current state of a specific tree
"""
function visualize_tree_state(
    session::EnsembleDebugSession,
    tree_id::Int,
    tree_data::Any;
    save_path::Union{String, Nothing} = nothing
)
    if !session.is_active
        error("Debug session not active")
    end
    
    # Generate visualization
    viz = create_tree_visualization(
        session.tree_visualizer,
        tree_id,
        tree_data
    )
    
    # Save if path provided
    if !isnothing(save_path)
        save_visualization(viz, save_path)
    end
    
    return viz
end

"""
Generate timeline visualization for ensemble execution
"""
function generate_timeline(
    session::EnsembleDebugSession;
    save_path::Union{String, Nothing} = nothing
)
    if !session.is_active
        error("Debug session not active")
    end
    
    # Create timeline visualization
    timeline_viz = create_timeline_visualization(session.timeline)
    
    # Save if requested
    if !isnothing(save_path)
        save_timeline_visualization(timeline_viz, save_path)
    end
    
    return timeline_viz
end

"""
Create feature selection heatmap
"""
function create_heatmap(
    session::EnsembleDebugSession,
    feature_data::Matrix{Float32};
    save_path::Union{String, Nothing} = nothing
)
    if !session.is_active
        error("Debug session not active")
    end
    
    # Update heatmap data
    update_heatmap_data!(session.heatmap_gen, feature_data)
    
    # Generate visualization
    heatmap_viz = generate_heatmap(session.heatmap_gen)
    
    # Save if requested
    if !isnothing(save_path)
        save_heatmap(heatmap_viz, save_path)
    end
    
    return heatmap_viz
end

"""
Profile a specific component execution
"""
function profile_component(
    session::EnsembleDebugSession,
    component_name::String,
    func::Function,
    args...;
    kwargs...
)
    if !session.is_active
        error("Debug session not active")
    end
    
    # Profile the function execution
    result, timing = profile_execution(
        session.profiler,
        component_name,
        func,
        args...;
        kwargs...
    )
    
    # Log profiling event
    log_debug(session.logger, "Component profiled",
        component = component_name,
        execution_time_ms = timing * 1000
    )
    
    return result
end

"""
Get current profiling report
"""
function get_profiling_report(session::EnsembleDebugSession)
    return generate_report(session.profiler)
end

"""
Set debug logging level
"""
function set_debug_level(session::EnsembleDebugSession, level::Symbol)
    set_log_level!(session.logger, level)
end

"""
Log debug message
"""
function log_debug(session::EnsembleDebugSession, message::String; kwargs...)
    log_message(session.logger, :debug, message; kwargs...)
end

"""
Integration with ensemble execution
"""
function hook_ensemble_events!(session::EnsembleDebugSession, ensemble::Any)
    # Hook into tree selection events
    on_tree_select = (tree_id, node_id) -> begin
        record_event!(session.timeline, :tree_select,
            tree_id = tree_id,
            node_id = node_id,
            timestamp = now()
        )
    end
    
    # Hook into GPU synchronization
    on_gpu_sync = (gpu_id, sync_type) -> begin
        record_event!(session.timeline, :gpu_sync,
            gpu_id = gpu_id,
            sync_type = sync_type,
            timestamp = now()
        )
    end
    
    # Hook into feature selection
    on_feature_select = (tree_id, feature_id) -> begin
        update_feature_count!(session.heatmap_gen, feature_id, tree_id)
    end
    
    # Register hooks with ensemble
    # Note: This would need to be integrated with the actual ensemble implementation
    
    return nothing
end

"""
Utility function to create debug session with common configuration
"""
function create_standard_debug_session(;
    output_dir::String = "debug_output_$(Dates.format(now(), "yyyymmdd_HHMMSS"))",
    log_level::Symbol = :info,
    profile_gpu::Bool = true,
    track_memory::Bool = true
)
    config = Dict{String, Any}(
        "log_level" => log_level,
        "profile_gpu" => profile_gpu,
        "track_memory" => track_memory,
        "timeline_resolution_ms" => 100,
        "heatmap_update_interval" => 1000
    )
    
    session = EnsembleDebugSession(
        output_dir = output_dir,
        config = config
    )
    
    return session
end

end # module