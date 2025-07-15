module CUDATiming

using CUDA
using Statistics
using Printf

export CUDATimer, CUDAEvent, EventPair
export create_timer, start_timing!, stop_timing!, elapsed_time
export TimingResult, get_timing_results, reset_timer!

"""
Event pair for timing CUDA operations
"""
struct EventPair
    start_event::CuEvent
    stop_event::CuEvent
end

"""
CUDA timing result
"""
struct TimingResult
    name::String
    elapsed_ms::Float32
    count::Int
    min_ms::Float32
    max_ms::Float32
    mean_ms::Float32
    std_ms::Float32
end

"""
CUDA timer for profiling kernel execution
"""
mutable struct CUDATimer
    events::Dict{String, Vector{EventPair}}
    active_events::Dict{String, EventPair}
    enabled::Bool
    max_history::Int
end

"""
Create a new CUDA timer
"""
function create_timer(; enabled::Bool = true, max_history::Int = 100)
    return CUDATimer(
        Dict{String, Vector{EventPair}}(),
        Dict{String, EventPair}(),
        enabled,
        max_history
    )
end

"""
Start timing a named operation
"""
function start_timing!(timer::CUDATimer, name::String)
    if !timer.enabled
        return
    end
    
    # Create event pair
    start_event = CuEvent()
    stop_event = CuEvent()
    event_pair = EventPair(start_event, stop_event)
    
    # Record start event
    CUDA.record(start_event)
    
    # Store active event
    timer.active_events[name] = event_pair
end

"""
Stop timing a named operation
"""
function stop_timing!(timer::CUDATimer, name::String)
    if !timer.enabled || !haskey(timer.active_events, name)
        return
    end
    
    # Get active event pair
    event_pair = timer.active_events[name]
    
    # Record stop event
    CUDA.record(event_pair.stop_event)
    
    # Move to history
    if !haskey(timer.events, name)
        timer.events[name] = EventPair[]
    end
    
    push!(timer.events[name], event_pair)
    
    # Limit history size
    if length(timer.events[name]) > timer.max_history
        popfirst!(timer.events[name])
    end
    
    # Remove from active
    delete!(timer.active_events, name)
end

"""
Calculate elapsed time for an event pair
"""
function elapsed_time(event_pair::EventPair)
    CUDA.synchronize(event_pair.stop_event)
    return CUDA.elapsed(event_pair.start_event, event_pair.stop_event)
end

"""
Get timing results for all recorded operations
"""
function get_timing_results(timer::CUDATimer)
    results = TimingResult[]
    
    for (name, event_pairs) in timer.events
        if isempty(event_pairs)
            continue
        end
        
        # Calculate elapsed times
        times_ms = Float32[]
        for event_pair in event_pairs
            push!(times_ms, elapsed_time(event_pair))
        end
        
        # Calculate statistics
        result = TimingResult(
            name,
            sum(times_ms),
            length(times_ms),
            minimum(times_ms),
            maximum(times_ms),
            mean(times_ms),
            std(times_ms)
        )
        
        push!(results, result)
    end
    
    return results
end

"""
Reset timer history
"""
function reset_timer!(timer::CUDATimer)
    empty!(timer.events)
    empty!(timer.active_events)
end

"""
Pretty print timing results
"""
function Base.show(io::IO, result::TimingResult)
    println(io, "$(result.name):")
    println(io, "  Total: $(round(result.elapsed_ms, digits=3)) ms")
    println(io, "  Count: $(result.count)")
    println(io, "  Mean: $(round(result.mean_ms, digits=3)) ms")
    println(io, "  Min: $(round(result.min_ms, digits=3)) ms")
    println(io, "  Max: $(round(result.max_ms, digits=3)) ms")
    println(io, "  Std: $(round(result.std_ms, digits=3)) ms")
end

"""
Timing macro for convenient profiling
"""
macro cuda_time(timer, name, expr)
    quote
        start_timing!($(esc(timer)), $(esc(name)))
        try
            $(esc(expr))
        finally
            stop_timing!($(esc(timer)), $(esc(name)))
        end
    end
end

export @cuda_time

end # module