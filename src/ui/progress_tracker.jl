module ProgressTracker

using Dates
using Statistics
using Printf

export ProgressState, Tracker, MCTSPhase
export create_progress_tracker, update_progress!, add_stage!
export get_eta, get_progress_percentage, get_current_stage
export pause_tracker!, resume_tracker!, is_paused
export save_progress, load_progress
export get_progress_bar, get_stage_indicator, get_speed_stats
export reset_tracker!, set_total_items!, complete_stage!
export get_history, get_formatted_eta, get_phase_name, get_phase_symbol

"""
MCTS phase enumeration
"""
@enum MCTSPhase begin
    MCTS_SELECTION = 1
    MCTS_EXPANSION = 2
    MCTS_SIMULATION = 3
    MCTS_BACKPROPAGATION = 4
    MCTS_IDLE = 5
end

"""
Progress stage information
"""
mutable struct ProgressStage
    name::String
    total_items::Int
    completed_items::Int
    start_time::DateTime
    last_update::DateTime
    phase::MCTSPhase
    
    function ProgressStage(name::String, total_items::Int = 0)
        now_time = now()
        new(name, total_items, 0, now_time, now_time, MCTS_IDLE)
    end
end

"""
Speed tracking with rolling window
"""
mutable struct SpeedTracker
    window_size::Int
    timestamps::Vector{DateTime}
    items_processed::Vector{Int}
    current_index::Int
    
    function SpeedTracker(window_size::Int = 100)
        new(window_size, DateTime[], Int[], 0)
    end
end

"""
Progress history for visualization
"""
mutable struct ProgressHistory
    max_entries::Int
    timestamps::Vector{DateTime}
    percentages::Vector{Float64}
    speeds::Vector{Float64}  # items per second
    
    function ProgressHistory(max_entries::Int = 600)  # 60 seconds at 10 FPS
        new(max_entries, DateTime[], Float64[], Float64[])
    end
end

"""
Main progress state manager
"""
mutable struct ProgressState
    stages::Vector{ProgressStage}
    current_stage_index::Int
    overall_total::Int
    overall_completed::Int
    start_time::DateTime
    last_update::DateTime
    paused::Bool
    pause_start::Union{DateTime, Nothing}
    total_pause_duration::Millisecond
    speed_tracker::SpeedTracker
    history::ProgressHistory
    persistence_file::String
    
    function ProgressState(;persistence_file::String = "")
        now_time = now()
        new(ProgressStage[], 0, 0, 0, now_time, now_time, 
            false, nothing, Millisecond(0),
            SpeedTracker(), ProgressHistory(), persistence_file)
    end
end

"""
Main progress tracker
"""
mutable struct Tracker
    state::ProgressState
    update_callback::Union{Function, Nothing}
    
    function Tracker(;kwargs...)
        state = ProgressState(;kwargs...)
        new(state, nothing)
    end
end

# Core Functions

"""
Create a new progress tracker
"""
function create_progress_tracker(;
    persistence_file::String = "",
    update_callback::Union{Function, Nothing} = nothing
)
    tracker = Tracker(persistence_file = persistence_file)
    tracker.update_callback = update_callback
    
    # Try to load existing progress
    if !isempty(persistence_file) && isfile(persistence_file)
        try
            load_progress(tracker, persistence_file)
        catch e
            @warn "Failed to load progress file" error=e
        end
    end
    
    return tracker
end

"""
Add a new stage to track
"""
function add_stage!(tracker::Tracker, name::String, total_items::Int = 0)
    stage = ProgressStage(name, total_items)
    push!(tracker.state.stages, stage)
    
    # Update overall total
    tracker.state.overall_total += total_items
    
    # Set as current stage if it's the first
    if tracker.state.current_stage_index == 0
        tracker.state.current_stage_index = 1
    end
    
    return length(tracker.state.stages)
end

"""
Update progress for current stage
"""
function update_progress!(tracker::Tracker, items_completed::Int = 1;
                         phase::MCTSPhase = MCTS_IDLE)
    if tracker.state.paused
        return
    end
    
    state = tracker.state
    if state.current_stage_index == 0 || isempty(state.stages)
        return
    end
    
    # Update current stage
    stage = state.stages[state.current_stage_index]
    stage.completed_items += items_completed
    stage.last_update = now()
    stage.phase = phase
    
    # Update overall progress
    state.overall_completed += items_completed
    state.last_update = now()
    
    # Update speed tracking
    update_speed!(state.speed_tracker, items_completed, state.last_update)
    
    # Update history
    update_history!(state.history, state.last_update, 
                   get_progress_percentage(tracker), 
                   get_current_speed(state.speed_tracker))
    
    # Trigger callback if set
    if !isnothing(tracker.update_callback)
        tracker.update_callback(tracker)
    end
    
    # Auto-save if persistence is enabled
    if !isempty(state.persistence_file)
        save_progress(tracker)
    end
end

"""
Set total items for current stage
"""
function set_total_items!(tracker::Tracker, total::Int)
    state = tracker.state
    if state.current_stage_index == 0 || isempty(state.stages)
        return
    end
    
    stage = state.stages[state.current_stage_index]
    old_total = stage.total_items
    stage.total_items = total
    
    # Update overall total
    state.overall_total += (total - old_total)
end

"""
Complete current stage and move to next
"""
function complete_stage!(tracker::Tracker)
    state = tracker.state
    if state.current_stage_index == 0 || isempty(state.stages)
        return
    end
    
    # Mark current stage as complete
    stage = state.stages[state.current_stage_index]
    items_to_complete = stage.total_items - stage.completed_items
    stage.completed_items = stage.total_items
    
    # Update overall completed count
    state.overall_completed += items_to_complete
    
    # Move to next stage if available
    if state.current_stage_index < length(state.stages)
        state.current_stage_index += 1
    end
end

"""
Get current progress percentage (0-100)
"""
function get_progress_percentage(tracker::Tracker)
    state = tracker.state
    if state.overall_total == 0
        return 0.0
    end
    
    return (state.overall_completed / state.overall_total) * 100.0
end

"""
Get current stage info
"""
function get_current_stage(tracker::Tracker)
    state = tracker.state
    if state.current_stage_index == 0 || isempty(state.stages)
        return nothing
    end
    
    stage = state.stages[state.current_stage_index]
    return (
        name = stage.name,
        progress = stage.total_items > 0 ? stage.completed_items / stage.total_items : 0.0,
        completed = stage.completed_items,
        total = stage.total_items,
        phase = stage.phase
    )
end

"""
Calculate ETA based on current speed
"""
function get_eta(tracker::Tracker)
    state = tracker.state
    items_remaining = state.overall_total - state.overall_completed
    
    if items_remaining <= 0
        return Millisecond(0)
    end
    
    current_speed = get_current_speed(state.speed_tracker)
    if current_speed <= 0
        return nothing  # Cannot estimate
    end
    
    # Calculate seconds remaining
    seconds_remaining = items_remaining / current_speed
    return Millisecond(round(Int, seconds_remaining * 1000))
end

"""
Get formatted ETA string
"""
function get_formatted_eta(tracker::Tracker)
    eta = get_eta(tracker)
    
    if isnothing(eta)
        return "Unknown"
    elseif eta == Millisecond(0)
        return "Complete"
    end
    
    # Convert to appropriate units
    total_seconds = eta.value / 1000
    if total_seconds < 60
        return @sprintf("%.0fs", total_seconds)
    elseif total_seconds < 3600
        minutes = floor(Int, total_seconds / 60)
        seconds = total_seconds % 60
        return @sprintf("%dm %ds", minutes, seconds)
    else
        hours = floor(Int, total_seconds / 3600)
        minutes = floor(Int, (total_seconds % 3600) / 60)
        return @sprintf("%dh %dm", hours, minutes)
    end
end

"""
Update speed tracker
"""
function update_speed!(tracker::SpeedTracker, items::Int, timestamp::DateTime)
    push!(tracker.timestamps, timestamp)
    push!(tracker.items_processed, items)
    
    # Maintain window size
    if length(tracker.timestamps) > tracker.window_size
        popfirst!(tracker.timestamps)
        popfirst!(tracker.items_processed)
    end
end

"""
Get current processing speed (items per second)
"""
function get_current_speed(tracker::SpeedTracker)
    if length(tracker.timestamps) < 2
        return 0.0
    end
    
    # Calculate time span
    time_span = tracker.timestamps[end] - tracker.timestamps[1]
    seconds = Dates.value(time_span) / 1000.0
    
    if seconds <= 0
        return 0.0
    end
    
    # Sum items processed
    total_items = sum(tracker.items_processed)
    return total_items / seconds
end

"""
Update progress history
"""
function update_history!(history::ProgressHistory, timestamp::DateTime, 
                        percentage::Float64, speed::Float64)
    push!(history.timestamps, timestamp)
    push!(history.percentages, percentage)
    push!(history.speeds, speed)
    
    # Maintain max entries
    while length(history.timestamps) > history.max_entries
        popfirst!(history.timestamps)
        popfirst!(history.percentages)
        popfirst!(history.speeds)
    end
end

"""
Pause progress tracking
"""
function pause_tracker!(tracker::Tracker)
    if !tracker.state.paused
        tracker.state.paused = true
        tracker.state.pause_start = now()
    end
end

"""
Resume progress tracking
"""
function resume_tracker!(tracker::Tracker)
    if tracker.state.paused && !isnothing(tracker.state.pause_start)
        pause_duration = now() - tracker.state.pause_start
        tracker.state.total_pause_duration += pause_duration
        tracker.state.paused = false
        tracker.state.pause_start = nothing
    end
end

"""
Check if tracker is paused
"""
function is_paused(tracker::Tracker)
    return tracker.state.paused
end

"""
Get speed statistics
"""
function get_speed_stats(tracker::Tracker)
    speeds = tracker.state.history.speeds
    if isempty(speeds)
        return (current = 0.0, average = 0.0, min = 0.0, max = 0.0)
    end
    
    return (
        current = get_current_speed(tracker.state.speed_tracker),
        average = mean(speeds),
        min = minimum(speeds),
        max = maximum(speeds)
    )
end

"""
Get progress history
"""
function get_history(tracker::Tracker; last_n::Int = 0)
    history = tracker.state.history
    
    if last_n > 0 && length(history.timestamps) > last_n
        start_idx = length(history.timestamps) - last_n + 1
        return (
            timestamps = history.timestamps[start_idx:end],
            percentages = history.percentages[start_idx:end],
            speeds = history.speeds[start_idx:end]
        )
    end
    
    return (
        timestamps = copy(history.timestamps),
        percentages = copy(history.percentages),
        speeds = copy(history.speeds)
    )
end

"""
Generate progress bar string
"""
function get_progress_bar(tracker::Tracker; 
                         width::Int = 50,
                         filled_char::Char = '█',
                         empty_char::Char = '░')
    percentage = get_progress_percentage(tracker)
    filled_count = round(Int, (percentage / 100.0) * width)
    empty_count = width - filled_count
    
    bar = string(filled_char)^filled_count * string(empty_char)^empty_count
    return @sprintf("[%s] %.1f%%", bar, percentage)
end

"""
Get MCTS phase indicator
"""
function get_stage_indicator(tracker::Tracker)
    stage_info = get_current_stage(tracker)
    if isnothing(stage_info)
        return "No active stage"
    end
    
    phase_symbol = get_phase_symbol(stage_info.phase)
    phase_name = get_phase_name(stage_info.phase)
    
    return @sprintf("%s %s: %s (%.1f%%)", 
                   phase_symbol, 
                   stage_info.name,
                   phase_name,
                   stage_info.progress * 100)
end

"""
Get phase symbol
"""
function get_phase_symbol(phase::MCTSPhase)
    if phase == MCTS_SELECTION
        return "◐"
    elseif phase == MCTS_EXPANSION
        return "◑"
    elseif phase == MCTS_SIMULATION
        return "◒"
    elseif phase == MCTS_BACKPROPAGATION
        return "◓"
    else
        return "○"
    end
end

"""
Get phase name
"""
function get_phase_name(phase::MCTSPhase)
    if phase == MCTS_SELECTION
        return "Selection"
    elseif phase == MCTS_EXPANSION
        return "Expansion"
    elseif phase == MCTS_SIMULATION
        return "Simulation"
    elseif phase == MCTS_BACKPROPAGATION
        return "Backpropagation"
    else
        return "Idle"
    end
end

"""
Reset tracker
"""
function reset_tracker!(tracker::Tracker)
    state = tracker.state
    
    # Reset all counters
    state.overall_completed = 0
    state.current_stage_index = isempty(state.stages) ? 0 : 1
    state.start_time = now()
    state.last_update = now()
    state.total_pause_duration = Millisecond(0)
    
    # Reset stages
    for stage in state.stages
        stage.completed_items = 0
        stage.start_time = now()
        stage.last_update = now()
        stage.phase = MCTS_IDLE
    end
    
    # Clear history
    state.speed_tracker = SpeedTracker()
    state.history = ProgressHistory()
end

"""
Save progress to file
"""
function save_progress(tracker::Tracker)
    if isempty(tracker.state.persistence_file)
        return
    end
    
    try
        data = Dict{String, Any}(
            "stages" => [Dict(
                "name" => s.name,
                "total_items" => s.total_items,
                "completed_items" => s.completed_items,
                "phase" => Int(s.phase)
            ) for s in tracker.state.stages],
            "current_stage_index" => tracker.state.current_stage_index,
            "overall_total" => tracker.state.overall_total,
            "overall_completed" => tracker.state.overall_completed,
            "start_time" => string(tracker.state.start_time),
            "last_update" => string(tracker.state.last_update),
            "total_pause_duration" => tracker.state.total_pause_duration.value
        )
        
        # Write to temporary file first
        temp_file = tracker.state.persistence_file * ".tmp"
        open(temp_file, "w") do io
            write(io, repr(data))
        end
        
        # Atomic rename
        mv(temp_file, tracker.state.persistence_file, force=true)
    catch e
        @warn "Failed to save progress" error=e
    end
end

"""
Load progress from file
"""
function load_progress(tracker::Tracker, filename::String)
    if !isfile(filename)
        return false
    end
    
    try
        content = read(filename, String)
        data = eval(Meta.parse(content))
        
        # Restore stages
        tracker.state.stages = ProgressStage[]
        for stage_data in data["stages"]
            stage = ProgressStage(stage_data["name"], stage_data["total_items"])
            stage.completed_items = stage_data["completed_items"]
            stage.phase = MCTSPhase(stage_data["phase"])
            push!(tracker.state.stages, stage)
        end
        
        # Restore state
        tracker.state.current_stage_index = data["current_stage_index"]
        tracker.state.overall_total = data["overall_total"]
        tracker.state.overall_completed = data["overall_completed"]
        tracker.state.start_time = DateTime(data["start_time"])
        tracker.state.last_update = DateTime(data["last_update"])
        tracker.state.total_pause_duration = Millisecond(data["total_pause_duration"])
        
        return true
    catch e
        @warn "Failed to load progress" error=e
        return false
    end
end

end # module