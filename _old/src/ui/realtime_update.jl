module RealtimeUpdate

using Term
using Dates
using Printf

# Re-export necessary types from ConsoleDashboard
include("console_dashboard.jl")
using .ConsoleDashboard

export LiveDashboard, DashboardUpdateConfig
export start_live_dashboard!, stop_live_dashboard!, update_live_data!
export is_running, get_update_stats, reset_stats!

"""
Configuration for real-time dashboard updates
"""
struct DashboardUpdateConfig
    update_interval_ms::Int      # Target update interval in milliseconds
    enable_double_buffer::Bool   # Use double buffering for smoother updates
    enable_delta_updates::Bool   # Only update changed panels
    max_queue_size::Int         # Maximum update queue size
    performance_tracking::Bool   # Track update performance metrics
    
    function DashboardUpdateConfig(;
        update_interval_ms::Int = 100,
        enable_double_buffer::Bool = true,
        enable_delta_updates::Bool = true,
        max_queue_size::Int = 100,
        performance_tracking::Bool = true
    )
        @assert update_interval_ms > 0 "Update interval must be positive"
        @assert max_queue_size > 0 "Queue size must be positive"
        
        new(update_interval_ms, enable_double_buffer, enable_delta_updates,
            max_queue_size, performance_tracking)
    end
end

"""
Live dashboard manager with real-time update capabilities
"""
mutable struct LiveDashboard
    dashboard::DashboardLayout
    config::DashboardUpdateConfig
    update_task::Union{Task, Nothing}
    is_running::Bool
    update_queue::Channel{Dict{Symbol, PanelContent}}
    last_update_time::DateTime
    frame_count::Int
    total_update_time_ms::Float64
    min_frame_time_ms::Float64
    max_frame_time_ms::Float64
    dropped_frames::Int
    previous_content_hash::Dict{Symbol, UInt64}
    
    function LiveDashboard(dashboard::DashboardLayout, config::DashboardUpdateConfig = DashboardUpdateConfig())
        update_queue = Channel{Dict{Symbol, PanelContent}}(config.max_queue_size)
        previous_content_hash = Dict{Symbol, UInt64}()
        
        # Initialize content hashes
        for (key, content) in dashboard.panel_contents
            previous_content_hash[key] = hash(content)
        end
        
        new(dashboard, config, nothing, false, update_queue, now(), 0, 0.0, 
            Inf, 0.0, 0, previous_content_hash)
    end
end

"""
Start the live dashboard update loop
"""
function start_live_dashboard!(live_dashboard::LiveDashboard)
    if live_dashboard.is_running
        @warn "Dashboard is already running"
        return
    end
    
    live_dashboard.is_running = true
    live_dashboard.last_update_time = now()
    
    # Clear screen and hide cursor
    print("\033[2J\033[H\033[?25l")
    
    # Start update task
    live_dashboard.update_task = @async begin
        try
            update_loop(live_dashboard)
        catch e
            @error "Update loop error" exception=e
            live_dashboard.is_running = false
            rethrow(e)
        finally
            # Show cursor on exit
            print("\033[?25h")
        end
    end
    
    return live_dashboard
end

"""
Stop the live dashboard
"""
function stop_live_dashboard!(live_dashboard::LiveDashboard)
    if !live_dashboard.is_running
        @warn "Dashboard is not running"
        return
    end
    
    live_dashboard.is_running = false
    
    # Close update queue
    close(live_dashboard.update_queue)
    
    # Wait for task to finish
    if !isnothing(live_dashboard.update_task)
        wait(live_dashboard.update_task)
    end
    
    # Show cursor
    print("\033[?25h")
    
    return live_dashboard
end

"""
Main update loop
"""
function update_loop(live_dashboard::LiveDashboard)
    config = live_dashboard.config
    dashboard = live_dashboard.dashboard
    
    # Frame timing
    target_frame_time = config.update_interval_ms / 1000.0
    last_frame_time = time()
    
    # Double buffering
    buffer_A = nothing
    buffer_B = nothing
    active_buffer = :A
    
    while live_dashboard.is_running
        frame_start_time = time()
        
        # Process update queue
        updates_processed = process_update_queue!(live_dashboard)
        
        # Check if any panels need updating
        panels_to_update = if config.enable_delta_updates
            find_changed_panels(live_dashboard)
        else
            collect(keys(dashboard.panels))
        end
        
        if !isempty(panels_to_update)
            # Prepare render in buffer
            if config.enable_double_buffer
                # Render to inactive buffer
                inactive_buffer = active_buffer == :A ? :B : :A
                rendered = prepare_render(dashboard, panels_to_update)
                
                if inactive_buffer == :A
                    buffer_A = rendered
                else
                    buffer_B = rendered
                end
                
                # Swap buffers
                active_buffer = inactive_buffer
            else
                # Direct render
                rendered = render_dashboard(dashboard)
                
                # Clear and draw
                print("\033[2J\033[H")
                println(rendered)
            end
            
            # Display active buffer
            if config.enable_double_buffer
                current_buffer = active_buffer == :A ? buffer_A : buffer_B
                if !isnothing(current_buffer)
                    print("\033[2J\033[H")
                    println(current_buffer)
                end
            end
            
            # Update performance stats
            update_performance_stats!(live_dashboard, frame_start_time)
        end
        
        # Frame rate control
        frame_end_time = time()
        frame_duration = frame_end_time - frame_start_time
        sleep_time = target_frame_time - frame_duration
        
        if sleep_time > 0
            sleep(sleep_time)
        else
            # Frame took too long
            live_dashboard.dropped_frames += 1
        end
        
        # Yield to other tasks
        yield()
    end
end

"""
Process pending updates from the queue
"""
function process_update_queue!(live_dashboard::LiveDashboard)
    updates_processed = 0
    max_updates_per_frame = 10  # Prevent queue flooding
    
    while isready(live_dashboard.update_queue) && updates_processed < max_updates_per_frame
        try
            updates = take!(live_dashboard.update_queue)
            update_dashboard!(live_dashboard.dashboard, updates)
            updates_processed += 1
        catch e
            if !(e isa InvalidStateException)  # Channel closed
                @error "Error processing update" exception=e
            end
            break
        end
    end
    
    return updates_processed
end

"""
Find panels that have changed content
"""
function find_changed_panels(live_dashboard::LiveDashboard)
    changed_panels = Symbol[]
    dashboard = live_dashboard.dashboard
    
    for (key, content) in dashboard.panel_contents
        current_hash = hash(content)
        previous_hash = get(live_dashboard.previous_content_hash, key, UInt64(0))
        
        if current_hash != previous_hash
            push!(changed_panels, key)
            live_dashboard.previous_content_hash[key] = current_hash
        end
    end
    
    return changed_panels
end

"""
Prepare render for specific panels only
"""
function prepare_render(dashboard::DashboardLayout, panels_to_update::Vector{Symbol})
    # For now, render everything - optimization can be added later
    return render_dashboard(dashboard)
end

"""
Update performance statistics
"""
function update_performance_stats!(live_dashboard::LiveDashboard, frame_start_time::Float64)
    frame_time_ms = (time() - frame_start_time) * 1000
    
    live_dashboard.frame_count += 1
    live_dashboard.total_update_time_ms += frame_time_ms
    live_dashboard.min_frame_time_ms = min(live_dashboard.min_frame_time_ms, frame_time_ms)
    live_dashboard.max_frame_time_ms = max(live_dashboard.max_frame_time_ms, frame_time_ms)
end

"""
Queue updates for the dashboard
"""
function update_live_data!(live_dashboard::LiveDashboard, updates::Dict{Symbol, PanelContent})
    if !live_dashboard.is_running
        @warn "Dashboard is not running, starting it..."
        start_live_dashboard!(live_dashboard)
    end
    
    try
        # Try to put update in queue, drop if full
        if isready(live_dashboard.update_queue) && 
           length(live_dashboard.update_queue.data) >= live_dashboard.config.max_queue_size
            # Queue is full, drop oldest
            take!(live_dashboard.update_queue)
        end
        
        put!(live_dashboard.update_queue, updates)
    catch e
        if e isa InvalidStateException
            @warn "Update queue is closed"
        else
            rethrow(e)
        end
    end
end

"""
Check if dashboard is running
"""
function is_running(live_dashboard::LiveDashboard)
    return live_dashboard.is_running
end

"""
Get update performance statistics
"""
function get_update_stats(live_dashboard::LiveDashboard)
    if live_dashboard.frame_count == 0
        return (
            avg_frame_time_ms = 0.0,
            min_frame_time_ms = 0.0,
            max_frame_time_ms = 0.0,
            fps = 0.0,
            dropped_frames = 0,
            uptime_seconds = 0.0
        )
    end
    
    avg_frame_time = live_dashboard.total_update_time_ms / live_dashboard.frame_count
    fps = live_dashboard.frame_count > 0 ? 1000.0 / avg_frame_time : 0.0
    uptime = (now() - live_dashboard.last_update_time).value / 1000.0  # Convert to seconds
    
    return (
        avg_frame_time_ms = avg_frame_time,
        min_frame_time_ms = live_dashboard.min_frame_time_ms,
        max_frame_time_ms = live_dashboard.max_frame_time_ms,
        fps = fps,
        dropped_frames = live_dashboard.dropped_frames,
        uptime_seconds = uptime
    )
end

"""
Reset performance statistics
"""
function reset_stats!(live_dashboard::LiveDashboard)
    live_dashboard.frame_count = 0
    live_dashboard.total_update_time_ms = 0.0
    live_dashboard.min_frame_time_ms = Inf
    live_dashboard.max_frame_time_ms = 0.0
    live_dashboard.dropped_frames = 0
    live_dashboard.last_update_time = now()
end

"""
Graceful shutdown handler
"""
function setup_signal_handlers(live_dashboard::LiveDashboard)
    # Handle Ctrl+C gracefully
    Base.exit_on_sigint(false)
    
    @async begin
        try
            while live_dashboard.is_running
                sleep(0.1)
            end
        catch e
            if e isa InterruptException
                println("\nShutting down dashboard...")
                stop_live_dashboard!(live_dashboard)
            else
                rethrow(e)
            end
        end
    end
end

end # module