module KeyboardHandler

using Dates

export KeyboardConfig, KeyboardState, KeyHandler
export create_keyboard_handler, process_key, register_hotkey!
export set_focus!, next_focus!, prev_focus!, get_focused_panel
export show_help_overlay, hide_help_overlay, toggle_help
export execute_command, show_command_palette
export add_focusable_panel!, remove_focusable_panel!, get_focus_info
export register_command!, is_paused

"""
Keyboard configuration and hotkey mappings
"""
struct KeyboardConfig
    enable_vim_navigation::Bool
    response_timeout_ms::Int
    command_key::Char  # Key to open command palette (default ':')
    help_key::Char     # Key to show help (default '?')
    
    function KeyboardConfig(;
        enable_vim_navigation::Bool = true,
        response_timeout_ms::Int = 100,
        command_key::Char = ':',
        help_key::Char = '?'
    )
        @assert response_timeout_ms > 0 "Response timeout must be positive"
        new(enable_vim_navigation, response_timeout_ms, command_key, help_key)
    end
end

"""
Hotkey binding structure
"""
struct HotkeyBinding
    key::Union{Char, Symbol}  # Key or special key symbol
    modifiers::Set{Symbol}    # Set of modifiers (:ctrl, :alt, :shift)
    description::String       # Description for help
    action::Function         # Function to execute
    enabled::Bool            # Whether the hotkey is currently active
    
    function HotkeyBinding(key::Union{Char, Symbol}, action::Function;
                          modifiers::Set{Symbol} = Set{Symbol}(),
                          description::String = "",
                          enabled::Bool = true)
        new(key, modifiers, description, action, enabled)
    end
end

"""
Panel focus information
"""
mutable struct PanelFocus
    panels::Vector{Symbol}           # List of focusable panels
    current_index::Int              # Currently focused panel index
    focus_indicators::Dict{Symbol, Bool}  # Which panels show focus indicator
    
    function PanelFocus(panels::Vector{Symbol} = Symbol[])
        indicators = Dict{Symbol, Bool}()
        for panel in panels
            indicators[panel] = false
        end
        if !isempty(panels)
            indicators[panels[1]] = true
        end
        new(panels, isempty(panels) ? 0 : 1, indicators)
    end
end

"""
Command palette state
"""
mutable struct CommandPalette
    visible::Bool
    commands::Dict{String, Function}
    command_buffer::String
    history::Vector{String}
    history_index::Int
    max_history::Int
    
    function CommandPalette(max_history::Int = 50)
        new(false, Dict{String, Function}(), "", String[], 0, max_history)
    end
end

"""
Main keyboard state manager
"""
mutable struct KeyboardState
    config::KeyboardConfig
    hotkeys::Dict{String, HotkeyBinding}  # Key string to binding
    panel_focus::PanelFocus
    command_palette::CommandPalette
    help_visible::Bool
    paused::Bool
    last_key_time::DateTime
    key_buffer::String  # For multi-key sequences
    recording_macro::Bool
    macro_buffer::Vector{String}
    
    function KeyboardState(config::KeyboardConfig = KeyboardConfig())
        new(config, Dict{String, HotkeyBinding}(), PanelFocus(), 
            CommandPalette(), false, false, now(), "", false, String[])
    end
end

"""
Main keyboard handler
"""
mutable struct KeyHandler
    state::KeyboardState
    active::Bool
    interrupt_flag::Bool
    
    function KeyHandler(config::KeyboardConfig = KeyboardConfig())
        new(KeyboardState(config), true, false)
    end
end

# Core Functions

"""
Create a new keyboard handler
"""
function create_keyboard_handler(;kwargs...)
    config = KeyboardConfig(;kwargs...)
    handler = KeyHandler(config)
    
    # Register default hotkeys
    register_default_hotkeys!(handler)
    
    return handler
end

"""
Register default hotkeys
"""
function register_default_hotkeys!(handler::KeyHandler)
    # Core controls
    register_hotkey!(handler, ' ', () -> toggle_pause!(handler), 
                    description="Pause/Resume")
    register_hotkey!(handler, 'q', () -> quit_handler(handler), 
                    description="Quit")
    register_hotkey!(handler, 'r', () -> reset_stats(handler), 
                    description="Reset Statistics")
    
    # Panel toggles
    for i in 1:6
        panel_num = i
        register_hotkey!(handler, Char('0' + i), 
                        () -> toggle_panel(handler, panel_num),
                        description="Toggle Panel $i")
    end
    
    # Navigation
    register_hotkey!(handler, '\t', () -> next_focus!(handler.state.panel_focus),
                    description="Next Panel (Tab)")
    register_hotkey!(handler, '\t', () -> prev_focus!(handler.state.panel_focus),
                    modifiers=Set([:shift]), description="Previous Panel (Shift+Tab)")
    
    # Vim navigation (if enabled)
    if handler.state.config.enable_vim_navigation
        register_vim_navigation!(handler)
    end
    
    # Help and command palette
    register_hotkey!(handler, handler.state.config.help_key, 
                    () -> toggle_help(handler),
                    description="Toggle Help")
    register_hotkey!(handler, handler.state.config.command_key,
                    () -> show_command_palette(handler),
                    description="Command Palette")
    
    # Advanced controls
    register_hotkey!(handler, 'e', () -> export_data(handler),
                    description="Export Data")
    register_hotkey!(handler, 's', () -> save_state(handler),
                    description="Save State")
    register_hotkey!(handler, 'c', () -> show_config(handler),
                    description="Configuration")
end

"""
Register vim-style navigation keys
"""
function register_vim_navigation!(handler::KeyHandler)
    register_hotkey!(handler, 'h', () -> navigate_left(handler),
                    description="Navigate Left")
    register_hotkey!(handler, 'j', () -> navigate_down(handler),
                    description="Navigate Down")
    register_hotkey!(handler, 'k', () -> navigate_up(handler),
                    description="Navigate Up")
    register_hotkey!(handler, 'l', () -> navigate_right(handler),
                    description="Navigate Right")
    
    # Page navigation
    register_hotkey!(handler, 'd', () -> page_down(handler),
                    modifiers=Set([:ctrl]), description="Page Down")
    register_hotkey!(handler, 'u', () -> page_up(handler),
                    modifiers=Set([:ctrl]), description="Page Up")
    
    # Line navigation (gg is handled as multi-key sequence)
    handler.state.hotkeys["gg"] = HotkeyBinding('g', () -> go_to_top(handler),
                                               description="Go to Top (gg)")
    register_hotkey!(handler, 'G', () -> go_to_bottom(handler),
                    description="Go to Bottom")
end

"""
Register a hotkey binding
"""
function register_hotkey!(handler::KeyHandler, key::Union{Char, Symbol}, 
                         action::Function; kwargs...)
    binding = HotkeyBinding(key, action; kwargs...)
    key_string = hotkey_to_string(key, get(kwargs, :modifiers, Set{Symbol}()))
    handler.state.hotkeys[key_string] = binding
end

"""
Convert hotkey to string representation
"""
function hotkey_to_string(key::Union{Char, Symbol}, modifiers::Set{Symbol})
    mod_parts = String[]
    :ctrl in modifiers && push!(mod_parts, "ctrl")
    :alt in modifiers && push!(mod_parts, "alt")
    :shift in modifiers && push!(mod_parts, "shift")
    
    if isempty(mod_parts)
        return string(key)
    else
        return join(mod_parts, "+") * "+" * string(key)
    end
end

"""
Process a keyboard input
"""
function process_key(handler::KeyHandler, key::Union{Char, Symbol}, 
                    modifiers::Set{Symbol} = Set{Symbol}())
    if !handler.active
        return false
    end
    
    # Update timing
    current_time = now()
    elapsed_ms = Dates.value(current_time - handler.state.last_key_time)
    handler.state.last_key_time = current_time
    
    # Check response time
    if elapsed_ms > handler.state.config.response_timeout_ms
        @warn "Keyboard response time exceeded threshold" elapsed_ms=elapsed_ms
    end
    
    # Handle command palette input
    if handler.state.command_palette.visible
        return process_command_palette_key(handler, key, modifiers)
    end
    
    # Handle help overlay
    if handler.state.help_visible && key == 'q'
        hide_help_overlay(handler)
        return true
    end
    
    # Look up hotkey
    key_string = hotkey_to_string(key, modifiers)
    
    # Record any key press for macro (before processing)
    if handler.state.recording_macro && isa(key, Char)
        record_macro_key(handler, key_string)
    end
    
    # Check for multi-key sequences (like 'gg')
    if !isempty(handler.state.key_buffer)
        combined_key = handler.state.key_buffer * string(key)
        if haskey(handler.state.hotkeys, combined_key)
            binding = handler.state.hotkeys[combined_key]
            if binding.enabled
                binding.action()
                handler.state.key_buffer = ""
                return true
            end
        end
        # Clear buffer if no match
        handler.state.key_buffer = ""
    end
    
    # Check single key
    if haskey(handler.state.hotkeys, key_string)
        binding = handler.state.hotkeys[key_string]
        if binding.enabled
            binding.action()
            return true
        end
    end
    
    # Check if this might be start of multi-key sequence
    if key == 'g' && handler.state.config.enable_vim_navigation
        handler.state.key_buffer = "g"
        return true
    end
    
    return false
end

"""
Process command palette keyboard input
"""
function process_command_palette_key(handler::KeyHandler, key::Union{Char, Symbol},
                                   modifiers::Set{Symbol})
    palette = handler.state.command_palette
    
    if key == :escape || (key == 'c' && :ctrl in modifiers)
        # Cancel command palette
        hide_command_palette(handler)
        return true
    elseif key == :enter || key == '\r'
        # Execute command
        execute_command_from_palette(handler)
        return true
    elseif key == :backspace
        # Delete character
        if !isempty(palette.command_buffer)
            palette.command_buffer = palette.command_buffer[1:end-1]
        end
        return true
    elseif key == :up
        # Previous command in history
        navigate_command_history(handler, -1)
        return true
    elseif key == :down
        # Next command in history
        navigate_command_history(handler, 1)
        return true
    elseif isa(key, Char) && isprint(key)
        # Add character to buffer
        palette.command_buffer *= key
        return true
    end
    
    return false
end

# Focus Management

"""
Set focus to specific panel
"""
function set_focus!(focus::PanelFocus, panel::Symbol)
    if panel in focus.panels
        # Clear all indicators
        for p in keys(focus.focus_indicators)
            focus.focus_indicators[p] = false
        end
        
        # Set new focus
        index = findfirst(==(panel), focus.panels)
        if !isnothing(index)
            focus.current_index = index
            focus.focus_indicators[panel] = true
        end
    end
end

"""
Move focus to next panel
"""
function next_focus!(focus::PanelFocus)
    if isempty(focus.panels)
        return
    end
    
    # Clear current focus
    if focus.current_index > 0
        current_panel = focus.panels[focus.current_index]
        focus.focus_indicators[current_panel] = false
    end
    
    # Move to next
    focus.current_index = mod1(focus.current_index + 1, length(focus.panels))
    new_panel = focus.panels[focus.current_index]
    focus.focus_indicators[new_panel] = true
end

"""
Move focus to previous panel
"""
function prev_focus!(focus::PanelFocus)
    if isempty(focus.panels)
        return
    end
    
    # Clear current focus
    if focus.current_index > 0
        current_panel = focus.panels[focus.current_index]
        focus.focus_indicators[current_panel] = false
    end
    
    # Move to previous
    focus.current_index = mod1(focus.current_index - 1, length(focus.panels))
    new_panel = focus.panels[focus.current_index]
    focus.focus_indicators[new_panel] = true
end

"""
Get currently focused panel
"""
function get_focused_panel(focus::PanelFocus)
    if focus.current_index > 0 && focus.current_index <= length(focus.panels)
        return focus.panels[focus.current_index]
    end
    return nothing
end

# Command Palette Functions

"""
Show command palette
"""
function show_command_palette(handler::KeyHandler)
    handler.state.command_palette.visible = true
    handler.state.command_palette.command_buffer = ""
end

"""
Hide command palette
"""
function hide_command_palette(handler::KeyHandler)
    handler.state.command_palette.visible = false
    handler.state.command_palette.command_buffer = ""
end

"""
Execute command from palette
"""
function execute_command_from_palette(handler::KeyHandler)
    palette = handler.state.command_palette
    command = strip(palette.command_buffer)
    
    if !isempty(command)
        # Add to history
        push!(palette.history, command)
        if length(palette.history) > palette.max_history
            popfirst!(palette.history)
        end
        palette.history_index = length(palette.history) + 1
        
        # Execute command
        execute_command(handler, String(command))
    end
    
    hide_command_palette(handler)
end

"""
Navigate command history
"""
function navigate_command_history(handler::KeyHandler, direction::Int)
    palette = handler.state.command_palette
    
    if isempty(palette.history)
        return
    end
    
    new_index = palette.history_index + direction
    if 1 <= new_index <= length(palette.history) + 1
        palette.history_index = new_index
        
        if new_index <= length(palette.history)
            palette.command_buffer = palette.history[new_index]
        else
            palette.command_buffer = ""
        end
    end
end

"""
Execute a command string
"""
function execute_command(handler::KeyHandler, command::String)
    parts = split(command)
    if isempty(parts)
        return
    end
    
    cmd = parts[1]
    args = parts[2:end]
    
    # Built-in commands
    if cmd == "quit" || cmd == "q"
        quit_handler(handler)
    elseif cmd == "help" || cmd == "h"
        show_help_overlay(handler)
    elseif cmd == "save"
        save_state(handler)
    elseif cmd == "export"
        export_data(handler)
    elseif cmd == "panel"
        if !isempty(args)
            panel_num = tryparse(Int, args[1])
            if !isnothing(panel_num) && 1 <= panel_num <= 6
                toggle_panel(handler, panel_num)
            end
        end
    elseif haskey(handler.state.command_palette.commands, cmd)
        # Custom command
        handler.state.command_palette.commands[cmd](args...)
    else
        @warn "Unknown command: $cmd"
    end
end

"""
Register custom command
"""
function register_command!(handler::KeyHandler, name::String, func::Function)
    handler.state.command_palette.commands[name] = func
end

# Help Overlay

"""
Show help overlay
"""
function show_help_overlay(handler::KeyHandler)
    handler.state.help_visible = true
end

"""
Hide help overlay
"""
function hide_help_overlay(handler::KeyHandler)
    handler.state.help_visible = false
end

"""
Toggle help overlay
"""
function toggle_help(handler::KeyHandler)
    handler.state.help_visible = !handler.state.help_visible
end

"""
Get help text for display
"""
function get_help_text(handler::KeyHandler)
    lines = String[]
    push!(lines, "Keyboard Shortcuts")
    push!(lines, "==================")
    push!(lines, "")
    
    # Group hotkeys by category
    categories = Dict{String, Vector{Tuple{String, String}}}()
    categories["Navigation"] = []
    categories["Controls"] = []
    categories["Panels"] = []
    categories["Advanced"] = []
    
    for (key_string, binding) in handler.state.hotkeys
        if !binding.enabled
            continue
        end
        
        # Determine category
        category = if binding.key in ['h', 'j', 'k', 'l', '\t', 'g', 'G']
            "Navigation"
        elseif binding.key in ['1', '2', '3', '4', '5', '6']
            "Panels"
        elseif binding.key in [' ', 'q', 'r']
            "Controls"
        else
            "Advanced"
        end
        
        # Format key display
        key_display = key_string == " " ? "Space" : key_string
        key_display = replace(key_display, "+" => " + ")
        key_display = replace(key_display, "ctrl" => "Ctrl")
        key_display = replace(key_display, "shift" => "Shift")
        key_display = replace(key_display, "alt" => "Alt")
        
        push!(categories[category], (key_display, binding.description))
    end
    
    # Display categories
    for (category, bindings) in categories
        if !isempty(bindings)
            push!(lines, "$category:")
            for (key, desc) in sort(bindings)
                push!(lines, "  $(rpad(key, 15)) $desc")
            end
            push!(lines, "")
        end
    end
    
    push!(lines, "Press 'q' to close help")
    
    return join(lines, "\n")
end

# Action Handlers

"""
Toggle pause state
"""
function toggle_pause!(handler::KeyHandler)
    handler.state.paused = !handler.state.paused
end

"""
Quit handler
"""
function quit_handler(handler::KeyHandler)
    handler.interrupt_flag = true
    handler.active = false
end

"""
Reset statistics
"""
function reset_stats(handler::KeyHandler)
    # This would be implemented to reset dashboard statistics
    @info "Reset statistics requested"
end

"""
Toggle panel visibility
"""
function toggle_panel(handler::KeyHandler, panel_num::Int)
    # This would be implemented to toggle specific panel
    @info "Toggle panel $panel_num"
end

"""
Export data
"""
function export_data(handler::KeyHandler)
    @info "Export data requested"
end

"""
Save state
"""
function save_state(handler::KeyHandler)
    @info "Save state requested"
end

"""
Show configuration
"""
function show_config(handler::KeyHandler)
    @info "Show configuration requested"
end

# Navigation handlers

"""
Navigate left in focused panel
"""
function navigate_left(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Navigate left in panel $focused"
    end
end

"""
Navigate down in focused panel
"""
function navigate_down(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Navigate down in panel $focused"
    end
end

"""
Navigate up in focused panel
"""
function navigate_up(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Navigate up in panel $focused"
    end
end

"""
Navigate right in focused panel
"""
function navigate_right(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Navigate right in panel $focused"
    end
end

"""
Page down in focused panel
"""
function page_down(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Page down in panel $focused"
    end
end

"""
Page up in focused panel
"""
function page_up(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Page up in panel $focused"
    end
end

"""
Go to top of content
"""
function go_to_top(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Go to top in panel $focused"
    end
end

"""
Go to bottom of content
"""
function go_to_bottom(handler::KeyHandler)
    focused = get_focused_panel(handler.state.panel_focus)
    if !isnothing(focused)
        @info "Go to bottom in panel $focused"
    end
end

# Macro Recording

"""
Record key for macro
"""
function record_macro_key(handler::KeyHandler, key_string::String)
    if handler.state.recording_macro
        push!(handler.state.macro_buffer, key_string)
    end
end

"""
Start macro recording
"""
function start_macro_recording(handler::KeyHandler)
    handler.state.recording_macro = true
    handler.state.macro_buffer = String[]
end

"""
Stop macro recording
"""
function stop_macro_recording(handler::KeyHandler)
    handler.state.recording_macro = false
end

"""
Play recorded macro
"""
function play_macro(handler::KeyHandler)
    for key_string in handler.state.macro_buffer
        if haskey(handler.state.hotkeys, key_string)
            binding = handler.state.hotkeys[key_string]
            if binding.enabled
                binding.action()
            end
        end
    end
end

# Utility Functions

"""
Check if handler is paused
"""
function is_paused(handler::KeyHandler)
    return handler.state.paused
end

"""
Get current focus info
"""
function get_focus_info(handler::KeyHandler)
    focus = handler.state.panel_focus
    return (
        panels = copy(focus.panels),
        current = get_focused_panel(focus),
        indicators = copy(focus.focus_indicators)
    )
end

"""
Add focusable panel
"""
function add_focusable_panel!(handler::KeyHandler, panel::Symbol)
    focus = handler.state.panel_focus
    if !(panel in focus.panels)
        push!(focus.panels, panel)
        focus.focus_indicators[panel] = false
        
        # If this is the first panel, focus it
        if length(focus.panels) == 1
            set_focus!(focus, panel)
        end
    end
end

"""
Remove focusable panel
"""
function remove_focusable_panel!(handler::KeyHandler, panel::Symbol)
    focus = handler.state.panel_focus
    index = findfirst(==(panel), focus.panels)
    
    if !isnothing(index)
        deleteat!(focus.panels, index)
        delete!(focus.focus_indicators, panel)
        
        # Adjust current index if needed
        if focus.current_index > length(focus.panels)
            focus.current_index = length(focus.panels)
        end
        
        # Update focus indicators
        if !isempty(focus.panels) && focus.current_index > 0
            new_panel = focus.panels[focus.current_index]
            focus.focus_indicators[new_panel] = true
        end
    end
end

end # module