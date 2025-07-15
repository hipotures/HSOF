# Example usage of Keyboard Handler System

using Dates

# Include the module
include("../src/ui/keyboard_handler.jl")
using .KeyboardHandler

"""
Basic keyboard handler demo
"""
function basic_keyboard_demo()
    println("Basic Keyboard Handler Demo")
    println("==========================")
    
    # Create keyboard handler
    handler = create_keyboard_handler()
    
    println("\nKeyboard handler created with default configuration:")
    println("  Vim navigation: $(handler.state.config.enable_vim_navigation)")
    println("  Response timeout: $(handler.state.config.response_timeout_ms)ms")
    println("  Command key: '$(handler.state.config.command_key)'")
    println("  Help key: '$(handler.state.config.help_key)'")
    
    # Simulate some key presses
    println("\nSimulating key presses:")
    
    # Toggle pause
    println("  Pressing space to pause...")
    process_key(handler, ' ')
    println("  Paused: $(is_paused(handler))")
    
    # Toggle pause again
    println("  Pressing space to resume...")
    process_key(handler, ' ')
    println("  Paused: $(is_paused(handler))")
    
    # Show help
    println("  Pressing '?' to show help...")
    process_key(handler, '?')
    println("  Help visible: $(handler.state.help_visible)")
    
    # Hide help
    println("  Pressing 'q' to hide help...")
    process_key(handler, 'q')
    println("  Help visible: $(handler.state.help_visible)")
end

"""
Panel focus management demo
"""
function panel_focus_demo()
    println("\n\nPanel Focus Management Demo")
    println("==========================")
    
    handler = create_keyboard_handler()
    
    # Add focusable panels
    panels = [:gpu_status, :search_progress, :performance, :feature_analysis, :system_log]
    for panel in panels
        add_focusable_panel!(handler, panel)
    end
    
    println("Added $(length(panels)) focusable panels:")
    for (i, panel) in enumerate(panels)
        println("  $i. $panel")
    end
    
    # Show current focus
    focus_info = get_focus_info(handler)
    println("\nCurrent focus: $(focus_info.current)")
    
    # Navigate with Tab
    println("\nNavigating with Tab:")
    for i in 1:3
        process_key(handler, '\t')
        focus_info = get_focus_info(handler)
        println("  Tab → Focus on: $(focus_info.current)")
    end
    
    # Navigate with Shift+Tab
    println("\nNavigating with Shift+Tab:")
    for i in 1:2
        process_key(handler, '\t', Set([:shift]))
        focus_info = get_focus_info(handler)
        println("  Shift+Tab → Focus on: $(focus_info.current)")
    end
    
    # Direct focus setting
    println("\nDirect focus setting:")
    set_focus!(handler.state.panel_focus, :system_log)
    focus_info = get_focus_info(handler)
    println("  Set focus to :system_log → Current: $(focus_info.current)")
end

"""
Command palette demo
"""
function command_palette_demo()
    println("\n\nCommand Palette Demo")
    println("===================")
    
    handler = create_keyboard_handler()
    
    # Register custom commands
    test_value = ""
    register_command!(handler, "test", 
        (args...) -> (global test_value = "Test command with $(length(args)) args"))
    
    register_command!(handler, "echo",
        (args...) -> println("Echo: $(join(args, " "))"))
    
    println("Registered custom commands: test, echo")
    
    # Simulate command palette interaction
    println("\nOpening command palette with ':'...")
    process_key(handler, ':')
    println("  Command palette visible: $(handler.state.command_palette.visible)")
    
    # Type a command
    println("\nTyping 'help'...")
    for char in "help"
        process_key(handler, char)
    end
    println("  Command buffer: '$(handler.state.command_palette.command_buffer)'")
    
    # Execute command
    println("\nPressing Enter to execute...")
    process_key(handler, :enter)
    println("  Help visible: $(handler.state.help_visible)")
    println("  Command palette visible: $(handler.state.command_palette.visible)")
    
    # Hide help
    hide_help_overlay(handler)
    
    # Test custom command
    println("\nTesting custom command...")
    execute_command(handler, "test arg1 arg2 arg3")
    println("  Result: $test_value")
    
    execute_command(handler, "echo Hello from command palette!")
end

"""
Vim navigation demo
"""
function vim_navigation_demo()
    println("\n\nVim Navigation Demo")
    println("==================")
    
    # Create handler with vim navigation enabled (default)
    handler = create_keyboard_handler()
    
    # Add a panel for navigation context
    add_focusable_panel!(handler, :text_panel)
    
    println("Vim navigation enabled. Available keys:")
    println("  h - Navigate left")
    println("  j - Navigate down")
    println("  k - Navigate up")
    println("  l - Navigate right")
    println("  gg - Go to top")
    println("  G - Go to bottom")
    println("  Ctrl+d - Page down")
    println("  Ctrl+u - Page up")
    
    # Simulate navigation
    println("\nSimulating navigation commands:")
    
    keys = ['h', 'j', 'k', 'l']
    for key in keys
        println("  Pressing '$key'...")
        process_key(handler, key)
    end
    
    # Test multi-key sequence
    println("\nTesting multi-key sequence 'gg':")
    println("  Pressing 'g'...")
    process_key(handler, 'g')
    println("  Key buffer: '$(handler.state.key_buffer)'")
    println("  Pressing 'g' again...")
    process_key(handler, 'g')
    println("  Key buffer: '$(handler.state.key_buffer)' (should be empty after execution)")
    
    # Test page navigation
    println("\nTesting page navigation:")
    println("  Pressing Ctrl+d...")
    process_key(handler, 'd', Set([:ctrl]))
    println("  Pressing Ctrl+u...")
    process_key(handler, 'u', Set([:ctrl]))
end

"""
Custom hotkey registration demo
"""
function custom_hotkey_demo()
    println("\n\nCustom Hotkey Registration Demo")
    println("==============================")
    
    handler = create_keyboard_handler()
    
    # Define custom actions
    custom_state = Dict{String, Any}("counter" => 0, "last_action" => "")
    
    # Register custom hotkeys
    register_hotkey!(handler, 'i', 
        () -> (custom_state["counter"] += 1; custom_state["last_action"] = "increment"),
        description = "Increment counter")
    
    register_hotkey!(handler, 'd',
        () -> (custom_state["counter"] -= 1; custom_state["last_action"] = "decrement"),
        description = "Decrement counter",
        modifiers = Set([:ctrl]))
    
    register_hotkey!(handler, 'z',
        () -> (custom_state["counter"] = 0; custom_state["last_action"] = "reset"),
        description = "Reset counter",
        modifiers = Set([:ctrl, :shift]))
    
    println("Registered custom hotkeys:")
    println("  i - Increment counter")
    println("  Ctrl+d - Decrement counter")
    println("  Ctrl+Shift+z - Reset counter")
    
    # Test hotkeys
    println("\nTesting custom hotkeys:")
    println("  Initial counter: $(custom_state["counter"])")
    
    # Increment
    for _ in 1:3
        process_key(handler, 'i')
    end
    println("  After 3x 'i': counter = $(custom_state["counter"]), last = $(custom_state["last_action"])")
    
    # Decrement
    process_key(handler, 'd', Set([:ctrl]))
    println("  After Ctrl+d: counter = $(custom_state["counter"]), last = $(custom_state["last_action"])")
    
    # Reset
    process_key(handler, 'z', Set([:ctrl, :shift]))
    println("  After Ctrl+Shift+z: counter = $(custom_state["counter"]), last = $(custom_state["last_action"])")
end

"""
Help overlay demo
"""
function help_overlay_demo()
    println("\n\nHelp Overlay Demo")
    println("================")
    
    handler = create_keyboard_handler()
    
    # Get help text
    help_text = KeyboardHandler.get_help_text(handler)
    
    println("Help overlay content:")
    println("─" ^ 60)
    println(help_text)
    println("─" ^ 60)
    
    # Show different ways to toggle help
    println("\nToggling help overlay:")
    
    # Method 1: Using '?'
    println("  Method 1 - Press '?':")
    process_key(handler, '?')
    println("    Help visible: $(handler.state.help_visible)")
    process_key(handler, 'q')
    println("    After 'q': $(handler.state.help_visible)")
    
    # Method 2: Using command
    println("  Method 2 - Command 'help':")
    execute_command(handler, "help")
    println("    Help visible: $(handler.state.help_visible)")
    hide_help_overlay(handler)
    println("    After hide: $(handler.state.help_visible)")
    
    # Method 3: Direct function
    println("  Method 3 - Direct function:")
    toggle_help(handler)
    println("    After toggle: $(handler.state.help_visible)")
    toggle_help(handler)
    println("    After toggle again: $(handler.state.help_visible)")
end

"""
Response time monitoring demo
"""
function response_time_demo()
    println("\n\nResponse Time Monitoring Demo")
    println("============================")
    
    # Create handler with strict response time
    handler = create_keyboard_handler(response_timeout_ms = 50)
    
    println("Created handler with 50ms response timeout")
    
    # Process keys with different delays
    println("\nProcessing keys with simulated delays:")
    
    # Fast response
    println("  Fast response (10ms delay):")
    process_key(handler, 'a')
    sleep(0.01)
    process_key(handler, 'b')
    
    # Slow response (will trigger warning)
    println("  Slow response (60ms delay):")
    sleep(0.06)
    println("    (Should see warning about exceeding threshold)")
    process_key(handler, 'c')
    
    # Normal response
    println("  Normal response (30ms delay):")
    sleep(0.03)
    process_key(handler, 'd')
end

"""
Interactive keyboard test
"""
function interactive_keyboard_test()
    println("\n\nInteractive Keyboard Test")
    println("========================")
    println("This would be an interactive test where actual keyboard input is captured.")
    println("In a real terminal application, you would:")
    println("1. Set terminal to raw mode")
    println("2. Capture individual key presses")
    println("3. Process them through the handler")
    println("4. Update the display based on actions")
    println("\nExample pseudo-code:")
    println("""
    while handler.active
        key = read_keyboard_input()  # Platform-specific
        modifiers = get_current_modifiers()
        process_key(handler, key, modifiers)
        
        if handler.interrupt_flag
            break
        end
        
        update_display()
    end
    """)
end

# Main menu
function main()
    println("Keyboard Handler System Examples")
    println("===============================")
    println("1. Basic Keyboard Handler")
    println("2. Panel Focus Management")
    println("3. Command Palette")
    println("4. Vim Navigation")
    println("5. Custom Hotkey Registration")
    println("6. Help Overlay")
    println("7. Response Time Monitoring")
    println("8. Interactive Test (Description)")
    println("9. Run All Demos")
    println("\nSelect demo (1-9): ")
    
    choice = readline()
    
    if choice == "1"
        basic_keyboard_demo()
    elseif choice == "2"
        panel_focus_demo()
    elseif choice == "3"
        command_palette_demo()
    elseif choice == "4"
        vim_navigation_demo()
    elseif choice == "5"
        custom_hotkey_demo()
    elseif choice == "6"
        help_overlay_demo()
    elseif choice == "7"
        response_time_demo()
    elseif choice == "8"
        interactive_keyboard_test()
    elseif choice == "9"
        basic_keyboard_demo()
        panel_focus_demo()
        command_palette_demo()
        vim_navigation_demo()
        custom_hotkey_demo()
        help_overlay_demo()
        println("\nSkipping response time demo in batch mode...")
        interactive_keyboard_test()
    else
        println("Invalid choice. Running basic demo...")
        basic_keyboard_demo()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end