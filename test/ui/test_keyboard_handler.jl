using Test
using Dates

# Include the module
include("../../src/ui/keyboard_handler.jl")
using .KeyboardHandler

@testset "Keyboard Handler Tests" begin
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = KeyboardConfig()
        @test config.enable_vim_navigation == true
        @test config.response_timeout_ms == 100
        @test config.command_key == ':'
        @test config.help_key == '?'
        
        # Test custom configuration
        custom_config = KeyboardConfig(
            enable_vim_navigation = false,
            response_timeout_ms = 50,
            command_key = '/',
            help_key = 'h'
        )
        @test custom_config.enable_vim_navigation == false
        @test custom_config.response_timeout_ms == 50
        @test custom_config.command_key == '/'
        @test custom_config.help_key == 'h'
        
        # Test invalid configuration
        @test_throws AssertionError KeyboardConfig(response_timeout_ms = 0)
        @test_throws AssertionError KeyboardConfig(response_timeout_ms = -10)
    end
    
    @testset "Hotkey Binding Tests" begin
        # Test basic hotkey creation
        action_called = false
        action = () -> (action_called = true)
        binding = KeyboardHandler.HotkeyBinding('a', action, description="Test")
        
        @test binding.key == 'a'
        @test isempty(binding.modifiers)
        @test binding.description == "Test"
        @test binding.enabled == true
        
        # Test with modifiers
        binding_with_mods = KeyboardHandler.HotkeyBinding(
            'c', action,
            modifiers = Set([:ctrl]),
            description = "Copy",
            enabled = false
        )
        @test :ctrl in binding_with_mods.modifiers
        @test binding_with_mods.enabled == false
        
        # Test action execution
        binding.action()
        @test action_called == true
    end
    
    @testset "Panel Focus Tests" begin
        # Test empty panel focus
        focus = KeyboardHandler.PanelFocus()
        @test isempty(focus.panels)
        @test focus.current_index == 0
        @test isempty(focus.focus_indicators)
        
        # Test with panels
        panels = [:gpu1, :gpu2, :progress, :log]
        focus = KeyboardHandler.PanelFocus(panels)
        @test focus.panels == panels
        @test focus.current_index == 1
        @test focus.focus_indicators[:gpu1] == true
        @test focus.focus_indicators[:gpu2] == false
        
        # Test focus navigation
        next_focus!(focus)
        @test focus.current_index == 2
        @test focus.focus_indicators[:gpu1] == false
        @test focus.focus_indicators[:gpu2] == true
        
        # Test wrap around
        for _ in 1:3
            next_focus!(focus)
        end
        @test focus.current_index == 1
        @test focus.focus_indicators[:gpu1] == true
        
        # Test previous focus
        prev_focus!(focus)
        @test focus.current_index == 4
        @test focus.focus_indicators[:log] == true
        
        # Test set focus
        set_focus!(focus, :progress)
        @test focus.current_index == 3
        @test focus.focus_indicators[:progress] == true
        @test focus.focus_indicators[:log] == false
        
        # Test get focused panel
        @test get_focused_panel(focus) == :progress
    end
    
    @testset "Key Handler Creation" begin
        handler = create_keyboard_handler()
        @test isa(handler, KeyHandler)
        @test handler.active == true
        @test handler.interrupt_flag == false
        @test !isempty(handler.state.hotkeys)
        
        # Check default hotkeys are registered
        @test haskey(handler.state.hotkeys, " ")  # Space for pause
        @test haskey(handler.state.hotkeys, "q")  # Q for quit
        @test haskey(handler.state.hotkeys, "r")  # R for reset
        @test haskey(handler.state.hotkeys, "?")  # ? for help
        @test haskey(handler.state.hotkeys, ":")  # : for command palette
        
        # Check vim navigation (enabled by default)
        @test haskey(handler.state.hotkeys, "h")
        @test haskey(handler.state.hotkeys, "j")
        @test haskey(handler.state.hotkeys, "k")
        @test haskey(handler.state.hotkeys, "l")
    end
    
    @testset "Key Processing Tests" begin
        handler = create_keyboard_handler()
        
        # Test pause toggle
        @test handler.state.paused == false
        process_key(handler, ' ')
        @test handler.state.paused == true
        process_key(handler, ' ')
        @test handler.state.paused == false
        
        # Test help toggle
        @test handler.state.help_visible == false
        process_key(handler, '?')
        @test handler.state.help_visible == true
        process_key(handler, 'q')  # Q closes help when visible
        @test handler.state.help_visible == false
        
        # Test command palette
        @test handler.state.command_palette.visible == false
        process_key(handler, ':')
        @test handler.state.command_palette.visible == true
        
        # Test escape from command palette
        process_key(handler, :escape)
        @test handler.state.command_palette.visible == false
        
        # Test inactive handler
        handler.active = false
        result = process_key(handler, 'a')
        @test result == false
    end
    
    @testset "Hotkey String Conversion" begin
        # Test simple keys
        @test KeyboardHandler.hotkey_to_string('a', Set{Symbol}()) == "a"
        @test KeyboardHandler.hotkey_to_string(' ', Set{Symbol}()) == " "
        @test KeyboardHandler.hotkey_to_string(:enter, Set{Symbol}()) == "enter"
        
        # Test with modifiers
        @test KeyboardHandler.hotkey_to_string('c', Set([:ctrl])) == "ctrl+c"
        @test KeyboardHandler.hotkey_to_string('v', Set([:ctrl, :shift])) in ["ctrl+shift+v", "shift+ctrl+v"]
        @test KeyboardHandler.hotkey_to_string('x', Set([:alt])) == "alt+x"
    end
    
    @testset "Command Palette Tests" begin
        handler = create_keyboard_handler()
        
        # Open command palette
        process_key(handler, ':')
        @test handler.state.command_palette.visible == true
        
        # Type command
        process_key(handler, 'h')
        process_key(handler, 'e')
        process_key(handler, 'l')
        process_key(handler, 'p')
        @test handler.state.command_palette.command_buffer == "help"
        
        # Execute command
        process_key(handler, :enter)
        @test handler.state.command_palette.visible == false
        @test handler.state.help_visible == true  # Help command executed
        
        # Test command history
        hide_help_overlay(handler)
        process_key(handler, ':')
        process_key(handler, :up)  # Should recall "help"
        @test handler.state.command_palette.command_buffer == "help"
        
        # Cancel command
        process_key(handler, :escape)
        @test handler.state.command_palette.visible == false
    end
    
    @testset "Custom Hotkey Registration" begin
        handler = create_keyboard_handler()
        
        # Register custom hotkey
        custom_called = false
        register_hotkey!(handler, 'x', () -> (custom_called = true),
                        description = "Custom Action")
        
        @test haskey(handler.state.hotkeys, "x")
        process_key(handler, 'x')
        @test custom_called == true
        
        # Register with modifiers
        ctrl_called = false
        register_hotkey!(handler, 's', () -> (ctrl_called = true),
                        modifiers = Set([:ctrl]),
                        description = "Save")
        
        process_key(handler, 's')  # Without ctrl
        @test ctrl_called == false
        
        process_key(handler, 's', Set([:ctrl]))  # With ctrl
        @test ctrl_called == true
    end
    
    @testset "Focus Management Tests" begin
        handler = create_keyboard_handler()
        
        # Add focusable panels
        add_focusable_panel!(handler, :panel1)
        add_focusable_panel!(handler, :panel2)
        add_focusable_panel!(handler, :panel3)
        
        focus_info = get_focus_info(handler)
        @test length(focus_info.panels) == 3
        @test focus_info.current == :panel1
        
        # Navigate with Tab
        process_key(handler, '\t')
        focus_info = get_focus_info(handler)
        @test focus_info.current == :panel2
        
        # Navigate with Shift+Tab
        process_key(handler, '\t', Set([:shift]))
        focus_info = get_focus_info(handler)
        @test focus_info.current == :panel1
        
        # Remove panel
        remove_focusable_panel!(handler, :panel2)
        focus_info = get_focus_info(handler)
        @test length(focus_info.panels) == 2
        @test :panel2 ∉ focus_info.panels
    end
    
    @testset "Help Text Generation" begin
        handler = create_keyboard_handler()
        
        help_text = KeyboardHandler.get_help_text(handler)
        @test occursin("Keyboard Shortcuts", help_text)
        @test occursin("Navigation:", help_text)
        @test occursin("Controls:", help_text)
        @test occursin("Panels:", help_text)
        @test occursin("Space", help_text)  # Should show space for pause
        @test occursin("q", help_text)      # Should show q for quit
    end
    
    @testset "Response Time Monitoring" begin
        handler = create_keyboard_handler(response_timeout_ms = 50)
        
        # Process key normally
        process_key(handler, 'a')
        
        # Simulate delay
        sleep(0.06)  # 60ms > 50ms threshold
        
        # This should trigger warning, but we just ensure it processes without error
        result = process_key(handler, 'b')
        @test result == false  # 'b' is not a registered hotkey
    end
    
    @testset "Multi-key Sequences" begin
        handler = create_keyboard_handler()
        
        # Test 'gg' sequence for go to top
        process_key(handler, 'g')
        @test handler.state.key_buffer == "g"
        
        go_to_top_called = false
        # Override the go_to_top action for testing
        handler.state.hotkeys["gg"] = KeyboardHandler.HotkeyBinding(
            'g', () -> (go_to_top_called = true),
            description = "Go to Top"
        )
        
        process_key(handler, 'g')
        @test go_to_top_called == true
        @test handler.state.key_buffer == ""  # Buffer cleared
        
        # Test incomplete sequence
        process_key(handler, 'g')
        process_key(handler, 'x')  # Not a valid sequence
        @test handler.state.key_buffer == ""  # Buffer cleared
    end
    
    @testset "Custom Commands" begin
        handler = create_keyboard_handler()
        
        # Register custom command
        test_value = 0
        register_command!(handler, "test", (args...) -> (test_value = length(args)))
        
        # Execute via command palette
        execute_command(handler, "test arg1 arg2")
        @test test_value == 2
        
        # Test built-in commands
        execute_command(handler, "help")
        @test handler.state.help_visible == true
        
        hide_help_overlay(handler)
        execute_command(handler, "panel 3")
        # Panel toggle would be logged
    end
    
    @testset "Macro Recording" begin
        handler = create_keyboard_handler()
        
        # Start recording
        KeyboardHandler.start_macro_recording(handler)
        @test handler.state.recording_macro == true
        
        # Record some keys
        process_key(handler, 'a')
        process_key(handler, 'b')
        process_key(handler, 'c')
        
        @test length(handler.state.macro_buffer) == 3
        @test handler.state.macro_buffer == ["a", "b", "c"]
        
        # Stop recording
        KeyboardHandler.stop_macro_recording(handler)
        @test handler.state.recording_macro == false
        
        # Play macro
        actions_played = String[]
        for key in ['a', 'b', 'c']
            register_hotkey!(handler, key, 
                           () -> push!(actions_played, string(key)))
        end
        
        KeyboardHandler.play_macro(handler)
        @test actions_played == ["a", "b", "c"]
    end
    
    @testset "Interrupt and Quit" begin
        handler = create_keyboard_handler()
        
        @test handler.interrupt_flag == false
        @test handler.active == true
        
        # Process quit command
        process_key(handler, 'q')
        
        @test handler.interrupt_flag == true
        @test handler.active == false
    end
end

println("All keyboard handler tests passed! ✓")