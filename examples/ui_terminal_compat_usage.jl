# Example usage of Terminal Compatibility System

using Printf

# Include the module
include("../src/ui/terminal_compat.jl")
using .TerminalCompat

"""
Terminal capabilities detection demo
"""
function capabilities_demo()
    println("Terminal Capabilities Detection Demo")
    println("===================================")
    
    # Detect capabilities
    caps = detect_terminal_capabilities()
    
    println("\nDetected Terminal Information:")
    println("  Terminal Type: $(caps.terminal_type)")
    println("  Dimensions: $(caps.width) × $(caps.height)")
    println("  Color Mode: $(caps.color_mode)")
    println("  Unicode Support: $(caps.unicode_support ? "Yes" : "No")")
    println("  Unicode Level: $(caps.unicode_level)")
    println("  Render Mode: $(caps.render_mode)")
    
    println("\nAdvanced Features:")
    println("  Cursor Save/Restore: $(caps.supports_cursor_save ? "✓" : "✗")")
    println("  Alternate Screen: $(caps.supports_alternate_screen ? "✓" : "✗")")
    println("  Mouse Support: $(caps.supports_mouse ? "✓" : "✗")")
end

"""
Color support testing demo
"""
function color_test_demo()
    println("\n\nColor Support Testing Demo")
    println("=========================")
    
    # Use built-in test function
    test_terminal_colors()
    
    # Custom gradient test
    println("\nCustom gradients:")
    
    # Red gradient (if true color supported)
    caps = detect_terminal_capabilities()
    if caps.color_mode == TerminalCompat.COLOR_TRUE
        print("Red gradient: ")
        for i in 0:20
            r = round(Int, 255 * i / 20)
            print("\033[38;2;$(r);0;0m█\033[0m")
        end
        println()
        
        print("Blue gradient: ")
        for i in 0:20
            b = round(Int, 255 * i / 20)
            print("\033[38;2;0;0;$(b)m█\033[0m")
        end
        println()
    else
        println("True color not supported, showing basic colors instead")
    end
end

"""
Unicode support testing demo
"""
function unicode_test_demo()
    println("\n\nUnicode Support Testing Demo")
    println("===========================")
    
    # Use built-in test
    test_unicode_support()
    
    # Additional tests
    caps = detect_terminal_capabilities()
    println("\nUnicode level: $(caps.unicode_level)")
    
    if caps.unicode_level >= 3
        println("Full Unicode support detected!")
    elseif caps.unicode_level >= 2
        println("Extended Unicode support (box drawing)")
    elseif caps.unicode_level >= 1
        println("Basic Unicode support")
    else
        println("No Unicode support - ASCII fallback mode")
    end
end

"""
Renderer fallback demo
"""
function renderer_fallback_demo()
    println("\n\nRenderer Fallback Demo")
    println("=====================")
    
    # Test content with Unicode and colors
    test_content = """
    ┌─────────────────┐
    │ \033[32m●\033[0m System Status │
    ├─────────────────┤
    │ CPU: ▓▓▓▓░░░░░░ │
    │ GPU: █████████░ │
    │ Temp: \033[31m▲ HIGH\033[0m   │
    └─────────────────┘
    """
    
    # Test with different renderer modes
    renderers = [
        (TerminalCompat.RENDER_BASIC, "Basic (ASCII only)"),
        (TerminalCompat.RENDER_STANDARD, "Standard"),
        (TerminalCompat.RENDER_ENHANCED, "Enhanced"),
        (TerminalCompat.RENDER_FULL, "Full")
    ]
    
    for (mode, name) in renderers
        println("\n$name Renderer:")
        println("─" ^ 30)
        
        caps = TerminalCapabilities(render_mode = mode)
        renderer = create_renderer(caps)
        rendered = render_with_fallback(renderer, test_content, caps)
        
        print(rendered)
    end
end

"""
ANSI sequence batching demo
"""
function batching_demo()
    println("\n\nANSI Sequence Batching Demo")
    println("==========================")
    
    # Create sequences
    sequences = [
        TerminalCompat.CURSOR_HOME,
        TerminalCompat.CLEAR_LINE,
        "\033[32m",     # Green
        "Hello, ",
        "\033[33m",     # Yellow
        "World!",
        "\033[0m"       # Reset
    ]
    
    println("Individual sequences: $(length(sequences))")
    
    # Batch them
    batched = batch_ansi_sequences(sequences)
    println("Batched length: $(length(batched)) characters")
    
    # Show result
    println("\nResult: ")
    print(batched)
    println()
    
    # Test efficiency with duplicates
    dup_sequences = ["\033[31m", "\033[31m", "Red", "\033[31m", "Text", "\033[0m"]
    optimized = batch_ansi_sequences(dup_sequences)
    println("\nOptimized duplicates: $(length(dup_sequences)) → $(length(split(optimized, "")))")
end

"""
Cursor movement optimization demo
"""
function cursor_optimization_demo()
    println("\n\nCursor Movement Optimization Demo")
    println("================================")
    
    # Test various movements
    movements = [
        (1, 1, 1, 10, "Vertical down"),
        (1, 10, 1, 1, "Vertical up"),
        (1, 1, 10, 1, "Horizontal right"),
        (10, 1, 1, 1, "Horizontal left"),
        (1, 1, 10, 10, "Diagonal")
    ]
    
    for (x1, y1, x2, y2, desc) in movements
        seq = optimize_cursor_movement(x1, y1, x2, y2)
        println("\n$desc: ($x1,$y1) → ($x2,$y2)")
        println("  Sequence: $(repr(seq))")
        println("  Length: $(length(seq)) characters")
    end
end

"""
Frame rate limiting demo
"""
function frame_rate_demo()
    println("\n\nFrame Rate Limiting Demo")
    println("=======================")
    
    # Create adaptive limiter
    limiter = FrameRateLimiter(
        target_fps = 10.0,
        adaptive = true,
        min_fps = 5.0,
        max_fps = 30.0
    )
    
    println("Target FPS: $(limiter.target_fps)")
    println("Adaptive: $(limiter.adaptive)")
    println("\nRendering 20 frames...")
    
    frame_count = 0
    start_time = now()
    
    # Render frames
    for i in 1:100
        if should_render_frame(limiter)
            frame_count += 1
            update_frame_time!(limiter)
            
            # Simulate rendering
            print("\rFrames: $frame_count")
            flush(stdout)
            
            # Break after 20 frames
            if frame_count >= 20
                break
            end
        end
        
        # Small delay
        sleep(0.01)
    end
    
    elapsed = Dates.value(now() - start_time) / 1000.0
    actual_fps = frame_count / elapsed
    
    println("\n\nResults:")
    println("  Frames rendered: $frame_count")
    println("  Time elapsed: $(round(elapsed, digits=2))s")
    println("  Actual FPS: $(round(actual_fps, digits=1))")
    println("  Current target FPS: $(round(limiter.target_fps, digits=1))")
end

"""
Partial redraw optimization demo
"""
function partial_redraw_demo()
    println("\n\nPartial Redraw Optimization Demo")
    println("===============================")
    
    # Create redraw region tracker
    region = PartialRedrawRegion()
    
    # Simulate panel updates
    panels = [
        (0, 0, 40, 10, "Header Panel"),
        (0, 10, 40, 20, "Content Panel"),
        (0, 20, 40, 30, "Footer Panel")
    ]
    
    println("Panel regions:")
    for (x1, y1, x2, y2, name) in panels
        println("  $name: ($x1,$y1) - ($x2,$y2)")
    end
    
    # Mark some panels dirty
    println("\nMarking panels dirty:")
    mark_dirty!(region, 0, 0, 40, 10)  # Header
    mark_dirty!(region, 0, 20, 40, 30)  # Footer
    
    # Check which need redraw
    println("\nRedraw needed:")
    for (x1, y1, x2, y2, name) in panels
        if is_dirty(region, x1, y1, x2, y2)
            println("  ✓ $name")
        else
            println("  - $name (cached)")
        end
    end
    
    # Cache content
    println("\nCaching content...")
    TerminalCompat.cache_content!(region, 0, 10, 40, 20, "Cached content")
    cached = TerminalCompat.get_cached_content(region, 0, 10, 40, 20)
    println("  Content Panel: '$cached'")
end

"""
Terminal-specific rendering demo
"""
function terminal_specific_demo()
    println("\n\nTerminal-Specific Rendering Demo")
    println("===============================")
    
    caps = detect_terminal_capabilities()
    println("Current terminal: $(caps.terminal_type)")
    
    # Create appropriate renderer
    renderer = create_renderer(caps)
    renderer_type = typeof(renderer)
    
    println("Selected renderer: $renderer_type")
    
    # Test with specific terminals
    test_terminals = [
        ("iTerm.app", "iTerm2"),
        ("Windows Terminal", "Windows Terminal"),
        ("gnome-terminal", "GNOME Terminal"),
        ("xterm", "XTerm")
    ]
    
    println("\nRenderer selection for different terminals:")
    for (term_type, name) in test_terminals
        test_caps = TerminalCapabilities(
            terminal_type = term_type,
            render_mode = TerminalCompat.RENDER_FULL
        )
        test_renderer = create_renderer(test_caps)
        println("  $name → $(typeof(test_renderer))")
    end
end

# Main menu
function main()
    println("Terminal Compatibility System Examples")
    println("====================================")
    println("1. Terminal Capabilities Detection")
    println("2. Color Support Testing")
    println("3. Unicode Support Testing")
    println("4. Renderer Fallback Demo")
    println("5. ANSI Sequence Batching")
    println("6. Cursor Movement Optimization")
    println("7. Frame Rate Limiting")
    println("8. Partial Redraw Optimization")
    println("9. Terminal-Specific Rendering")
    println("10. Run All Demos")
    println("\nSelect demo (1-10): ")
    
    choice = readline()
    
    if choice == "1"
        capabilities_demo()
    elseif choice == "2"
        color_test_demo()
    elseif choice == "3"
        unicode_test_demo()
    elseif choice == "4"
        renderer_fallback_demo()
    elseif choice == "5"
        batching_demo()
    elseif choice == "6"
        cursor_optimization_demo()
    elseif choice == "7"
        frame_rate_demo()
    elseif choice == "8"
        partial_redraw_demo()
    elseif choice == "9"
        terminal_specific_demo()
    elseif choice == "10"
        capabilities_demo()
        color_test_demo()
        unicode_test_demo()
        renderer_fallback_demo()
        batching_demo()
        cursor_optimization_demo()
        println("\nSkipping frame rate demo in batch mode...")
        partial_redraw_demo()
        terminal_specific_demo()
    else
        println("Invalid choice. Running capabilities demo...")
        capabilities_demo()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end