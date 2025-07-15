using Test
using Dates

# Include the module
include("../../src/ui/terminal_compat.jl")
using .TerminalCompat

@testset "Terminal Compatibility Tests" begin
    
    @testset "Color Mode Detection" begin
        # Test various TERM values
        @test TerminalCompat.detect_color_mode("xterm-256color", "") == TerminalCompat.COLOR_256
        @test TerminalCompat.detect_color_mode("xterm", "") == TerminalCompat.COLOR_8BIT
        @test TerminalCompat.detect_color_mode("dumb", "") == TerminalCompat.COLOR_NONE
        @test TerminalCompat.detect_color_mode("", "") == TerminalCompat.COLOR_NONE
        
        # Test COLORTERM for true color
        @test TerminalCompat.detect_color_mode("xterm", "truecolor") == TerminalCompat.COLOR_TRUE
        @test TerminalCompat.detect_color_mode("xterm", "24bit") == TerminalCompat.COLOR_TRUE
    end
    
    @testset "Terminal Capabilities Detection" begin
        # Test capability detection
        caps = detect_terminal_capabilities()
        @test isa(caps, TerminalCapabilities)
        @test caps.width > 0
        @test caps.height > 0
        @test caps.color_mode isa TerminalCompat.ColorMode
        @test caps.render_mode isa TerminalCompat.RenderMode
    end
    
    @testset "Render Mode Determination" begin
        # Test render mode selection based on capabilities
        @test TerminalCompat.determine_render_mode(TerminalCompat.COLOR_TRUE, 3) == TerminalCompat.RENDER_FULL
        @test TerminalCompat.determine_render_mode(TerminalCompat.COLOR_256, 2) == TerminalCompat.RENDER_ENHANCED
        @test TerminalCompat.determine_render_mode(TerminalCompat.COLOR_8BIT, 1) == TerminalCompat.RENDER_STANDARD
        @test TerminalCompat.determine_render_mode(TerminalCompat.COLOR_NONE, 0) == TerminalCompat.RENDER_BASIC
    end
    
    @testset "Renderer Creation" begin
        # Create capabilities with specific settings
        caps_basic = TerminalCapabilities(render_mode = TerminalCompat.RENDER_BASIC)
        caps_full = TerminalCapabilities(render_mode = TerminalCompat.RENDER_FULL)
        caps_iterm = TerminalCapabilities(terminal_type = "iTerm.app", render_mode = TerminalCompat.RENDER_FULL)
        
        # Test renderer creation
        @test isa(create_renderer(caps_basic), TerminalCompat.BasicRenderer)
        @test isa(create_renderer(caps_full), TerminalCompat.FullRenderer)
        @test isa(create_renderer(caps_iterm), TerminalCompat.ITerm2Renderer)
    end
    
    @testset "ANSI Sequence Batching" begin
        # Test batch creation
        batch = TerminalCompat.ANSIBatch(5, auto_flush = false)
        @test isempty(batch.sequences)
        
        # Add sequences
        TerminalCompat.add_sequence!(batch, "\033[31m")
        TerminalCompat.add_sequence!(batch, "Hello")
        TerminalCompat.add_sequence!(batch, "\033[0m")
        
        @test length(batch.sequences) == 3
        
        # Flush batch
        result = TerminalCompat.flush_batch!(batch)
        @test result == "\033[31mHello\033[0m"
        @test isempty(batch.sequences)
        
        # Test auto-flush
        batch_auto = TerminalCompat.ANSIBatch(2, auto_flush = true)
        TerminalCompat.add_sequence!(batch_auto, "A")
        result = TerminalCompat.add_sequence!(batch_auto, "B")
        @test result == "AB"
        @test isempty(batch_auto.sequences)
    end
    
    @testset "Batch Optimization" begin
        # Test sequence optimization
        sequences = ["\033[31m", "\033[31m", "Text", "\033[0m"]
        optimized = batch_ansi_sequences(sequences)
        @test optimized == "\033[31mText\033[0m"
        
        # Test with all unique sequences
        sequences = ["\033[31m", "\033[32m", "\033[33m"]
        optimized = batch_ansi_sequences(sequences)
        @test optimized == "\033[31m\033[32m\033[33m"
    end
    
    @testset "Cursor Movement Optimization" begin
        # Test no movement
        @test optimize_cursor_movement(5, 5, 5, 5) == ""
        
        # Test simple movements
        @test optimize_cursor_movement(1, 1, 1, 5) == "\033[4B"  # Move down 4
        @test optimize_cursor_movement(1, 5, 1, 1) == "\033[4A"  # Move up 4
        @test optimize_cursor_movement(1, 1, 5, 1) == "\033[4C"  # Move right 4
        @test optimize_cursor_movement(5, 1, 1, 1) == "\033[4D"  # Move left 4
        
        # Test diagonal movement
        result = optimize_cursor_movement(1, 1, 5, 5)
        # Should use absolute positioning for diagonal moves
        @test result == "\033[5;5H" || (occursin("B", result) && occursin("C", result))
    end
    
    @testset "Frame Rate Limiter" begin
        # Create limiter
        limiter = FrameRateLimiter(target_fps = 10.0, adaptive = false)
        @test limiter.target_fps == 10.0
        @test limiter.frame_duration == 100.0
        
        # First frame should always render
        update_frame_time!(limiter)
        
        # Immediate check should be false
        @test should_render_frame(limiter) == false
        
        # After delay should be true
        sleep(0.11)  # Sleep longer than frame duration
        @test should_render_frame(limiter) == true
    end
    
    @testset "Adaptive Frame Rate" begin
        limiter = FrameRateLimiter(
            target_fps = 10.0,
            adaptive = true,
            min_fps = 5.0,
            max_fps = 20.0
        )
        
        # Simulate frame times
        for i in 1:10
            push!(limiter.frame_times, 150.0)  # Slow frames
        end
        
        # Should adapt down
        TerminalCompat.adapt_frame_rate!(limiter)
        @test limiter.target_fps < 10.0
        @test limiter.target_fps >= 5.0
    end
    
    @testset "Partial Redraw Regions" begin
        region = PartialRedrawRegion()
        
        # Initially no dirty regions
        @test !is_dirty(region, 0, 0, 10, 10)
        
        # Mark region dirty
        mark_dirty!(region, 0, 0, 10, 10)
        @test is_dirty(region, 0, 0, 10, 10)
        @test !is_dirty(region, 20, 20, 30, 30)
        
        # Clear dirty flags
        clear_dirty!(region)
        @test !is_dirty(region, 0, 0, 10, 10)
        
        # Test full redraw
        @test !TerminalCompat.needs_full_redraw(region)
        TerminalCompat.mark_full_redraw!(region)
        @test TerminalCompat.needs_full_redraw(region)
    end
    
    @testset "Content Caching" begin
        region = PartialRedrawRegion()
        
        # Cache content
        TerminalCompat.cache_content!(region, 0, 0, 10, 10, "Hello")
        @test TerminalCompat.get_cached_content(region, 0, 0, 10, 10) == "Hello"
        @test TerminalCompat.get_cached_content(region, 1, 1, 11, 11) == ""
    end
    
    @testset "Basic Renderer Fallbacks" begin
        renderer = TerminalCompat.BasicRenderer()
        caps = TerminalCapabilities()
        
        # Test Unicode to ASCII conversion
        unicode_content = "█▓▒░ ●▲■ ┌─┐│└┘"
        ascii_result = TerminalCompat.render_content(renderer, unicode_content, caps)
        @test ascii_result == "#=-. *^# +-+|++"
        
        # Test ANSI stripping
        ansi_content = "\033[31mRed Text\033[0m"
        stripped = TerminalCompat.render_content(renderer, ansi_content, caps)
        @test stripped == "Red Text"
    end
    
    @testset "Terminal Size Detection" begin
        width, height = TerminalCompat.get_terminal_size()
        @test width > 0
        @test height > 0
        @test width <= 1000  # Reasonable maximum
        @test height <= 1000  # Reasonable maximum
    end
    
    @testset "Unicode Level Detection" begin
        # Test Unicode rendering detection
        level = TerminalCompat.test_unicode_rendering()
        @test level >= 0
        @test level <= 3
        
        # If we have UTF-8 locale, should have some Unicode support
        if occursin("UTF", get(ENV, "LANG", ""))
            @test level > 0
        end
    end
    
    @testset "ANSI Constants" begin
        # Test that constants are defined correctly
        @test TerminalCompat.ESC == "\033"
        @test TerminalCompat.CSI == "\033["
        @test TerminalCompat.CURSOR_HOME == "\033[H"
        @test TerminalCompat.CLEAR_SCREEN == "\033[2J"
        @test TerminalCompat.RESET == "\033[0m"
    end
    
    @testset "Terminal Type Detection" begin
        term_type = get_terminal_type()
        @test isa(term_type, String)
        @test !isempty(term_type) || term_type == ""
    end
    
    @testset "Renderer Fallback Chain" begin
        # Test rendering with different capabilities
        caps_basic = TerminalCapabilities(render_mode = TerminalCompat.RENDER_BASIC)
        caps_full = TerminalCapabilities(render_mode = TerminalCompat.RENDER_FULL)
        
        renderer_basic = create_renderer(caps_basic)
        renderer_full = create_renderer(caps_full)
        
        test_content = "█ Test \033[31mColor\033[0m"
        
        basic_result = render_with_fallback(renderer_basic, test_content, caps_basic)
        full_result = render_with_fallback(renderer_full, test_content, caps_full)
        
        # Basic should strip colors and convert Unicode
        @test !occursin("█", basic_result)
        @test !occursin("\033", basic_result)
        
        # Full should preserve everything
        @test occursin("█", full_result)
        @test occursin("\033[31m", full_result)
    end
end

println("All terminal compatibility tests passed! ✓")