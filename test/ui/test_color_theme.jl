using Test
using Printf

# Include the module
include("../../src/ui/color_theme.jl")
using .ColorTheme

@testset "Color Theme Tests" begin
    
    @testset "Color Scheme Tests" begin
        # Test color scheme creation
        scheme = ColorTheme.ColorScheme(
            :test,
            (r=0, g=255, b=0),      # normal
            (r=255, g=255, b=0),    # warning
            (r=255, g=0, b=0),      # critical
            (r=0, g=0, b=0),        # background
            (r=255, g=255, b=255),  # text
            (r=0, g=0, b=255),      # accent
            (r=128, g=128, b=128)   # dim
        )
        
        @test scheme.name == :test
        @test scheme.normal == (r=0, g=255, b=0)
        @test scheme.warning == (r=255, g=255, b=0)
        @test scheme.critical == (r=255, g=0, b=0)
    end
    
    @testset "Metric Threshold Tests" begin
        # Test threshold creation
        threshold = MetricThreshold(:gpu_temp, 80.0, 90.0, unit="°C")
        @test threshold.metric_name == :gpu_temp
        @test threshold.warning_threshold == 80.0
        @test threshold.critical_threshold == 90.0
        @test threshold.invert == false
        @test threshold.unit == "°C"
        
        # Test inverted threshold
        inv_threshold = MetricThreshold(:accuracy, 0.8, 0.6, invert=true)
        @test inv_threshold.invert == true
        @test inv_threshold.warning_threshold > inv_threshold.critical_threshold
        
        # Test invalid thresholds
        @test_throws AssertionError MetricThreshold(:test, 80.0, 80.0)  # Same values
        @test_throws AssertionError MetricThreshold(:test, 90.0, 80.0)  # Wrong order
        @test_throws AssertionError MetricThreshold(:test, 0.6, 0.8, invert=true)  # Wrong order for inverted
    end
    
    @testset "Theme Configuration Tests" begin
        # Test default theme creation
        theme = create_theme()
        @test theme.current_scheme == :dark
        @test length(theme.schemes) >= 4  # At least 4 default schemes
        @test haskey(theme.schemes, :dark)
        @test haskey(theme.schemes, :light)
        @test haskey(theme.schemes, :high_contrast)
        @test haskey(theme.schemes, :colorblind)
        
        # Test custom theme settings
        custom_theme = create_theme(
            current_scheme = :light,
            colorblind_mode = true,
            flash_critical = false
        )
        @test custom_theme.current_scheme == :light
        @test custom_theme.colorblind_mode == true
        @test custom_theme.flash_critical == false
    end
    
    @testset "Status Color Tests" begin
        theme = create_theme()
        
        # Test getting status colors
        normal_color = get_status_color(theme, :normal)
        @test normal_color == (r=34, g=139, b=34)  # Forest green for dark theme
        
        warning_color = get_status_color(theme, :warning)
        @test warning_color == (r=255, g=215, b=0)  # Gold
        
        critical_color = get_status_color(theme, :critical)
        @test critical_color == (r=220, g=20, b=60)  # Crimson
        
        # Test with different theme
        set_theme!(theme, :light)
        light_normal = get_status_color(theme, :normal)
        @test light_normal == (r=0, g=128, b=0)  # Different green for light theme
    end
    
    @testset "Metric-based Color Selection" begin
        theme = create_theme()
        
        # Test GPU temperature coloring
        color_normal = get_color(theme, :gpu_temp, 70.0)
        color_warning = get_color(theme, :gpu_temp, 85.0)
        color_critical = get_color(theme, :gpu_temp, 95.0)
        
        # With smooth transitions off
        theme.smooth_transitions = false
        @test get_color(theme, :gpu_temp, 70.0) == get_status_color(theme, :normal)
        @test get_color(theme, :gpu_temp, 85.0) == get_status_color(theme, :warning)
        @test get_color(theme, :gpu_temp, 95.0) == get_status_color(theme, :critical)
        
        # Test inverted metric (nodes_per_sec)
        @test get_color(theme, :nodes_per_sec, 10000.0) == get_status_color(theme, :normal)
        @test get_color(theme, :nodes_per_sec, 3000.0) == get_status_color(theme, :warning)
        @test get_color(theme, :nodes_per_sec, 500.0) == get_status_color(theme, :critical)
    end
    
    @testset "Color Interpolation Tests" begin
        # Test basic interpolation
        color1 = (r=0, g=0, b=0)
        color2 = (r=255, g=255, b=255)
        
        mid_color = interpolate_color(color1, color2, 0.5)
        @test mid_color.r ≈ 128 atol=1
        @test mid_color.g ≈ 128 atol=1
        @test mid_color.b ≈ 128 atol=1
        
        # Test edge cases
        @test interpolate_color(color1, color2, 0.0) == color1
        @test interpolate_color(color1, color2, 1.0) == color2
        @test interpolate_color(color1, color2, -0.5) == color1  # Clamped
        @test interpolate_color(color1, color2, 1.5) == color2   # Clamped
    end
    
    @testset "Smooth Transitions Tests" begin
        theme = create_theme(smooth_transitions = true)
        
        # Test smooth transition for GPU temp
        # Values: 70 (normal), 80 (warning), 85 (between), 90 (critical), 95 (beyond)
        color_70 = get_color(theme, :gpu_temp, 70.0)
        color_80 = get_color(theme, :gpu_temp, 80.0)
        color_85 = get_color(theme, :gpu_temp, 85.0)
        color_90 = get_color(theme, :gpu_temp, 90.0)
        
        # Color at 85 should be between warning and critical
        # Since we're interpolating, check that it's not exactly warning or critical
        @test color_85 != color_80  # Different from warning threshold
        @test color_85 != color_90  # Different from critical threshold
        
        # Test inverted metric smooth transition
        color_high = get_color(theme, :nodes_per_sec, 10000.0)
        color_mid = get_color(theme, :nodes_per_sec, 3000.0)
        color_low = get_color(theme, :nodes_per_sec, 500.0)
        
        # Should transition from normal to warning to critical as value decreases
        # For inverted metrics with smooth transitions, just verify they're different
        @test color_high != color_mid
        @test color_mid != color_low
        @test color_high != color_low
    end
    
    @testset "ANSI Color Application Tests" begin
        theme = create_theme()
        
        # Test foreground color
        colored_text = apply_theme_color("Test", (r=255, g=0, b=0))
        @test occursin("\033[38;2;255;0;0m", colored_text)
        @test occursin("Test", colored_text)
        @test occursin("\033[0m", colored_text)
        
        # Test background color
        bg_text = apply_theme_color("Alert", (r=255, g=255, b=0), background=true)
        @test occursin("\033[48;2;255;255;0m", bg_text)
        @test occursin("Alert", bg_text)
        @test occursin("\033[0m", bg_text)
    end
    
    @testset "Formatted Output Tests" begin
        theme = create_theme()
        
        # Test basic formatting
        formatted = format_with_status(theme, :gpu_temp, 85.5)
        @test occursin("85.5°C", formatted)
        @test occursin("\033[", formatted)  # Has ANSI codes
        
        # Test custom format string
        formatted_custom = format_with_status(theme, :gpu_utilization, 92.345, "%.0f")
        @test occursin("92%", formatted_custom)
        
        # Test colorblind mode
        set_colorblind_mode!(theme, true, true)
        cb_formatted = format_with_status(theme, :gpu_temp, 95.0)
        @test occursin("■", cb_formatted)  # Critical pattern
        @test occursin("95.0°C", cb_formatted)
    end
    
    @testset "Gradient Creation Tests" begin
        theme = create_theme()
        
        # Create gradient from normal to critical
        gradient = create_gradient(theme, :normal, :critical, 5)
        @test length(gradient) == 5
        @test gradient[1] == get_status_color(theme, :normal)
        @test gradient[5] == get_status_color(theme, :critical)
        
        # Middle colors should be interpolated
        @test gradient[3].r > gradient[1].r
        @test gradient[3].r < gradient[5].r
    end
    
    @testset "Alert Flashing Tests" begin
        theme = create_theme(flash_critical = true)
        
        # Can't easily test time-based flashing, but test the function exists
        text = "CRITICAL ALERT"
        flashed = flash_alert(theme, text, 1.0, 2.0)
        @test occursin(text, flashed) || flashed == text
        
        # Test with flashing disabled
        theme.flash_critical = false
        no_flash = flash_alert(theme, text)
        @test no_flash == text
    end
    
    @testset "Colorblind Mode Tests" begin
        theme = create_theme()
        
        # Test pattern symbols
        @test get_pattern_symbol(:normal) == "●"
        @test get_pattern_symbol(:warning) == "▲"
        @test get_pattern_symbol(:critical) == "■"
        
        # Test background patterns
        @test get_colorblind_pattern(:normal) == "····"
        @test get_colorblind_pattern(:warning) == "////"
        @test get_colorblind_pattern(:critical) == "XXXX"
        
        # Test colorblind mode activation
        set_colorblind_mode!(theme, true)
        @test theme.colorblind_mode == true
        @test theme.current_scheme == :colorblind
    end
    
    @testset "Status Indicator Tests" begin
        theme = create_theme()
        
        # Test normal status
        indicator = status_indicator(theme, :gpu_temp, 70.0)
        @test occursin("●", indicator)
        @test occursin("\033[38;2;34;139;34m", indicator)  # Normal color
        
        # Test with colorblind mode
        set_colorblind_mode!(theme, true, true)
        cb_indicator = status_indicator(theme, :gpu_temp, 95.0)
        @test occursin("■", cb_indicator)  # Critical symbol
    end
    
    @testset "Custom Threshold Tests" begin
        theme = create_theme()
        
        # Add custom threshold
        custom = MetricThreshold(:custom_metric, 50.0, 75.0, unit="units")
        add_threshold!(theme, custom)
        
        @test haskey(theme.thresholds.thresholds, :custom_metric)
        @test theme.thresholds.thresholds[:custom_metric].warning_threshold == 50.0
        
        # Test coloring with custom threshold
        # Disable smooth transitions for exact color match
        theme.smooth_transitions = false
        color = get_color(theme, :custom_metric, 60.0)
        @test color == get_status_color(theme, :warning)
    end
    
    @testset "Theme Switching Tests" begin
        theme = create_theme()
        
        # Test switching themes
        set_theme!(theme, :light)
        @test theme.current_scheme == :light
        
        set_theme!(theme, :high_contrast)
        @test theme.current_scheme == :high_contrast
        
        # Test invalid theme
        @test_throws ErrorException set_theme!(theme, :nonexistent)
    end
end

println("All color theme tests passed! ✓")