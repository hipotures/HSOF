# Example usage of Color Theme System

using Printf

# Include the module
include("../src/ui/color_theme.jl")
using .ColorTheme

"""
Basic theme usage demo
"""
function basic_theme_demo()
    println("Basic Color Theme Demo")
    println("=====================")
    
    # Create default theme (dark)
    theme = create_theme()
    
    # Display some colored text
    println("\nStatus Colors:")
    normal_text = apply_theme_color("● Normal Status", get_status_color(theme, :normal))
    warning_text = apply_theme_color("▲ Warning Status", get_status_color(theme, :warning))
    critical_text = apply_theme_color("■ Critical Status", get_status_color(theme, :critical))
    
    println("  $normal_text")
    println("  $warning_text")
    println("  $critical_text")
    
    # Display with background colors
    println("\nBackground Colors:")
    bg_normal = apply_theme_color(" NORMAL ", get_status_color(theme, :normal), background=true)
    bg_warning = apply_theme_color(" WARNING ", get_status_color(theme, :warning), background=true)
    bg_critical = apply_theme_color(" CRITICAL ", get_status_color(theme, :critical), background=true)
    
    println("  $bg_normal $bg_warning $bg_critical")
end

"""
Metric-based coloring demo
"""
function metric_coloring_demo()
    println("\n\nMetric-based Coloring Demo")
    println("=========================")
    
    theme = create_theme()
    
    # GPU Temperature examples
    println("\nGPU Temperature:")
    temps = [60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
    for temp in temps
        colored = format_with_status(theme, :gpu_temp, temp)
        indicator = status_indicator(theme, :gpu_temp, temp)
        println("  $indicator $colored")
    end
    
    # GPU Utilization examples
    println("\nGPU Utilization:")
    utils = [50.0, 70.0, 80.0, 85.0, 90.0, 95.0, 99.0]
    for util in utils
        colored = format_with_status(theme, :gpu_utilization, util)
        indicator = status_indicator(theme, :gpu_utilization, util)
        println("  $indicator $colored")
    end
    
    # Performance metrics (inverted - lower is worse)
    println("\nNodes per Second (inverted threshold):")
    nodes = [10000.0, 7500.0, 5000.0, 3000.0, 2000.0, 1000.0, 500.0]
    for n in nodes
        colored = format_with_status(theme, :nodes_per_sec, n, "%.0f")
        indicator = status_indicator(theme, :nodes_per_sec, n)
        println("  $indicator $colored")
    end
end

"""
Theme switching demo
"""
function theme_switching_demo()
    println("\n\nTheme Switching Demo")
    println("===================")
    
    theme = create_theme()
    
    # Test value
    test_temp = 85.0
    
    # Show same metric in different themes
    themes = [:dark, :light, :high_contrast, :colorblind]
    
    for theme_name in themes
        set_theme!(theme, theme_name)
        println("\n$(uppercase(string(theme_name))) Theme:")
        
        # Show status colors
        normal = apply_theme_color("Normal", get_status_color(theme, :normal))
        warning = apply_theme_color("Warning", get_status_color(theme, :warning))
        critical = apply_theme_color("Critical", get_status_color(theme, :critical))
        
        println("  Status: $normal | $warning | $critical")
        
        # Show metric coloring
        temp_colored = format_with_status(theme, :gpu_temp, test_temp)
        println("  GPU Temp: $temp_colored")
    end
end

"""
Smooth transitions demo
"""
function smooth_transitions_demo()
    println("\n\nSmooth Transitions Demo")
    println("======================")
    
    # Create theme with smooth transitions
    theme = create_theme(smooth_transitions = true)
    
    println("\nGPU Temperature Gradient (70°C - 95°C):")
    for temp in 70:1:95
        colored = format_with_status(theme, :gpu_temp, Float64(temp), "%.0f")
        bar = apply_theme_color("█", get_color(theme, :gpu_temp, Float64(temp)))
        if temp % 5 == 0
            println("$bar $colored")
        else
            print(bar)
        end
    end
    println()
    
    # Compare with non-smooth transitions
    theme.smooth_transitions = false
    println("\nSame range without smooth transitions:")
    for temp in 70:5:95
        colored = format_with_status(theme, :gpu_temp, Float64(temp), "%.0f")
        bar = apply_theme_color("████", get_color(theme, :gpu_temp, Float64(temp)))
        println("$bar $colored")
    end
end

"""
Colorblind mode demo
"""
function colorblind_mode_demo()
    println("\n\nColorblind Mode Demo")
    println("===================")
    
    theme = create_theme()
    
    # Normal mode
    println("\nNormal Mode:")
    display_metric_row(theme, :gpu_temp, [70.0, 85.0, 95.0])
    
    # Colorblind mode with patterns
    set_colorblind_mode!(theme, true, true)
    println("\nColorblind Mode with Patterns:")
    display_metric_row(theme, :gpu_temp, [70.0, 85.0, 95.0])
    
    # Show pattern legend
    println("\nPattern Legend:")
    println("  $(get_pattern_symbol(:normal)) = Normal")
    println("  $(get_pattern_symbol(:warning)) = Warning")
    println("  $(get_pattern_symbol(:critical)) = Critical")
end

"""
Alert flashing demo
"""
function alert_flashing_demo()
    println("\n\nAlert Flashing Demo")
    println("==================")
    println("Critical alerts can flash (simulated):")
    
    theme = create_theme(flash_critical = true)
    
    # Simulate flashing over time
    critical_text = "CRITICAL: GPU Temperature 95°C!"
    
    println("\nFlashing alert simulation:")
    for i in 1:10
        # Simulate time progression
        flashed = flash_alert(theme, critical_text, 1.0, 2.0)
        print("\r  $flashed")
        flush(stdout)
        sleep(0.1)
    end
    println()
end

"""
Custom threshold demo
"""
function custom_threshold_demo()
    println("\n\nCustom Threshold Demo")
    println("====================")
    
    theme = create_theme()
    
    # Add custom metric threshold
    custom_threshold = MetricThreshold(
        :custom_score, 
        0.7,  # warning below 0.7
        0.5,  # critical below 0.5
        invert = true,  # lower is worse
        unit = ""
    )
    add_threshold!(theme, custom_threshold)
    
    println("\nCustom Score Metric:")
    scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
    for score in scores
        colored = format_with_status(theme, :custom_score, score, "%.2f")
        indicator = status_indicator(theme, :custom_score, score)
        println("  $indicator Score: $colored")
    end
end

"""
Gradient creation demo
"""
function gradient_creation_demo()
    println("\n\nGradient Creation Demo")
    println("=====================")
    
    theme = create_theme()
    
    # Create gradient from normal to critical
    println("\nGradient from Normal to Critical:")
    gradient = create_gradient(theme, :normal, :critical, 20)
    
    for (i, color) in enumerate(gradient)
        block = apply_theme_color("█", color)
        print(block)
    end
    println()
    
    # Create gradient from normal to warning
    println("\nGradient from Normal to Warning:")
    gradient = create_gradient(theme, :normal, :warning, 20)
    
    for (i, color) in enumerate(gradient)
        block = apply_theme_color("█", color)
        print(block)
    end
    println()
end

# Helper function
function display_metric_row(theme::ThemeConfig, metric::Symbol, values::Vector{Float64})
    for val in values
        colored = format_with_status(theme, metric, val)
        indicator = status_indicator(theme, metric, val)
        print("  $indicator $colored")
    end
    println()
end

# Main menu
function main()
    println("Color Theme System Examples")
    println("==========================")
    println("1. Basic Theme Usage")
    println("2. Metric-based Coloring")
    println("3. Theme Switching")
    println("4. Smooth Transitions")
    println("5. Colorblind Mode")
    println("6. Alert Flashing")
    println("7. Custom Thresholds")
    println("8. Gradient Creation")
    println("9. Run All Demos")
    println("\nSelect demo (1-9): ")
    
    choice = readline()
    
    if choice == "1"
        basic_theme_demo()
    elseif choice == "2"
        metric_coloring_demo()
    elseif choice == "3"
        theme_switching_demo()
    elseif choice == "4"
        smooth_transitions_demo()
    elseif choice == "5"
        colorblind_mode_demo()
    elseif choice == "6"
        alert_flashing_demo()
    elseif choice == "7"
        custom_threshold_demo()
    elseif choice == "8"
        gradient_creation_demo()
    elseif choice == "9"
        basic_theme_demo()
        metric_coloring_demo()
        theme_switching_demo()
        smooth_transitions_demo()
        colorblind_mode_demo()
        println("\nSkipping alert flashing in batch mode...")
        custom_threshold_demo()
        gradient_creation_demo()
    else
        println("Invalid choice. Running basic demo...")
        basic_theme_demo()
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end