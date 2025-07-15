using Test
using Dates
using Statistics

# Include the module
include("../../src/ui/sparkline_graph.jl")
using .SparklineGraph

@testset "Sparkline Graph Tests" begin
    
    @testset "Circular Buffer Tests" begin
        # Test buffer creation
        buffer = CircularBuffer{Float64}(10)
        @test buffer.capacity == 10
        @test buffer.size == 0
        @test buffer.head == 1
        
        # Test pushing data
        push_data!(buffer, 1.0)
        @test buffer.size == 1
        @test buffer.data[1] == 1.0
        
        # Test multiple pushes
        for i in 2:5
            push_data!(buffer, Float64(i))
        end
        @test buffer.size == 5
        
        # Test get_data
        data, timestamps = get_data(buffer)
        @test length(data) == 5
        @test data == [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test buffer overflow
        for i in 6:15
            push_data!(buffer, Float64(i))
        end
        @test buffer.size == 10  # Max capacity
        
        data, _ = get_data(buffer)
        @test length(data) == 10
        @test data == collect(6.0:15.0)  # Should have oldest data removed
        
        # Test get_recent_data
        recent_data, _ = get_recent_data(buffer, 5)
        @test length(recent_data) == 5
        @test recent_data == [11.0, 12.0, 13.0, 14.0, 15.0]
        
        # Test clear
        clear_buffer!(buffer)
        @test buffer.size == 0
        @test buffer.head == 1
    end
    
    @testset "Sparkline Config Tests" begin
        # Test default config
        config = SparklineConfig()
        @test config.width == 20
        @test config.height == 8
        @test config.smooth == false
        @test config.gradient == false
        @test length(config.unicode_blocks) == 8
        @test config.unicode_blocks[1] == '▁'
        @test config.unicode_blocks[8] == '█'
        
        # Test custom config
        custom_config = SparklineConfig(
            width = 10,
            height = 4,
            smooth = true,
            gradient = true,
            unicode_blocks = ['_', '-', '=', '#']
        )
        @test custom_config.width == 10
        @test custom_config.height == 4
        @test custom_config.smooth == true
        @test custom_config.gradient == true
        @test length(custom_config.unicode_blocks) == 4
        
        # Test invalid configs
        @test_throws AssertionError SparklineConfig(width = 0)
        @test_throws AssertionError SparklineConfig(height = 0)
        @test_throws AssertionError SparklineConfig(smooth_window = 0)
        @test_throws AssertionError SparklineConfig(unicode_blocks = ['a', 'b'])  # Wrong length
    end
    
    @testset "Basic Sparkline Creation" begin
        # Test empty data
        @test create_sparkline(Float64[]) == ""
        
        # Test single value
        sparkline = create_sparkline([5.0])
        @test length(sparkline) == 1
        @test sparkline in string.(SparklineConfig().unicode_blocks)
        
        # Test flat line
        flat_data = fill(5.0, 10)
        sparkline = create_sparkline(flat_data)
        @test length(sparkline) == 10
        @test all(c == sparkline[1] for c in sparkline)
        
        # Test ascending data
        ascending = collect(1.0:10.0)
        sparkline = create_sparkline(ascending)
        @test length(sparkline) == 10
        @test sparkline[1] == '▁'  # Lowest
        @test sparkline[end] == '█'  # Highest
        
        # Test descending data
        descending = collect(10.0:-1.0:1.0)
        sparkline = create_sparkline(descending)
        @test sparkline[1] == '█'  # Highest
        @test sparkline[end] == '▁'  # Lowest
        
        # Test with custom width
        config = SparklineConfig(width = 5)
        sparkline = create_sparkline(collect(1.0:20.0), config)
        @test length(sparkline) == 5
    end
    
    @testset "Data Smoothing" begin
        # Test moving average smoothing
        noisy_data = [1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0]
        smoothed = smooth_data(noisy_data, 3)
        
        # Check that smoothing reduces variance
        @test std(smoothed) < std(noisy_data)
        
        # Test exponential smoothing
        exp_smoothed = SparklineGraph.exponential_smoothing(noisy_data, 0.3)
        @test length(exp_smoothed) == length(noisy_data)
        @test exp_smoothed[1] == noisy_data[1]  # First value unchanged
        
        # Test smoothing in sparkline
        config_smooth = SparklineConfig(smooth = true, smooth_window = 3)
        sparkline_smooth = create_sparkline(noisy_data, config_smooth)
        
        config_no_smooth = SparklineConfig(smooth = false)
        sparkline_no_smooth = create_sparkline(noisy_data, config_no_smooth)
        
        # Smoothed should be different (usually)
        @test sparkline_smooth != sparkline_no_smooth || std(noisy_data) < 0.1
    end
    
    @testset "Data Resampling" begin
        # Test downsampling
        data = collect(1.0:100.0)
        resampled = SparklineGraph.resample_data(data, 10)
        @test length(resampled) == 10
        @test resampled[1] ≈ 1.0
        @test resampled[end] ≈ 100.0
        
        # Test upsampling
        data = collect(1.0:5.0)
        resampled = SparklineGraph.resample_data(data, 10)
        @test length(resampled) == 10
        @test resampled[1] ≈ 1.0
        @test resampled[end] ≈ 5.0
        # Check monotonicity
        @test all(resampled[i] <= resampled[i+1] for i in 1:length(resampled)-1)
        
        # Test no resampling needed
        data = collect(1.0:10.0)
        resampled = SparklineGraph.resample_data(data, 10)
        @test resampled == data
    end
    
    @testset "Colored Sparklines" begin
        data = collect(1.0:10.0)
        
        # Test gradient disabled
        config_no_gradient = SparklineConfig(gradient = false)
        sparkline = create_colored_sparkline(data, config_no_gradient)
        @test !occursin("\033[", sparkline)  # No ANSI codes
        
        # Test gradient enabled
        config_gradient = SparklineConfig(gradient = true, min_color = "green", max_color = "red")
        sparkline = create_colored_sparkline(data, config_gradient)
        @test occursin("\033[", sparkline)  # Contains ANSI codes
        
        # Test apply_color function
        colored_text = SparklineGraph.apply_color("test", "red")
        @test occursin("\033[31m", colored_text)
        @test occursin("test", colored_text)
        @test occursin("\033[0m", colored_text)
        
        # Test gradient colors
        gradient_sparkline = apply_gradient_colors("████", [1.0, 2.0, 3.0, 4.0], "green", "red")
        @test occursin("\033[", gradient_sparkline)
    end
    
    @testset "Bar Sparklines" begin
        data = collect(0.0:10.0)
        bar_sparkline = create_bar_sparkline(data, SparklineConfig())
        
        @test length(bar_sparkline) == 11
        @test bar_sparkline[1] == ' '  # Space for zero
        @test bar_sparkline[end] == '█'  # Full bar for max
    end
    
    @testset "Specialized Sparklines" begin
        # Test score sparkline
        scores = [0.7, 0.72, 0.71, 0.74, 0.76, 0.78, 0.77, 0.80]
        score_sparkline = SparklineGraph.create_score_sparkline(scores)
        @test occursin("↑", score_sparkline)  # Upward trend
        
        scores_down = reverse(scores)
        score_sparkline_down = SparklineGraph.create_score_sparkline(scores_down)
        @test occursin("↓", score_sparkline_down)  # Downward trend
        
        # Test GPU sparkline
        gpu_values = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0]
        gpu_sparkline = SparklineGraph.create_gpu_sparkline(gpu_values, 80.0)
        @test occursin("\033[91m", gpu_sparkline)  # Bright red for exceeding threshold
        
        gpu_values_low = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
        gpu_sparkline_low = SparklineGraph.create_gpu_sparkline(gpu_values_low, 80.0)
        @test !occursin("\033[91m", gpu_sparkline_low)  # No red
        
        # Test performance sparkline
        perf_values = [100.0, 110.0, 105.0, 115.0, 120.0]
        perf_sparkline = SparklineGraph.create_performance_sparkline(perf_values, 100.0)
        @test occursin("\033[92m", perf_sparkline)  # Bright green for above target
        
        perf_values_low = [70.0, 75.0, 80.0, 75.0, 70.0]
        perf_sparkline_low = SparklineGraph.create_performance_sparkline(perf_values_low, 100.0)
        @test occursin("\033[31m", perf_sparkline_low)  # Red for below target
    end
    
    @testset "Sparkline Renderer" begin
        config = SparklineConfig(width = 10)
        renderer = SparklineRenderer(config, 20)
        
        @test renderer.auto_scale == true
        @test renderer.min_value == Inf
        @test renderer.max_value == -Inf
        
        # Push some data
        for i in 1:15
            push_data!(renderer, Float64(i))
        end
        
        @test renderer.min_value == 1.0
        @test renderer.max_value == 15.0
        
        # Test rendering
        sparkline = SparklineGraph.render(renderer)
        @test !isempty(sparkline)
        @test length(sparkline) <= 10  # Respects width config
        
        # Test stats
        stats = SparklineGraph.get_stats(renderer)
        @test stats.min == 1.0
        @test stats.max == 15.0
        @test stats.mean ≈ 8.0
        @test stats.trend == :increasing
        
        # Test with gradient
        config_gradient = SparklineConfig(width = 10, gradient = true)
        renderer_gradient = SparklineRenderer(config_gradient, 20)
        for i in 1:15
            push_data!(renderer_gradient, Float64(i))
        end
        
        sparkline_gradient = SparklineGraph.render(renderer_gradient)
        @test occursin("\033[", sparkline_gradient)  # Has color codes
    end
    
    @testset "Edge Cases" begin
        # Test very small values
        small_data = [1e-10, 2e-10, 3e-10, 4e-10]
        sparkline = create_sparkline(small_data)
        @test !isempty(sparkline)
        @test sparkline[1] == '▁'
        @test sparkline[end] == '█'
        
        # Test negative values
        negative_data = [-10.0, -5.0, 0.0, 5.0, 10.0]
        sparkline = create_sparkline(negative_data)
        @test !isempty(sparkline)
        @test length(sparkline) == 5
        
        # Test single repeated value
        same_value = fill(42.0, 20)
        sparkline = create_sparkline(same_value)
        @test all(c == sparkline[1] for c in sparkline)
        
        # Test boundary values
        config_boundaries = SparklineConfig(show_boundaries = true)
        sparkline_bounds = create_sparkline([1.0, 5.0, 10.0], config_boundaries)
        @test occursin("1.0", sparkline_bounds)
        @test occursin("10.0", sparkline_bounds)
    end
end

println("All sparkline tests passed! ✓")