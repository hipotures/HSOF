using Test
using Term
using Dates

# Include the module
include("../../src/ui/console_dashboard.jl")
using .ConsoleDashboard

@testset "Console Dashboard Tests" begin
    
    @testset "Configuration Tests" begin
        # Test default config
        config = DashboardConfig()
        @test config.refresh_rate_ms == 100
        @test config.color_scheme == :default
        @test config.border_style == :rounded
        @test config.show_timestamps == true
        @test config.responsive == true
        @test config.min_width == 80
        @test config.min_height == 24
        
        # Test custom config
        custom_config = DashboardConfig(
            refresh_rate_ms = 200,
            color_scheme = :dark,
            border_style = :double,
            show_timestamps = false,
            responsive = false,
            min_width = 100,
            min_height = 30
        )
        @test custom_config.refresh_rate_ms == 200
        @test custom_config.color_scheme == :dark
        @test custom_config.border_style == :double
        @test custom_config.show_timestamps == false
        @test custom_config.responsive == false
    end
    
    @testset "Panel Content Types Tests" begin
        # Test GPUPanelContent
        gpu_content = GPUPanelContent(1, 75.0, 8.5, 12.0, 65.0, 150.0, 1800.0, 50.0)
        @test gpu_content.gpu_id == 1
        @test gpu_content.utilization == 75.0
        @test gpu_content.memory_used == 8.5
        @test gpu_content.memory_total == 12.0
        @test gpu_content.temperature == 65.0
        @test gpu_content.power_draw == 150.0
        @test gpu_content.clock_speed == 1800.0
        @test gpu_content.fan_speed == 50.0
        
        # Test ProgressPanelContent
        progress_content = ProgressPanelContent(
            "Stage 2", 45.0, 60.0, 1000, 2000, 0.85, 0.92, 3600
        )
        @test progress_content.stage == "Stage 2"
        @test progress_content.overall_progress == 45.0
        @test progress_content.stage_progress == 60.0
        @test progress_content.items_processed == 1000
        @test progress_content.items_total == 2000
        @test progress_content.current_score == 0.85
        @test progress_content.best_score == 0.92
        @test progress_content.eta_seconds == 3600
        
        # Test MetricsPanelContent
        metrics_content = MetricsPanelContent(
            1500.0, 4.5, 65.0, 16.0, 85.0, 10, 24
        )
        @test metrics_content.nodes_per_second == 1500.0
        @test metrics_content.bandwidth_gbps == 4.5
        @test metrics_content.cpu_usage == 65.0
        @test metrics_content.ram_usage == 16.0
        @test metrics_content.cache_hit_rate == 85.0
        @test metrics_content.queue_depth == 10
        @test metrics_content.active_threads == 24
        
        # Test AnalysisPanelContent
        analysis_content = AnalysisPanelContent(
            1000, 50, 95.0,
            [("feature_1", 0.95), ("feature_2", 0.88), ("feature_3", 0.75)],
            "High correlation detected"
        )
        @test analysis_content.total_features == 1000
        @test analysis_content.selected_features == 50
        @test analysis_content.reduction_percentage == 95.0
        @test length(analysis_content.top_features) == 3
        @test analysis_content.top_features[1][1] == "feature_1"
        @test analysis_content.top_features[1][2] == 0.95
        
        # Test LogPanelContent
        log_entries = [
            (now(), :info, "System started"),
            (now(), :warn, "High memory usage"),
            (now(), :error, "Connection failed")
        ]
        log_content = LogPanelContent(log_entries, 100)
        @test length(log_content.entries) == 3
        @test log_content.entries[1][2] == :info
        @test log_content.entries[2][2] == :warn
        @test log_content.entries[3][2] == :error
        @test log_content.max_entries == 100
    end
    
    @testset "Helper Functions Tests" begin
        # Test get_border_box
        @test ConsoleDashboard.get_border_box(:single) == Term.BOXES[:SQUARE]
        @test ConsoleDashboard.get_border_box(:double) == Term.BOXES[:DOUBLE]
        @test ConsoleDashboard.get_border_box(:rounded) == Term.BOXES[:ROUNDED]
        @test ConsoleDashboard.get_border_box(:heavy) == Term.BOXES[:HEAVY]
        @test ConsoleDashboard.get_border_box(:invalid) == Term.BOXES[:ROUNDED]  # Default
        
        # Test get_temperature_color
        @test ConsoleDashboard.get_temperature_color(95.0) == "bright_red"
        @test ConsoleDashboard.get_temperature_color(85.0) == "bright_yellow"
        @test ConsoleDashboard.get_temperature_color(75.0) == "yellow"
        @test ConsoleDashboard.get_temperature_color(65.0) == "bright_green"
        
        # Test format_duration
        @test ConsoleDashboard.format_duration(45) == "45s"
        @test ConsoleDashboard.format_duration(125) == "2m 5s"
        @test ConsoleDashboard.format_duration(3725) == "1h 2m"
        
        # Test apply_color
        colored_text = ConsoleDashboard.apply_color("Test", "bright_red")
        @test occursin("bright_red", colored_text)
        @test occursin("Test", colored_text)
        
        # Test apply_style
        styled_text = ConsoleDashboard.apply_style("Bold", "bold")
        @test occursin("bold", styled_text)
        @test occursin("Bold", styled_text)
    end
    
    @testset "Progress Bar Tests" begin
        # Test different progress bar types
        bar1 = ConsoleDashboard.create_progress_bar(75.0, 100.0, 20, :utilization)
        @test occursin("[", bar1)
        @test occursin("]", bar1)
        @test occursin("█", bar1)
        @test occursin("░", bar1)
        
        bar2 = ConsoleDashboard.create_progress_bar(50.0, 100.0, 10, :memory)
        @test occursin("▓", bar2)
        
        bar3 = ConsoleDashboard.create_progress_bar(80.0, 100.0, 15, :overall)
        @test occursin("━", bar3)
        @test occursin("─", bar3)
        
        bar4 = ConsoleDashboard.create_progress_bar(90.0, 100.0, 12, :cpu)
        @test occursin("▪", bar4)
        @test occursin("▫", bar4)
        
        # Test edge cases
        bar_empty = ConsoleDashboard.create_progress_bar(0.0, 100.0, 10, :cpu)
        @test !occursin("▪", bar_empty)  # No filled chars
        
        bar_full = ConsoleDashboard.create_progress_bar(100.0, 100.0, 10, :cpu)
        @test !occursin("▫", bar_full)  # No empty chars
    end
    
    @testset "Content Rendering Tests" begin
        config = DashboardConfig()
        
        # Test GPU content rendering
        gpu_content = GPUPanelContent(1, 75.0, 8.5, 12.0, 65.0, 150.0, 1800.0, 50.0)
        gpu_render = ConsoleDashboard.render_gpu_content(gpu_content, config)
        @test occursin("Util:", gpu_render)
        @test occursin("75.0%", gpu_render)
        @test occursin("Mem:", gpu_render)
        @test occursin("8.5/12.0 GB", gpu_render)
        @test occursin("Temp:", gpu_render)
        @test occursin("65.0°C", gpu_render)
        @test occursin("Power: 150.0 W", gpu_render)
        @test occursin("Clock: 1800 MHz", gpu_render)
        @test occursin("Fan:", gpu_render)
        @test occursin("50.0%", gpu_render)
        
        # Test Progress content rendering
        progress_content = ProgressPanelContent(
            "Stage 2: Feature Selection", 45.0, 60.0, 1000, 2000, 0.85, 0.92, 3600
        )
        progress_render = ConsoleDashboard.render_progress_content(progress_content, config)
        @test occursin("Stage: ", progress_render)
        @test occursin("Stage 2: Feature Selection", progress_render)
        @test occursin("Overall:", progress_render)
        @test occursin("45.0%", progress_render)
        @test occursin("Stage:", progress_render)
        @test occursin("60.0%", progress_render)
        @test occursin("Items: 1000 / 2000", progress_render)
        @test occursin("Score: 0.850000", progress_render)
        @test occursin("Best: 0.920000", progress_render)
        @test occursin("ETA: 1h 0m", progress_render)
        
        # Test Metrics content rendering
        metrics_content = MetricsPanelContent(
            1500.0, 4.5, 65.0, 16.0, 85.0, 10, 24
        )
        metrics_render = ConsoleDashboard.render_metrics_content(metrics_content, config)
        @test occursin("Nodes/sec: 1500.00", metrics_render)
        @test occursin("Bandwidth: 4.50 GB/s", metrics_render)
        @test occursin("CPU:", metrics_render)
        @test occursin("65.0%", metrics_render)
        @test occursin("RAM: 16.00 GB", metrics_render)
        @test occursin("Cache:", metrics_render)
        @test occursin("85.0%", metrics_render)
        @test occursin("Threads: 24", metrics_render)
        @test occursin("Queue: 10", metrics_render)
        
        # Test Analysis content rendering
        analysis_content = AnalysisPanelContent(
            1000, 50, 95.0,
            [("feature_importance_1", 0.95), ("feature_corr_2", 0.88), ("feature_gain_3", 0.75)],
            "Correlation summary: Low multicollinearity detected"
        )
        analysis_render = ConsoleDashboard.render_analysis_content(analysis_content, config)
        @test occursin("Total Features: 1000", analysis_render)
        @test occursin("Selected: 50", analysis_render)
        # The reduction percentage will be color-formatted, so check for the value without color
        @test occursin("95.0%", analysis_render)
        @test occursin("Top Features:", analysis_render)
        @test occursin("1. feature_importance_1: 0.9500", analysis_render)
        @test occursin("2. feature_corr_2: 0.8800", analysis_render)
        @test occursin("3. feature_gain_3: 0.7500", analysis_render)
        @test occursin("Correlation summary: Low multicollinearity detected", analysis_render)
        
        # Test Log content rendering
        log_entries = [
            (now() - Minute(5), :info, "System initialized"),
            (now() - Minute(3), :warn, "High GPU temperature detected"),
            (now() - Minute(1), :error, "Failed to connect to database"),
            (now(), :info, "Retrying connection...")
        ]
        log_content = LogPanelContent(log_entries, 100)
        log_render = ConsoleDashboard.render_log_content(log_content, config)
        # Log levels are color-formatted, check for the content
        @test occursin("info", log_render)
        @test occursin("warn", log_render) 
        @test occursin("error", log_render)
        @test occursin("System initialized", log_render)
        @test occursin("High GPU temperature", log_render)
        @test occursin("Failed to connect", log_render)
        @test occursin("Retrying connection", log_render)
        
        # Test without timestamps
        config_no_ts = DashboardConfig(show_timestamps=false)
        log_render_no_ts = ConsoleDashboard.render_log_content(log_content, config_no_ts)
        # Check that timestamp pattern is not present (but may have ":" in level tags)
        @test length(findall(":", log_render_no_ts)) < length(log_entries)
    end
    
    @testset "Dashboard Creation Tests" begin
        # Test dashboard creation with default config
        # Skip if terminal is too small
        width, height = ConsoleDashboard.get_terminal_size()
        if width >= 80 && height >= 24
            dashboard = create_dashboard()
            @test isa(dashboard, DashboardLayout)
            @test length(dashboard.panels) == 6
            @test haskey(dashboard.panels, :gpu1)
            @test haskey(dashboard.panels, :gpu2)
            @test haskey(dashboard.panels, :progress)
            @test haskey(dashboard.panels, :metrics)
            @test haskey(dashboard.panels, :analysis)
            @test haskey(dashboard.panels, :log)
            
            # Test with custom config
            custom_config = DashboardConfig(
                border_style = :double,
                color_scheme = :dark
            )
            dashboard_custom = create_dashboard(custom_config)
            @test dashboard_custom.config.border_style == :double
            @test dashboard_custom.config.color_scheme == :dark
        else
            @test_skip "Terminal too small for dashboard tests"
        end
    end
    
    @testset "Terminal Size Tests" begin
        # Test get_terminal_size
        width, height = ConsoleDashboard.get_terminal_size()
        @test width > 0
        @test height > 0
        
        # Test adjust_panel_sizes
        # Very wide terminal
        adj_w, adj_h = ConsoleDashboard.adjust_panel_sizes(300, 40, 50, 20)
        @test adj_w > 50  # Should be wider
        @test adj_h == 20
        
        # Narrow terminal
        adj_w, adj_h = ConsoleDashboard.adjust_panel_sizes(80, 60, 25, 30)
        @test adj_w == 25
        @test adj_h > 30  # Should be taller
        
        # Standard terminal (aspect ratio 2.0, between 1.5 and 2.5)
        adj_w, adj_h = ConsoleDashboard.adjust_panel_sizes(160, 80, 40, 20)
        # Standard aspect ratio should return base values
        @test adj_w == 40
        @test adj_h == 20
    end
    
    @testset "Update Functions Tests" begin
        width, height = ConsoleDashboard.get_terminal_size()
        if width >= 80 && height >= 24
            dashboard = create_dashboard()
            
            # Test updating panels
            updates = Dict{Symbol, PanelContent}(
                :gpu1 => GPUPanelContent(1, 80.0, 9.0, 12.0, 70.0, 160.0, 1900.0, 55.0),
                :progress => ProgressPanelContent("Stage 3", 75.0, 90.0, 1800, 2000, 0.89, 0.92, 600),
                :metrics => MetricsPanelContent(2000.0, 5.0, 70.0, 18.0, 90.0, 5, 32)
            )
            
            # This should not throw an error
            ConsoleDashboard.update_dashboard!(dashboard, updates)
            
            # Test individual panel update
            log_content = LogPanelContent(
                [(now(), :info, "Test update")],
                100
            )
            ConsoleDashboard.update_panel!(dashboard, :log, log_content)
        else
            @test_skip "Terminal too small for update tests"
        end
    end
    
    @testset "Resize Handling Tests" begin
        width, height = ConsoleDashboard.get_terminal_size()
        if width >= 80 && height >= 24
            dashboard = create_dashboard()
            
            # Test resize detection
            resized, new_w, new_h = ConsoleDashboard.check_terminal_resize(dashboard)
            @test isa(resized, Bool)
            @test new_w > 0
            @test new_h > 0
            
            # Test handle_resize
            if resized
                new_dashboard = ConsoleDashboard.handle_resize!(dashboard, new_w, new_h)
                @test isa(new_dashboard, DashboardLayout)
            end
        else
            @test_skip "Terminal too small for resize tests"
        end
    end
    
    @testset "Render Dashboard Tests" begin
        width, height = ConsoleDashboard.get_terminal_size()
        if width >= 80 && height >= 24
            dashboard = create_dashboard()
            
            # Test rendering
            rendered = render_dashboard(dashboard)
            @test !isnothing(rendered)
            # We can't test much more without actually displaying,
            # but we verify it doesn't error
        else
            @test_skip "Terminal too small for render tests"
        end
    end
end

println("All console dashboard tests passed! ✓")