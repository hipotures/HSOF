"""
Test suite for Attention Mechanism Visualization
Testing attention weight extraction, aggregation, and visualization tools
"""

using Test
using CUDA
using Statistics
using LinearAlgebra
using Random

# Include required modules
include("../../src/metamodel/attention_visualization.jl")
using .AttentionVisualization

include("../../src/metamodel/neural_architecture.jl")
using .NeuralArchitecture

@testset "Attention Visualization Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        config = default_attention_viz_config()
        
        @test config.save_attention_weights == true
        @test config.max_stored_samples == 1000
        @test config.heatmap_resolution == (800, 600)
        @test config.color_scheme == "viridis"
        @test config.head_aggregation_method == "mean"
        @test config.min_attention_threshold == 0.01f0
        @test config.export_format == "png"
    end
    
    @testset "Attention Capturing Setup Tests" begin
        # Create base attention mechanism
        dim = 256
        num_heads = 8
        base_attention = MultiHeadAttention(dim, num_heads)
        
        # Create capturing wrapper
        config = default_attention_viz_config()
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        @test attn_cap.base_attention === base_attention
        @test attn_cap.config === config
        @test isempty(attn_cap.attention_weights_history)
        @test attn_cap.total_inferences == 0
        @test attn_cap.current_attention_weights === nothing
    end
    
    @testset "Forward Pass with Attention Capture Tests" begin
        dim = 128
        num_heads = 4
        batch_size = 8
        
        # Create attention mechanism
        base_attention = MultiHeadAttention(dim, num_heads)
        config = default_attention_viz_config()
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        # Create test input
        x = randn(Float32, dim, batch_size)
        feature_indices = collect(Int32(1):Int32(batch_size))
        
        # Forward pass
        output = attn_cap(x; feature_indices=feature_indices)
        
        # Verify output shape
        @test size(output) == (dim, batch_size)
        
        # Verify attention weights were captured
        @test attn_cap.current_attention_weights !== nothing
        @test size(attn_cap.current_attention_weights) == (batch_size, batch_size, num_heads)
        @test attn_cap.current_feature_indices == feature_indices
        @test attn_cap.total_inferences == 1
        
        # Verify weights are stored in history
        @test length(attn_cap.attention_weights_history) == 1
        @test length(attn_cap.feature_indices_history) == 1
        @test length(attn_cap.timestamps) == 1
    end
    
    @testset "Attention Weight Storage and Memory Management Tests" begin
        dim = 64
        num_heads = 2
        batch_size = 4
        
        # Create attention mechanism with small storage limit
        base_attention = MultiHeadAttention(dim, num_heads)
        config = default_attention_viz_config()
        config = AttentionVizConfig(
            config.save_attention_weights,
            5,  # max_stored_samples = 5
            config.heatmap_resolution,
            config.color_scheme,
            config.normalize_per_head,
            config.head_aggregation_method,
            config.feature_grouping_size,
            config.min_attention_threshold,
            config.export_format,
            config.export_quality,
            config.include_metadata,
            config.update_frequency_ms,
            config.max_features_display,
            config.enable_interactive
        )
        
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        # Generate multiple samples
        x = randn(Float32, dim, batch_size)
        
        for i in 1:10  # More than max_stored_samples
            output = attn_cap(x; feature_indices=collect(Int32(1):Int32(batch_size)))
        end
        
        # Verify memory management
        @test length(attn_cap.attention_weights_history) == 5  # Should be limited to max_stored_samples
        @test length(attn_cap.feature_indices_history) == 5
        @test length(attn_cap.timestamps) == 5
        @test attn_cap.total_inferences == 10
        @test attn_cap.memory_usage_mb > 0
    end
    
    @testset "Attention Head Aggregation Tests" begin
        batch_size = 6
        num_heads = 3
        
        # Create test attention weights
        attention_weights = randn(Float32, batch_size, batch_size, num_heads)
        attention_weights = abs.(attention_weights)  # Make positive
        
        # Test mean aggregation
        mean_aggregated = aggregate_attention_heads(attention_weights, "mean")
        @test size(mean_aggregated) == (batch_size, batch_size)
        
        expected_mean = mean(attention_weights, dims=3)[:, :, 1]
        @test mean_aggregated ≈ expected_mean
        
        # Test max aggregation
        max_aggregated = aggregate_attention_heads(attention_weights, "max")
        @test size(max_aggregated) == (batch_size, batch_size)
        
        expected_max = maximum(attention_weights, dims=3)[:, :, 1]
        @test max_aggregated ≈ expected_max
        
        # Test sum aggregation
        sum_aggregated = aggregate_attention_heads(attention_weights, "sum")
        @test size(sum_aggregated) == (batch_size, batch_size)
        
        expected_sum = sum(attention_weights, dims=3)[:, :, 1]
        @test sum_aggregated ≈ expected_sum
        
        # Test invalid method
        @test_throws ErrorException aggregate_attention_heads(attention_weights, "invalid")
    end
    
    @testset "Feature Interaction Heatmap Creation Tests" begin
        n_features = 10
        
        # Create test attention weights (symmetric matrix)
        attention_weights = randn(Float32, n_features, n_features)
        attention_weights = abs.(attention_weights)
        attention_weights = (attention_weights + attention_weights') / 2  # Make symmetric
        
        feature_indices = collect(Int32(1):Int32(n_features))
        config = default_attention_viz_config()
        
        # Create heatmap
        plot_obj, processed_weights = create_feature_interaction_heatmap(
            attention_weights, feature_indices, config
        )
        
        @test plot_obj !== nothing
        @test size(processed_weights) == size(attention_weights)
        
        # Test with threshold filtering
        config_with_threshold = AttentionVizConfig(
            config.save_attention_weights,
            config.max_stored_samples,
            config.heatmap_resolution,
            config.color_scheme,
            config.normalize_per_head,
            config.head_aggregation_method,
            config.feature_grouping_size,
            0.5f0,  # High threshold
            config.export_format,
            config.export_quality,
            config.include_metadata,
            config.update_frequency_ms,
            config.max_features_display,
            config.enable_interactive
        )
        
        _, filtered_weights = create_feature_interaction_heatmap(
            attention_weights, feature_indices, config_with_threshold
        )
        
        # Verify threshold was applied
        @test all(filtered_weights[filtered_weights .> 0] .>= 0.5f0)
    end
    
    @testset "Feature Grouping Tests" begin
        n_features = 12
        group_size = 3
        
        # Create test data
        attention_weights = randn(Float32, n_features, n_features)
        attention_weights = abs.(attention_weights)
        feature_labels = [string(i) for i in 1:n_features]
        
        # Test grouping
        grouped_weights, grouped_labels = AttentionVisualization.group_features(
            attention_weights, feature_labels, Int32(group_size)
        )
        
        expected_groups = ceil(Int, n_features / group_size)
        @test size(grouped_weights) == (expected_groups, expected_groups)
        @test length(grouped_labels) == expected_groups
        
        # Verify group labels
        @test grouped_labels[1] == "1-3"
        @test grouped_labels[2] == "4-6"
        @test grouped_labels[3] == "7-9"
        @test grouped_labels[4] == "10-12"
    end
    
    @testset "Top Interactions Extraction Tests" begin
        n_features = 8
        
        # Create test attention weights with known pattern
        attention_weights = zeros(Float32, n_features, n_features)
        
        # Set some strong interactions
        attention_weights[1, 3] = 0.9f0
        attention_weights[2, 5] = 0.8f0
        attention_weights[4, 7] = 0.7f0
        attention_weights[1, 6] = 0.6f0
        
        feature_indices = collect(Int32(101):Int32(108))  # Use different indices
        
        # Extract top interactions
        top_interactions = extract_top_interactions(
            attention_weights, feature_indices, 3
        )
        
        @test length(top_interactions) == 3
        
        # Verify order (highest first)
        @test top_interactions[1][3] ≈ 0.9f0  # weight
        @test top_interactions[2][3] ≈ 0.8f0
        @test top_interactions[3][3] ≈ 0.7f0
        
        # Verify feature indices
        @test top_interactions[1][1:2] == (101, 103)  # features 1->3 maps to 101->103
        @test top_interactions[2][1:2] == (102, 105)  # features 2->5 maps to 102->105
    end
    
    @testset "Attention Statistics Generation Tests" begin
        dim = 32
        num_heads = 2
        batch_size = 4
        
        # Create attention mechanism
        base_attention = MultiHeadAttention(dim, num_heads)
        config = default_attention_viz_config()
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        # Test with no data
        stats_empty = generate_attention_statistics(attn_cap)
        @test haskey(stats_empty, "error")
        
        # Generate some samples
        x = randn(Float32, dim, batch_size)
        
        for i in 1:5
            attn_cap(x; feature_indices=collect(Int32(1):Int32(batch_size)))
        end
        
        # Generate statistics
        stats = generate_attention_statistics(attn_cap)
        
        @test stats["total_samples"] == 5
        @test stats["total_inferences"] == 5
        @test stats["memory_usage_mb"] > 0
        
        # Verify attention weight statistics
        @test haskey(stats, "attention_weights")
        weight_stats = stats["attention_weights"]
        @test haskey(weight_stats, "mean")
        @test haskey(weight_stats, "std")
        @test haskey(weight_stats, "min")
        @test haskey(weight_stats, "max")
        @test haskey(weight_stats, "quantiles")
        
        # Verify per-head statistics
        @test haskey(stats, "per_head")
        @test length(stats["per_head"]) == num_heads
        
        for head_stat in stats["per_head"]
            @test haskey(head_stat, "head")
            @test haskey(head_stat, "mean")
            @test haskey(head_stat, "std")
            @test haskey(head_stat, "entropy")
        end
        
        # Verify temporal statistics
        @test haskey(stats, "temporal")
        temporal_stats = stats["temporal"]
        @test haskey(temporal_stats, "mean_trend")
        @test haskey(temporal_stats, "variance_over_time")
        @test haskey(temporal_stats, "stability_score")
    end
    
    @testset "Linear Trend Calculation Tests" begin
        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = AttentionVisualization.linear_trend(increasing_values)
        @test trend > 0  # Should be positive
        
        # Test decreasing trend
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = AttentionVisualization.linear_trend(decreasing_values)
        @test trend < 0  # Should be negative
        
        # Test flat trend
        flat_values = [3.0, 3.0, 3.0, 3.0, 3.0]
        trend = AttentionVisualization.linear_trend(flat_values)
        @test abs(trend) < 1e-10  # Should be near zero
        
        # Test edge cases
        @test AttentionVisualization.linear_trend([1.0]) == 0.0
        @test AttentionVisualization.linear_trend(Float64[]) == 0.0
    end
    
    @testset "Real-time Dashboard Update Tests" begin
        dim = 64
        num_heads = 4
        batch_size = 16
        
        # Create attention mechanism
        base_attention = MultiHeadAttention(dim, num_heads)
        config = default_attention_viz_config()
        
        # Set lower max features for testing
        config = AttentionVizConfig(
            config.save_attention_weights,
            config.max_stored_samples,
            config.heatmap_resolution,
            config.color_scheme,
            config.normalize_per_head,
            config.head_aggregation_method,
            config.feature_grouping_size,
            config.min_attention_threshold,
            config.export_format,
            config.export_quality,
            config.include_metadata,
            config.update_frequency_ms,
            8,  # max_features_display
            config.enable_interactive
        )
        
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        # Test with no current weights
        plot_obj = update_realtime_dashboard(attn_cap)
        @test plot_obj === nothing
        
        # Generate a sample
        x = randn(Float32, dim, batch_size)
        feature_indices = collect(Int32(1):Int32(batch_size))
        attn_cap(x; feature_indices=feature_indices)
        
        # Test real-time update
        plot_obj = update_realtime_dashboard(attn_cap)
        @test plot_obj !== nothing
        
        # Test with fewer features than max display
        x_small = randn(Float32, dim, 4)
        feature_indices_small = collect(Int32(1):Int32(4))
        attn_cap(x_small; feature_indices=feature_indices_small)
        
        plot_obj_small = update_realtime_dashboard(attn_cap)
        @test plot_obj_small !== nothing
    end
    
    @testset "Attention Head Comparison Tests" begin
        batch_size = 6
        num_heads = 4
        
        # Create test attention weights
        attention_weights = rand(Float32, batch_size, batch_size, num_heads)
        feature_indices = collect(Int32(1):Int32(batch_size))
        config = default_attention_viz_config()
        
        # Create comparison plot
        comparison_plot = create_attention_heads_comparison(
            attention_weights, feature_indices, config
        )
        
        @test comparison_plot !== nothing
        
        # Test with empty feature indices
        comparison_plot_empty = create_attention_heads_comparison(
            attention_weights, Int32[], config
        )
        
        @test comparison_plot_empty !== nothing
    end
    
    @testset "Interactive Explorer Creation Tests" begin
        dim = 32
        num_heads = 2
        batch_size = 4
        
        # Create attention mechanism
        base_attention = MultiHeadAttention(dim, num_heads)
        config = default_attention_viz_config()
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        # Test with no data
        explorer_empty = create_interactive_explorer(attn_cap)
        @test explorer_empty === nothing
        
        # Generate some samples
        x = randn(Float32, dim, batch_size)
        feature_indices = collect(Int32(1):Int32(batch_size))
        
        for i in 1:3
            attn_cap(x; feature_indices=feature_indices)
        end
        
        # Create explorer
        explorer = create_interactive_explorer(attn_cap)
        @test explorer !== nothing
    end
    
    @testset "Integration Tests with Neural Architecture" begin
        # Test full integration with neural architecture
        dim = 128
        num_heads = 8
        batch_size = 16
        
        # Create metamodel with attention visualization
        base_attention = MultiHeadAttention(dim, num_heads)
        config = default_attention_viz_config()
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        # Test multiple forward passes
        for i in 1:10
            x = randn(Float32, dim, batch_size)
            feature_indices = rand(Int32(1):Int32(500), batch_size)  # Realistic feature indices
            
            output = attn_cap(x; feature_indices=feature_indices)
            
            @test size(output) == (dim, batch_size)
            @test attn_cap.current_attention_weights !== nothing
            @test attn_cap.current_feature_indices == feature_indices
        end
        
        @test attn_cap.total_inferences == 10
        @test length(attn_cap.attention_weights_history) == 10
        
        # Test statistical analysis
        stats = generate_attention_statistics(attn_cap)
        @test stats["total_samples"] == 10
        @test stats["total_inferences"] == 10
        
        # Test visualization generation
        latest_weights = attn_cap.attention_weights_history[end]
        latest_indices = attn_cap.feature_indices_history[end]
        
        # Test aggregated heatmap
        aggregated = aggregate_attention_heads(latest_weights, "mean")
        plot_obj, _ = create_feature_interaction_heatmap(aggregated, latest_indices, config)
        @test plot_obj !== nothing
        
        # Test head comparison
        heads_plot = create_attention_heads_comparison(latest_weights, latest_indices, config)
        @test heads_plot !== nothing
        
        # Test top interactions
        top_interactions = extract_top_interactions(aggregated, latest_indices, 10)
        @test length(top_interactions) <= 10
        @test all(interaction -> length(interaction) == 3, top_interactions)  # (feature_i, feature_j, weight)
        
        # Test real-time dashboard
        dashboard_plot = update_realtime_dashboard(attn_cap)
        @test dashboard_plot !== nothing
    end
    
    @testset "Performance and Memory Tests" begin
        dim = 256
        num_heads = 8
        batch_size = 32
        
        # Create attention mechanism
        base_attention = MultiHeadAttention(dim, num_heads)
        config = default_attention_viz_config()
        attn_cap = AttentionCapturingMultiHeadAttention(base_attention, config)
        
        # Measure performance of forward pass with visualization
        x = randn(Float32, dim, batch_size)
        feature_indices = collect(Int32(1):Int32(batch_size))
        
        # Time multiple forward passes
        start_time = time()
        for i in 1:100
            output = attn_cap(x; feature_indices=feature_indices)
        end
        elapsed = time() - start_time
        
        avg_time_per_inference = elapsed / 100
        @test avg_time_per_inference < 0.1  # Should be less than 100ms per inference
        
        # Check memory usage tracking
        @test attn_cap.memory_usage_mb > 0
        @test attn_cap.total_inferences == 100
        
        # Test memory cleanup with limits
        @test length(attn_cap.attention_weights_history) <= config.max_stored_samples
    end
end

println("Attention Visualization tests completed successfully!")