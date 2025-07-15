"""
Attention Mechanism Visualization
Tools to interpret and visualize attention weights showing which feature interactions
the metamodel considers important
"""
module AttentionVisualization

using CUDA
using Statistics
using LinearAlgebra
using Plots
using PlotlyJS
using Colors
using JSON3
using FileIO
using Printf

# Include neural architecture for attention extraction
include("neural_architecture.jl")
using .NeuralArchitecture

"""
Configuration for attention visualization
"""
struct AttentionVizConfig
    # Attention extraction
    save_attention_weights::Bool        # Whether to save attention weights during inference
    max_stored_samples::Int32           # Maximum samples to store in memory
    
    # Visualization settings
    heatmap_resolution::Tuple{Int32, Int32}  # Resolution for heatmaps (width, height)
    color_scheme::String                # Color scheme for visualizations
    normalize_per_head::Bool            # Normalize attention weights per head
    
    # Aggregation settings
    head_aggregation_method::String     # How to aggregate across heads: "mean", "max", "sum"
    feature_grouping_size::Int32        # Group features for visualization (1 = no grouping)
    min_attention_threshold::Float32    # Minimum attention weight to display
    
    # Export settings
    export_format::String              # Export format: "png", "svg", "html", "json"
    export_quality::Int32              # Export quality for raster formats
    include_metadata::Bool             # Include metadata in exports
    
    # Real-time display
    update_frequency_ms::Int32         # Update frequency for real-time display
    max_features_display::Int32        # Maximum features to show in real-time
    enable_interactive::Bool           # Enable interactive visualizations
end

"""
Default configuration for attention visualization
"""
function default_attention_viz_config()
    AttentionVizConfig(
        true,                          # save_attention_weights
        1000,                          # max_stored_samples
        (800, 600),                    # heatmap_resolution
        "viridis",                     # color_scheme
        true,                          # normalize_per_head
        "mean",                        # head_aggregation_method
        10,                            # feature_grouping_size
        0.01f0,                        # min_attention_threshold
        "png",                         # export_format
        300,                           # export_quality
        true,                          # include_metadata
        100,                           # update_frequency_ms
        50,                            # max_features_display
        true                           # enable_interactive
    )
end

"""
Modified MultiHeadAttention that captures attention weights
"""
mutable struct AttentionCapturingMultiHeadAttention
    base_attention::MultiHeadAttention
    config::AttentionVizConfig
    
    # Storage for attention weights
    attention_weights_history::Vector{Array{Float32, 3}}  # Stored attention weights
    feature_indices_history::Vector{Vector{Int32}}        # Feature indices for each sample
    timestamps::Vector{Float64}                           # Timestamps for each sample
    
    # Current batch attention weights
    current_attention_weights::Union{Nothing, Array{Float32, 3}}
    current_feature_indices::Union{Nothing, Vector{Int32}}
    
    # Statistics
    total_inferences::Int64
    memory_usage_mb::Float32
end

"""
Create attention capturing wrapper
"""
function AttentionCapturingMultiHeadAttention(
    base_attention::MultiHeadAttention,
    config::AttentionVizConfig = default_attention_viz_config()
)
    AttentionCapturingMultiHeadAttention(
        base_attention,
        config,
        Vector{Array{Float32, 3}}(),
        Vector{Vector{Int32}}(),
        Vector{Float64}(),
        nothing,
        nothing,
        0,
        0.0f0
    )
end

"""
Forward pass with attention weight capture
"""
function (attn_cap::AttentionCapturingMultiHeadAttention)(x::AbstractArray; feature_indices::Union{Nothing, Vector{Int32}} = nothing)
    # Call base attention mechanism
    output, attention_weights = forward_with_attention_weights(attn_cap.base_attention, x)
    
    # Store attention weights if enabled
    if attn_cap.config.save_attention_weights
        store_attention_weights!(attn_cap, attention_weights, feature_indices)
    end
    
    # Update current weights for real-time visualization
    attn_cap.current_attention_weights = attention_weights
    attn_cap.current_feature_indices = feature_indices
    attn_cap.total_inferences += 1
    
    return output
end

"""
Modified forward pass that returns attention weights
"""
function forward_with_attention_weights(mha::MultiHeadAttention, x::AbstractArray)
    batch_size = size(x, 2)
    seq_len = 1  # For feature vectors, sequence length is 1
    dim = size(x, 1)
    
    # Linear projections in batch (batch_size, dim)
    Q = mha.W_q(x)  # (dim, batch_size)
    K = mha.W_k(x)  # (dim, batch_size)
    V = mha.W_v(x)  # (dim, batch_size)
    
    # Reshape for multi-head attention
    # Split into multiple heads: (head_dim, num_heads, batch_size)
    Q = reshape(Q, mha.head_dim, mha.num_heads, batch_size)
    K = reshape(K, mha.head_dim, mha.num_heads, batch_size)
    V = reshape(V, mha.head_dim, mha.num_heads, batch_size)
    
    # Compute attention scores for each head
    # scores = Q^T * K / sqrt(head_dim)
    # Use similar to allocate on same device as input
    scores = similar(x, batch_size, batch_size, mha.num_heads)
    
    # Use batched matrix multiplication for efficiency
    scale = eltype(x)(1.0 / sqrt(mha.head_dim))
    for h in 1:mha.num_heads
        # Extract head h for all batches
        Q_h = Q[:, h, :]  # (head_dim, batch_size)
        K_h = K[:, h, :]  # (head_dim, batch_size)
        
        # Compute attention scores: Q_h' * K_h
        scores_h = @view scores[:, :, h]
        mul!(scores_h, Q_h', K_h)
        scores_h .*= scale
    end
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, dims=2)
    
    # Store attention weights before dropout (for visualization)
    attention_weights_viz = copy(attention_weights)
    
    # Apply dropout
    attention_weights = mha.dropout(attention_weights)
    
    # Apply attention to values
    output = similar(x, mha.head_dim, mha.num_heads, batch_size)
    
    for h in 1:mha.num_heads
        V_h = V[:, h, :]  # (head_dim, batch_size)
        weights_h = @view attention_weights[:, :, h]  # (batch_size, batch_size)
        output_h = @view output[:, h, :]  # (head_dim, batch_size)
        
        # Apply attention: V_h * weights_h
        mul!(output_h, V_h, weights_h)
    end
    
    # Reshape and combine heads
    output = reshape(output, dim, batch_size)
    
    # Output projection
    output = mha.W_o(output)
    
    return output, Array(attention_weights_viz)  # Convert to CPU array for visualization
end

"""
Store attention weights with memory management
"""
function store_attention_weights!(
    attn_cap::AttentionCapturingMultiHeadAttention,
    attention_weights::Array{Float32, 3},
    feature_indices::Union{Nothing, Vector{Int32}}
)
    # Add to history
    push!(attn_cap.attention_weights_history, attention_weights)
    push!(attn_cap.feature_indices_history, feature_indices === nothing ? Int32[] : feature_indices)
    push!(attn_cap.timestamps, time())
    
    # Memory management - remove oldest if exceeding limit
    if length(attn_cap.attention_weights_history) > attn_cap.config.max_stored_samples
        popfirst!(attn_cap.attention_weights_history)
        popfirst!(attn_cap.feature_indices_history)
        popfirst!(attn_cap.timestamps)
    end
    
    # Update memory usage estimate
    sample_size = sizeof(attention_weights) + sizeof(feature_indices) + sizeof(Float64)
    attn_cap.memory_usage_mb = Float32(length(attn_cap.attention_weights_history) * sample_size / 1024 / 1024)
end

"""
Aggregate attention weights across heads
"""
function aggregate_attention_heads(
    attention_weights::Array{Float32, 3},
    method::String = "mean"
)
    batch_size, _, num_heads = size(attention_weights)
    
    if method == "mean"
        return mean(attention_weights, dims=3)[:, :, 1]
    elseif method == "max"
        return maximum(attention_weights, dims=3)[:, :, 1]
    elseif method == "sum"
        return sum(attention_weights, dims=3)[:, :, 1]
    else
        error("Unknown aggregation method: $method")
    end
end

"""
Create feature interaction heatmap
"""
function create_feature_interaction_heatmap(
    attention_weights::Array{Float32, 2},
    feature_indices::Vector{Int32} = Int32[],
    config::AttentionVizConfig = default_attention_viz_config()
)
    # Apply minimum threshold
    filtered_weights = copy(attention_weights)
    filtered_weights[filtered_weights .< config.min_attention_threshold] .= 0.0f0
    
    # Normalize if requested
    if config.normalize_per_head
        max_weight = maximum(filtered_weights)
        if max_weight > 0
            filtered_weights ./= max_weight
        end
    end
    
    # Create feature labels
    if isempty(feature_indices)
        feature_labels = [string(i) for i in 1:size(filtered_weights, 1)]
    else
        feature_labels = [string(idx) for idx in feature_indices]
    end
    
    # Group features if requested
    if config.feature_grouping_size > 1
        grouped_weights, grouped_labels = group_features(filtered_weights, feature_labels, config.feature_grouping_size)
    else
        grouped_weights = filtered_weights
        grouped_labels = feature_labels
    end
    
    # Create heatmap
    p = heatmap(
        grouped_weights,
        xlabel="Features",
        ylabel="Features", 
        title="Feature Interaction Attention Weights",
        color=Symbol(config.color_scheme),
        aspect_ratio=:equal,
        size=config.heatmap_resolution,
        xticks=(1:length(grouped_labels), grouped_labels),
        yticks=(1:length(grouped_labels), grouped_labels),
        xrotation=45
    )
    
    return p, grouped_weights
end

"""
Group features for visualization
"""
function group_features(
    attention_weights::Array{Float32, 2},
    feature_labels::Vector{String},
    group_size::Int32
)
    n_features = size(attention_weights, 1)
    n_groups = ceil(Int, n_features / group_size)
    
    grouped_weights = zeros(Float32, n_groups, n_groups)
    grouped_labels = String[]
    
    for i in 1:n_groups
        start_i = (i-1) * group_size + 1
        end_i = min(i * group_size, n_features)
        
        for j in 1:n_groups
            start_j = (j-1) * group_size + 1
            end_j = min(j * group_size, n_features)
            
            # Aggregate attention weights in group
            group_weights = attention_weights[start_i:end_i, start_j:end_j]
            grouped_weights[i, j] = mean(group_weights)
        end
        
        # Create group label
        if start_i == end_i
            push!(grouped_labels, feature_labels[start_i])
        else
            push!(grouped_labels, "$(feature_labels[start_i])-$(feature_labels[end_i])")
        end
    end
    
    return grouped_weights, grouped_labels
end

"""
Create attention head comparison visualization
"""
function create_attention_heads_comparison(
    attention_weights::Array{Float32, 3},
    feature_indices::Vector{Int32} = Int32[],
    config::AttentionVizConfig = default_attention_viz_config()
)
    batch_size, _, num_heads = size(attention_weights)
    
    plots = []
    
    for head in 1:num_heads
        head_weights = attention_weights[:, :, head]
        
        # Apply threshold and normalization
        filtered_weights = copy(head_weights)
        filtered_weights[filtered_weights .< config.min_attention_threshold] .= 0.0f0
        
        if config.normalize_per_head
            max_weight = maximum(filtered_weights)
            if max_weight > 0
                filtered_weights ./= max_weight
            end
        end
        
        # Create feature labels
        if isempty(feature_indices)
            feature_labels = [string(i) for i in 1:size(filtered_weights, 1)]
        else
            feature_labels = [string(idx) for idx in feature_indices]
        end
        
        # Create heatmap for this head
        p = heatmap(
            filtered_weights,
            title="Attention Head $head",
            color=Symbol(config.color_scheme),
            aspect_ratio=:equal,
            showaxis=false,
            grid=false
        )
        
        push!(plots, p)
    end
    
    # Combine all head plots
    combined_plot = plot(plots..., layout=(2, 4), size=(1600, 800))
    
    return combined_plot
end

"""
Extract top feature interactions
"""
function extract_top_interactions(
    attention_weights::Array{Float32, 2},
    feature_indices::Vector{Int32} = Int32[],
    top_k::Int = 20
)
    n_features = size(attention_weights, 1)
    
    # Create list of all interactions (excluding diagonal)
    interactions = []
    
    for i in 1:n_features
        for j in 1:n_features
            if i != j  # Exclude self-attention
                weight = attention_weights[i, j]
                
                if isempty(feature_indices)
                    feature_i = i
                    feature_j = j
                else
                    feature_i = feature_indices[i]
                    feature_j = feature_indices[j]
                end
                
                push!(interactions, (feature_i, feature_j, weight))
            end
        end
    end
    
    # Sort by attention weight
    sort!(interactions, by=x->x[3], rev=true)
    
    # Return top k interactions
    return interactions[1:min(top_k, length(interactions))]
end

"""
Generate attention summary statistics
"""
function generate_attention_statistics(
    attn_cap::AttentionCapturingMultiHeadAttention
)
    if isempty(attn_cap.attention_weights_history)
        return Dict{String, Any}("error" => "No attention weights stored")
    end
    
    stats = Dict{String, Any}()
    
    # Basic statistics
    stats["total_samples"] = length(attn_cap.attention_weights_history)
    stats["total_inferences"] = attn_cap.total_inferences
    stats["memory_usage_mb"] = attn_cap.memory_usage_mb
    
    # Attention weight statistics across all samples
    all_weights = vcat([vec(w) for w in attn_cap.attention_weights_history]...)
    
    stats["attention_weights"] = Dict(
        "mean" => mean(all_weights),
        "std" => std(all_weights),
        "min" => minimum(all_weights),
        "max" => maximum(all_weights),
        "median" => median(all_weights),
        "quantiles" => [quantile(all_weights, q) for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]]
    )
    
    # Per-head statistics
    if !isempty(attn_cap.attention_weights_history)
        first_sample = attn_cap.attention_weights_history[1]
        num_heads = size(first_sample, 3)
        
        head_stats = []
        for head in 1:num_heads
            head_weights = vcat([vec(w[:, :, head]) for w in attn_cap.attention_weights_history]...)
            
            head_stat = Dict(
                "head" => head,
                "mean" => mean(head_weights),
                "std" => std(head_weights),
                "entropy" => -sum(head_weights .* log.(head_weights .+ 1e-8)) / length(head_weights)
            )
            
            push!(head_stats, head_stat)
        end
        
        stats["per_head"] = head_stats
    end
    
    # Temporal statistics (if multiple samples)
    if length(attn_cap.attention_weights_history) > 1
        temporal_means = [mean(w) for w in attn_cap.attention_weights_history]
        
        stats["temporal"] = Dict(
            "mean_trend" => linear_trend(temporal_means),
            "variance_over_time" => var(temporal_means),
            "stability_score" => 1.0 / (1.0 + var(temporal_means))
        )
    end
    
    return stats
end

"""
Calculate linear trend
"""
function linear_trend(values::Vector{Float64})
    n = length(values)
    if n < 2
        return 0.0
    end
    
    x = collect(1:n)
    y = values
    
    # Simple linear regression
    x_mean = mean(x)
    y_mean = mean(y)
    
    numerator = sum((x .- x_mean) .* (y .- y_mean))
    denominator = sum((x .- x_mean).^2)
    
    if denominator == 0
        return 0.0
    end
    
    return numerator / denominator
end

"""
Export attention visualization
"""
function export_attention_visualization(
    plot_object,
    filename::String,
    config::AttentionVizConfig = default_attention_viz_config()
)
    # Create full filename with format
    if !endswith(filename, "." * config.export_format)
        full_filename = filename * "." * config.export_format
    else
        full_filename = filename
    end
    
    # Export based on format
    if config.export_format in ["png", "svg", "pdf"]
        savefig(plot_object, full_filename)
    elseif config.export_format == "html"
        PlotlyJS.savefig(plot_object, full_filename)
    else
        @warn "Unsupported export format: $(config.export_format)"
        return false
    end
    
    @info "Attention visualization exported to: $full_filename"
    return true
end

"""
Export attention data as JSON
"""
function export_attention_data(
    attn_cap::AttentionCapturingMultiHeadAttention,
    filename::String
)
    if isempty(attn_cap.attention_weights_history)
        @warn "No attention data to export"
        return false
    end
    
    # Prepare data for export
    export_data = Dict(
        "config" => attn_cap.config,
        "statistics" => generate_attention_statistics(attn_cap),
        "samples" => []
    )
    
    # Add sample data
    for (i, (weights, indices, timestamp)) in enumerate(zip(
        attn_cap.attention_weights_history,
        attn_cap.feature_indices_history,
        attn_cap.timestamps
    ))
        sample_data = Dict(
            "sample_id" => i,
            "timestamp" => timestamp,
            "feature_indices" => indices,
            "attention_weights" => weights,
            "aggregated_weights" => aggregate_attention_heads(weights, attn_cap.config.head_aggregation_method)
        )
        
        push!(export_data["samples"], sample_data)
    end
    
    # Write to file
    try
        open(filename, "w") do f
            JSON3.write(f, export_data)
        end
        @info "Attention data exported to: $filename"
        return true
    catch e
        @error "Failed to export attention data: $e"
        return false
    end
end

"""
Real-time attention dashboard update
"""
function update_realtime_dashboard(
    attn_cap::AttentionCapturingMultiHeadAttention
)
    if attn_cap.current_attention_weights === nothing
        return nothing
    end
    
    # Aggregate across heads
    aggregated_weights = aggregate_attention_heads(
        attn_cap.current_attention_weights,
        attn_cap.config.head_aggregation_method
    )
    
    # Limit features for display
    max_features = min(attn_cap.config.max_features_display, size(aggregated_weights, 1))
    
    if max_features < size(aggregated_weights, 1)
        # Select top features by attention sum
        feature_attention_sums = sum(aggregated_weights, dims=2)[:, 1]
        top_indices = sortperm(feature_attention_sums, rev=true)[1:max_features]
        
        display_weights = aggregated_weights[top_indices, top_indices]
        display_feature_indices = attn_cap.current_feature_indices === nothing ? 
            top_indices : attn_cap.current_feature_indices[top_indices]
    else
        display_weights = aggregated_weights
        display_feature_indices = attn_cap.current_feature_indices === nothing ?
            collect(1:max_features) : attn_cap.current_feature_indices
    end
    
    # Create real-time heatmap
    p, _ = create_feature_interaction_heatmap(
        display_weights,
        display_feature_indices,
        attn_cap.config
    )
    
    # Add timestamp and statistics
    title_text = @sprintf("Real-time Feature Interactions (Sample #%d)", attn_cap.total_inferences)
    plot!(p, title=title_text)
    
    return p
end

"""
Create interactive attention explorer
"""
function create_interactive_explorer(
    attn_cap::AttentionCapturingMultiHeadAttention
)
    if isempty(attn_cap.attention_weights_history)
        @warn "No attention data available for exploration"
        return nothing
    end
    
    # Use PlotlyJS for interactivity
    plotlyjs()
    
    # Create sample selector
    n_samples = length(attn_cap.attention_weights_history)
    
    # Start with the most recent sample
    latest_weights = attn_cap.attention_weights_history[end]
    latest_indices = attn_cap.feature_indices_history[end]
    
    # Create head-by-head comparison
    heads_plot = create_attention_heads_comparison(latest_weights, latest_indices, attn_cap.config)
    
    # Create aggregated view
    aggregated_weights = aggregate_attention_heads(latest_weights, attn_cap.config.head_aggregation_method)
    main_plot, _ = create_feature_interaction_heatmap(aggregated_weights, latest_indices, attn_cap.config)
    
    # Combine plots
    combined = plot(
        main_plot, heads_plot,
        layout=(2, 1),
        size=(1200, 1600),
        title="Interactive Attention Explorer - Sample $n_samples"
    )
    
    return combined
end

# Export main types and functions
export AttentionVizConfig, AttentionCapturingMultiHeadAttention
export default_attention_viz_config
export create_feature_interaction_heatmap, create_attention_heads_comparison
export extract_top_interactions, generate_attention_statistics
export export_attention_visualization, export_attention_data
export update_realtime_dashboard, create_interactive_explorer
export aggregate_attention_heads

end # module AttentionVisualization