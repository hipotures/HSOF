module FeatureHeatmap

using CUDA
using JSON3
using Dates
using Printf
using Statistics
using Plots
using ColorSchemes

export HeatmapGenerator, update_feature_count!, update_heatmap_data!
export generate_heatmap, save_heatmap, clear!

"""
Feature selection heatmap generator
"""
mutable struct HeatmapGenerator
    feature_counts::Matrix{Int}  # [feature_id, tree_id]
    feature_scores::Matrix{Float32}  # [feature_id, tree_id]
    n_features::Int
    n_trees::Int
    update_count::Int
    last_update::DateTime
    
    function HeatmapGenerator(; n_features::Int = 500, n_trees::Int = 100)
        new(
            zeros(Int, n_features, n_trees),
            zeros(Float32, n_features, n_trees),
            n_features,
            n_trees,
            0,
            now()
        )
    end
end

"""
Update feature selection count
"""
function update_feature_count!(
    gen::HeatmapGenerator,
    feature_id::Int,
    tree_id::Int;
    score::Union{Float32, Nothing} = nothing
)
    if feature_id < 1 || feature_id > gen.n_features
        @warn "Invalid feature_id: $feature_id"
        return
    end
    
    if tree_id < 1 || tree_id > gen.n_trees
        @warn "Invalid tree_id: $tree_id"
        return
    end
    
    # Update count
    gen.feature_counts[feature_id, tree_id] += 1
    
    # Update score if provided
    if !isnothing(score)
        gen.feature_scores[feature_id, tree_id] = score
    end
    
    gen.update_count += 1
    gen.last_update = now()
end

"""
Update heatmap with batch data
"""
function update_heatmap_data!(
    gen::HeatmapGenerator,
    feature_data::Matrix{Float32}
)
    # Validate dimensions
    if size(feature_data, 1) != gen.n_features || size(feature_data, 2) != gen.n_trees
        error("Feature data dimensions mismatch. Expected ($(gen.n_features), $(gen.n_trees)), got $(size(feature_data))")
    end
    
    # Update scores
    gen.feature_scores .= feature_data
    gen.update_count += 1
    gen.last_update = now()
end

"""
Generate heatmap visualization
"""
function generate_heatmap(gen::HeatmapGenerator; mode::Symbol = :count)
    # Select data based on mode
    data = if mode == :count
        gen.feature_counts
    elseif mode == :score
        gen.feature_scores
    else
        error("Invalid mode: $mode. Use :count or :score")
    end
    
    # Create main heatmap plot
    plt = plot(
        size = (1400, 800),
        layout = (2, 2)
    )
    
    # Plot 1: Full heatmap
    create_full_heatmap!(plt[1], data, gen)
    
    # Plot 2: Feature importance (aggregated across trees)
    create_feature_importance!(plt[2], data, gen)
    
    # Plot 3: Tree diversity (feature selection patterns)
    create_tree_diversity!(plt[3], data, gen)
    
    # Plot 4: Selection consensus
    create_consensus_plot!(plt[4], data, gen)
    
    return plt
end

"""
Create full feature selection heatmap
"""
function create_full_heatmap!(plt::Plots.Subplot, data::Matrix, gen::HeatmapGenerator)
    # Downsample if too many features
    display_features = if gen.n_features > 100
        # Show top 100 most selected features
        feature_sums = sum(data, dims=2)[:, 1]
        top_indices = sortperm(feature_sums, rev=true)[1:100]
        data[top_indices, :]
    else
        data
    end
    
    heatmap!(plt,
        display_features',
        color = :viridis,
        xlabel = "Feature Index",
        ylabel = "Tree ID",
        title = "Feature Selection Heatmap",
        colorbar_title = "Selection Count",
        aspect_ratio = :auto
    )
end

"""
Create feature importance plot
"""
function create_feature_importance!(plt::Plots.Subplot, data::Matrix, gen::HeatmapGenerator)
    # Calculate feature importance (sum across all trees)
    feature_importance = sum(data, dims=2)[:, 1]
    
    # Get top 50 features
    top_n = min(50, gen.n_features)
    top_indices = sortperm(feature_importance, rev=true)[1:top_n]
    top_values = feature_importance[top_indices]
    
    bar!(plt,
        top_indices,
        top_values,
        xlabel = "Feature Index",
        ylabel = "Total Selection Count",
        title = "Top $top_n Feature Importance",
        color = :steelblue,
        legend = false
    )
    
    # Add threshold line for average
    avg_importance = mean(feature_importance[feature_importance .> 0])
    hline!(plt, [avg_importance], 
        color = :red, 
        linestyle = :dash, 
        label = "Average"
    )
end

"""
Create tree diversity visualization
"""
function create_tree_diversity!(plt::Plots.Subplot, data::Matrix, gen::HeatmapGenerator)
    # Calculate diversity metrics for each tree
    tree_diversity = Float64[]
    
    for tree_id in 1:gen.n_trees
        tree_selections = data[:, tree_id]
        selected_features = findall(tree_selections .> 0)
        
        if !isempty(selected_features)
            # Calculate entropy as diversity measure
            probs = tree_selections[selected_features] ./ sum(tree_selections[selected_features])
            entropy = -sum(p * log(p + 1e-10) for p in probs)
            push!(tree_diversity, entropy)
        else
            push!(tree_diversity, 0.0)
        end
    end
    
    scatter!(plt,
        1:gen.n_trees,
        tree_diversity,
        xlabel = "Tree ID",
        ylabel = "Selection Diversity (Entropy)",
        title = "Tree Selection Diversity",
        color = :green,
        markersize = 4,
        label = "Diversity"
    )
    
    # Add trend line
    if length(tree_diversity) > 10
        trend = smooth_trend(tree_diversity, 10)
        plot!(plt, 1:length(trend), trend,
            color = :red,
            linewidth = 2,
            label = "Trend"
        )
    end
end

"""
Create consensus plot showing agreement between trees
"""
function create_consensus_plot!(plt::Plots.Subplot, data::Matrix, gen::HeatmapGenerator)
    # Calculate consensus scores for each feature
    consensus_scores = Float64[]
    feature_indices = Int[]
    
    for feature_id in 1:gen.n_features
        feature_selections = data[feature_id, :]
        n_selecting_trees = count(feature_selections .> 0)
        
        if n_selecting_trees > 0
            # Consensus = percentage of trees selecting this feature
            consensus = n_selecting_trees / gen.n_trees * 100
            push!(consensus_scores, consensus)
            push!(feature_indices, feature_id)
        end
    end
    
    # Sort by consensus
    sorted_indices = sortperm(consensus_scores, rev=true)
    top_n = min(30, length(sorted_indices))
    
    bar!(plt,
        feature_indices[sorted_indices[1:top_n]],
        consensus_scores[sorted_indices[1:top_n]],
        xlabel = "Feature Index",
        ylabel = "Consensus (%)",
        title = "Feature Selection Consensus",
        color = :purple,
        legend = false
    )
    
    # Add threshold lines
    hline!(plt, [50], color = :orange, linestyle = :dash, label = "50% consensus")
    hline!(plt, [75], color = :red, linestyle = :dash, label = "75% consensus")
end

"""
Clear heatmap data
"""
function clear!(gen::HeatmapGenerator)
    fill!(gen.feature_counts, 0)
    fill!(gen.feature_scores, 0.0f0)
    gen.update_count = 0
    gen.last_update = now()
end

"""
Save heatmap visualization
"""
function save_heatmap(gen::HeatmapGenerator, filepath::String; mode::Symbol = :count)
    plt = generate_heatmap(gen, mode = mode)
    savefig(plt, filepath)
end

"""
Smooth trend calculation
"""
function smooth_trend(data::Vector{Float64}, window::Int)
    n = length(data)
    trend = similar(data)
    
    for i in 1:n
        start_idx = max(1, i - div(window, 2))
        end_idx = min(n, i + div(window, 2))
        trend[i] = mean(data[start_idx:end_idx])
    end
    
    return trend
end

"""
Export heatmap data to JSON
"""
function export_heatmap_data(gen::HeatmapGenerator, filepath::String)
    # Prepare data for export
    export_data = Dict{String, Any}(
        "metadata" => Dict(
            "n_features" => gen.n_features,
            "n_trees" => gen.n_trees,
            "update_count" => gen.update_count,
            "last_update" => gen.last_update
        ),
        "feature_statistics" => calculate_feature_statistics(gen),
        "tree_statistics" => calculate_tree_statistics(gen),
        "consensus_features" => get_consensus_features(gen)
    )
    
    # Write to file
    open(filepath, "w") do io
        JSON3.pretty(io, export_data)
    end
end

"""
Calculate feature-level statistics
"""
function calculate_feature_statistics(gen::HeatmapGenerator)
    stats = Dict{Int, Dict{String, Any}}()
    
    for feature_id in 1:gen.n_features
        feature_counts = gen.feature_counts[feature_id, :]
        feature_scores = gen.feature_scores[feature_id, :]
        
        if any(feature_counts .> 0)
            stats[feature_id] = Dict(
                "total_selections" => sum(feature_counts),
                "selecting_trees" => count(feature_counts .> 0),
                "avg_score" => mean(feature_scores[feature_counts .> 0]),
                "consensus_percentage" => count(feature_counts .> 0) / gen.n_trees * 100
            )
        end
    end
    
    return stats
end

"""
Calculate tree-level statistics
"""
function calculate_tree_statistics(gen::HeatmapGenerator)
    stats = Dict{Int, Dict{String, Any}}()
    
    for tree_id in 1:gen.n_trees
        tree_counts = gen.feature_counts[:, tree_id]
        tree_scores = gen.feature_scores[:, tree_id]
        
        selected_features = findall(tree_counts .> 0)
        
        stats[tree_id] = Dict(
            "n_features_selected" => length(selected_features),
            "total_selections" => sum(tree_counts),
            "avg_score" => isempty(selected_features) ? 0.0 : mean(tree_scores[selected_features]),
            "top_features" => selected_features[sortperm(tree_counts[selected_features], rev=true)][1:min(10, length(selected_features))]
        )
    end
    
    return stats
end

"""
Get features with high consensus
"""
function get_consensus_features(gen::HeatmapGenerator; threshold::Float64 = 0.75)
    consensus_features = Int[]
    
    for feature_id in 1:gen.n_features
        consensus = count(gen.feature_counts[feature_id, :] .> 0) / gen.n_trees
        if consensus >= threshold
            push!(consensus_features, feature_id)
        end
    end
    
    return consensus_features
end

"""
Create animated heatmap showing evolution over time
"""
function create_animated_heatmap(
    gen::HeatmapGenerator,
    snapshots::Vector{Matrix{Int}};
    fps::Int = 5,
    output_path::String = "heatmap_animation.gif"
)
    anim = @animate for (i, snapshot) in enumerate(snapshots)
        heatmap(
            snapshot',
            color = :viridis,
            xlabel = "Feature Index",
            ylabel = "Tree ID",
            title = "Feature Selection Evolution - Step $i",
            clims = (0, maximum(maximum.(snapshots)))
        )
    end
    
    gif(anim, output_path, fps = fps)
end

end # module