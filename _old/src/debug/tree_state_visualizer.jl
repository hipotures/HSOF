module TreeStateVisualizer

using CUDA
using JSON3
using Dates
using Printf
using Statistics
using Plots
using ColorSchemes

export TreeVisualizer, initialize!, create_tree_visualization, save_visualization

"""
Tree state visualizer for MCTS debugging
"""
mutable struct TreeVisualizer
    output_dir::String
    visualization_cache::Dict{Int, Any}
    max_depth::Int
    node_colors::Dict{String, RGB}
    
    function TreeVisualizer(output_dir::String; max_depth::Int = 10)
        mkpath(output_dir)
        
        # Define color scheme for different node states
        node_colors = Dict{String, RGB}(
            "unexplored" => RGB(0.7, 0.7, 0.7),
            "exploring" => RGB(1.0, 0.8, 0.0),
            "selected" => RGB(0.0, 0.8, 0.0),
            "rejected" => RGB(0.8, 0.0, 0.0),
            "leaf" => RGB(0.0, 0.0, 0.8)
        )
        
        new(output_dir, Dict{Int, Any}(), max_depth, node_colors)
    end
end

"""
Initialize the tree visualizer
"""
function initialize!(viz::TreeVisualizer)
    empty!(viz.visualization_cache)
end

"""
Node structure for visualization
"""
struct TreeNode
    id::Int
    parent_id::Union{Int, Nothing}
    feature_indices::Vector{Int}
    score::Float32
    visit_count::Int
    state::String  # unexplored, exploring, selected, rejected, leaf
    depth::Int
    children::Vector{Int}
end

"""
Create visualization for a tree
"""
function create_tree_visualization(
    viz::TreeVisualizer,
    tree_id::Int,
    tree_data::Any
)
    # Extract tree structure
    nodes = extract_tree_nodes(tree_data)
    
    # Create plot
    plt = plot(
        size = (1200, 800),
        title = "MCTS Tree $tree_id - Feature Selection State",
        xlabel = "Tree Width",
        ylabel = "Tree Depth",
        legend = :topright,
        background_color = :white
    )
    
    # Plot tree structure
    plot_tree_structure!(plt, nodes, viz.node_colors, viz.max_depth)
    
    # Add statistics panel
    add_statistics_panel!(plt, nodes, tree_id)
    
    # Cache visualization
    viz.visualization_cache[tree_id] = plt
    
    return plt
end

"""
Extract tree nodes from raw tree data
"""
function extract_tree_nodes(tree_data::Any)::Vector{TreeNode}
    nodes = TreeNode[]
    
    # Convert tree data to TreeNode objects
    # This assumes tree_data has a specific structure from the MCTS implementation
    function traverse_tree(node_data, parent_id, depth)
        node_id = length(nodes) + 1
        
        # Extract node information
        feature_indices = get(node_data, :feature_indices, Int[])
        score = Float32(get(node_data, :score, 0.0))
        visit_count = get(node_data, :visit_count, 0)
        state = get(node_data, :state, "unexplored")
        children_data = get(node_data, :children, [])
        
        # Create node
        node = TreeNode(
            node_id,
            parent_id,
            feature_indices,
            score,
            visit_count,
            state,
            depth,
            Int[]
        )
        
        push!(nodes, node)
        
        # Process children
        for child_data in children_data
            child_id = traverse_tree(child_data, node_id, depth + 1)
            push!(node.children, child_id)
        end
        
        return node_id
    end
    
    # Start traversal from root
    if !isnothing(tree_data)
        traverse_tree(tree_data, nothing, 0)
    end
    
    return nodes
end

"""
Plot tree structure with node states
"""
function plot_tree_structure!(
    plt::Plots.Plot,
    nodes::Vector{TreeNode},
    node_colors::Dict{String, RGB},
    max_depth::Int
)
    if isempty(nodes)
        return
    end
    
    # Calculate node positions
    positions = calculate_node_positions(nodes, max_depth)
    
    # Draw edges first
    for node in nodes
        if !isnothing(node.parent_id)
            parent_pos = positions[node.parent_id]
            node_pos = positions[node.id]
            
            plot!(plt, 
                [parent_pos[1], node_pos[1]], 
                [parent_pos[2], node_pos[2]],
                color = :gray,
                alpha = 0.5,
                linewidth = 1
            )
        end
    end
    
    # Draw nodes
    for node in nodes
        pos = positions[node.id]
        color = node_colors[node.state]
        
        # Node size based on visit count
        size = 5 + min(15, node.visit_count / 10)
        
        scatter!(plt,
            [pos[1]], [pos[2]],
            color = color,
            markersize = size,
            markerstrokewidth = 2,
            markerstrokecolor = :black,
            alpha = 0.8,
            label = ""
        )
        
        # Add feature count annotation for selected nodes
        if node.state == "selected" && !isempty(node.feature_indices)
            annotate!(plt,
                pos[1], pos[2] + 0.1,
                text("$(length(node.feature_indices))", 8, :center)
            )
        end
    end
    
    # Add legend
    for (state, color) in node_colors
        scatter!(plt,
            [], [],
            color = color,
            markersize = 10,
            label = state
        )
    end
end

"""
Calculate node positions for visualization
"""
function calculate_node_positions(nodes::Vector{TreeNode}, max_depth::Int)
    positions = Dict{Int, Tuple{Float64, Float64}}()
    
    # Group nodes by depth
    depth_groups = Dict{Int, Vector{TreeNode}}()
    for node in nodes
        depth = node.depth
        if !haskey(depth_groups, depth)
            depth_groups[depth] = TreeNode[]
        end
        push!(depth_groups[depth], node)
    end
    
    # Calculate positions
    for depth in 0:max_depth
        if haskey(depth_groups, depth)
            nodes_at_depth = depth_groups[depth]
            n_nodes = length(nodes_at_depth)
            
            for (i, node) in enumerate(nodes_at_depth)
                x = (i - 0.5) / n_nodes * 10 - 5  # Spread from -5 to 5
                y = -depth  # Depth increases downward
                positions[node.id] = (x, y)
            end
        end
    end
    
    return positions
end

"""
Add statistics panel to visualization
"""
function add_statistics_panel!(plt::Plots.Plot, nodes::Vector{TreeNode}, tree_id::Int)
    # Calculate statistics
    total_nodes = length(nodes)
    explored_nodes = count(n -> n.state != "unexplored", nodes)
    selected_features = sum(length(n.feature_indices) for n in nodes if n.state == "selected")
    max_depth_reached = maximum(n.depth for n in nodes; init=0)
    avg_score = mean(n.score for n in nodes if n.score > 0; init=0.0)
    
    # Create statistics text
    stats_text = """
    Tree $tree_id Statistics:
    Total Nodes: $total_nodes
    Explored: $explored_nodes ($(round(explored_nodes/total_nodes*100, digits=1))%)
    Selected Features: $selected_features
    Max Depth: $max_depth_reached
    Avg Score: $(round(avg_score, digits=4))
    """
    
    # Add text annotation
    annotate!(plt,
        4.5, 0.5,
        text(stats_text, 10, :left, :top),
        box = true,
        boxcolor = :lightgray,
        boxalpha = 0.8
    )
end

"""
Save visualization to file
"""
function save_visualization(viz::Plots.Plot, save_path::String)
    savefig(viz, save_path)
end

"""
Create animated visualization showing tree evolution
"""
function create_tree_animation(
    viz::TreeVisualizer,
    tree_id::Int,
    tree_snapshots::Vector{Any};
    fps::Int = 10
)
    anim = @animate for (i, snapshot) in enumerate(tree_snapshots)
        plt = create_tree_visualization(viz, tree_id, snapshot)
        title!(plt, "MCTS Tree $tree_id - Iteration $i")
    end
    
    gif_path = joinpath(viz.output_dir, "tree_$(tree_id)_evolution.gif")
    gif(anim, gif_path, fps = fps)
    
    return gif_path
end

"""
Create feature selection path visualization
"""
function visualize_selection_path(
    viz::TreeVisualizer,
    tree_id::Int,
    selection_path::Vector{Int},
    tree_data::Any
)
    nodes = extract_tree_nodes(tree_data)
    
    plt = plot(
        size = (1000, 600),
        title = "Feature Selection Path - Tree $tree_id",
        xlabel = "Selection Step",
        ylabel = "Feature Index",
        legend = :topright,
        background_color = :white
    )
    
    # Plot selection path
    plot!(plt,
        1:length(selection_path),
        selection_path,
        marker = :circle,
        markersize = 8,
        linewidth = 2,
        label = "Selection Order",
        color = :blue
    )
    
    # Highlight important features
    for (i, feature) in enumerate(selection_path)
        annotate!(plt,
            i, feature + 2,
            text("F$feature", 8, :center)
        )
    end
    
    return plt
end

end # module