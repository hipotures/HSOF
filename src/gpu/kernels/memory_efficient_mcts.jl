module MemoryEfficientMCTS

using CUDA
using Statistics
using Dates
using JSON3

# Import all optimization modules
using ..MCTSTypes
using ..CompressedNodeStorage
using ..SharedFeatureStorage
using ..LazyExpansion

"""
Memory-efficient MCTS engine using compressed nodes, shared feature storage,
and lazy expansion for 50 trees per GPU with minimal memory footprint.
"""

"""
Memory-efficient tree ensemble managing 50 trees per GPU with optimizations:
- Compressed node representation (64 bytes vs 256 bytes)
- Shared feature storage with deduplication
- Lazy child node expansion
- NUMA-aware memory allocation
"""
struct MemoryEfficientTreeEnsemble
    # Core compressed tree storage
    tree::CompressedTreeSoA
    
    # Shared feature storage pool
    feature_pool::SharedFeaturePool
    
    # Lazy expansion manager
    expansion_manager::LazyExpansionManager
    
    # Per-tree configuration
    tree_configs::Vector{TreeConfig}
    
    # Memory statistics
    memory_stats::MemoryStatistics
    
    # Device information
    device::CuDevice
    
    # Ensemble parameters
    max_trees::Int32
    max_nodes_per_tree::Int32
    
    function MemoryEfficientTreeEnsemble(
        device::CuDevice;
        max_trees::Int32 = 50,
        max_nodes_per_tree::Int32 = 20000,
        feature_pool_size::Float32 = 0.8f0
    )
        CUDA.device!(device) do
            # Create compressed tree storage
            tree = CompressedTreeSoA(device, max_trees, max_nodes_per_tree)
            
            # Create shared feature pool
            feature_pool = SharedFeaturePool(device, feature_pool_size)
            
            # Create lazy expansion manager
            expansion_manager = LazyExpansionManager(device, feature_pool, max_trees, max_nodes_per_tree)
            
            # Initialize per-tree configurations
            tree_configs = [TreeConfig(i) for i in 1:max_trees]
            
            # Initialize memory statistics
            memory_stats = MemoryStatistics(device)
            
            new(tree, feature_pool, expansion_manager, tree_configs, memory_stats,
                device, max_trees, max_nodes_per_tree)
        end
    end
end

"""
Configuration for individual trees in the ensemble.
"""
struct TreeConfig
    tree_id::Int32
    exploration_constant::Float32
    feature_subset_ratio::Float32
    random_seed::UInt32
    virtual_loss::Int32
    max_depth::UInt8
    
    function TreeConfig(tree_id::Int32)
        new(
            tree_id,
            0.5f0 + 1.5f0 * rand(Float32),  # Random exploration constant 0.5-2.0
            0.8f0,                          # Use 80% of features
            UInt32(tree_id * 12345),        # Deterministic but different seed
            10,                             # Virtual loss
            UInt8(50)                       # Max depth
        )
    end
end

"""
Memory usage statistics for the ensemble.
"""
mutable struct MemoryStatistics
    # Compressed node storage
    compressed_nodes_mb::Float64
    
    # Shared feature storage
    feature_pool_mb::Float64
    
    # Lazy expansion contexts
    expansion_contexts_mb::Float64
    
    # Total memory usage
    total_mb::Float64
    
    # Memory efficiency metrics
    compression_ratio::Float64
    sharing_efficiency::Float64
    lazy_savings_mb::Float64
    
    # Device memory info
    device_total_mb::Float64
    device_free_mb::Float64
    
    function MemoryStatistics(device::CuDevice)
        CUDA.device!(device) do
            # Get device memory info
            device_total_mb = CUDA.total_memory(device) / (1024.0^2)
            device_free_mb = CUDA.available_memory(device) / (1024.0^2)
            
            new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, device_total_mb, device_free_mb)
        end
    end
end

"""
Initialize the memory-efficient ensemble with root nodes.
"""
function initialize_ensemble!(
    ensemble::MemoryEfficientTreeEnsemble,
    initial_features::Vector{Vector{Int}} = [Int[] for _ in 1:ensemble.max_trees]
)
    CUDA.device!(ensemble.device) do
        # Initialize each tree
        for tree_id in 1:ensemble.max_trees
            init_tree!(ensemble, tree_id, initial_features[tree_id])
        end
        
        # Update memory statistics
        update_memory_statistics!(ensemble)
        
        @info "Memory-efficient ensemble initialized" trees=ensemble.max_trees memory_mb=ensemble.memory_stats.total_mb
    end
end

"""
Initialize a single tree in the ensemble.
"""
function init_tree!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    initial_features::Vector{Int}
)
    # Reset tree state
    ensemble.tree.tree_states[tree_id] = UInt8(1)  # Active
    ensemble.tree.next_free_nodes[tree_id] = UInt32(2)  # Root is node 1
    ensemble.tree.total_nodes[tree_id] = UInt32(1)
    
    # Initialize root node
    root_id = UInt16(1)
    ensemble.tree.root_nodes[tree_id] = root_id
    
    # Create root node
    init_compressed_node!(ensemble.tree, tree_id, root_id, UInt16(0))
    
    # Create initial feature set
    if !isempty(initial_features)
        # Convert feature list to bitfield tuple
        features = ntuple(FEATURE_CHUNKS) do i
            chunk = UInt64(0)
            for feature_idx in initial_features
                if feature_idx > 0 && feature_idx <= MAX_FEATURES
                    chunk_idx = div(feature_idx - 1, 64) + 1
                    bit_idx = mod(feature_idx - 1, 64)
                    
                    if chunk_idx == i
                        chunk |= (UInt64(1) << bit_idx)
                    end
                end
            end
            return chunk
        end
        
        # Store in shared feature pool
        feature_pool_id = store_feature_set!(ensemble.feature_pool, features, UInt16(1))
        ensemble.tree.feature_pool_ids[root_id, tree_id] = feature_pool_id
        
        # Set feature hash in root node
        feature_hash = hash_feature_set(features)
        set_feature_hash!(ensemble.tree, tree_id, root_id, feature_hash)
    else
        # Empty feature set
        empty_features = ntuple(i -> UInt64(0), FEATURE_CHUNKS)
        feature_pool_id = store_feature_set!(ensemble.feature_pool, empty_features, UInt16(1))
        ensemble.tree.feature_pool_ids[root_id, tree_id] = feature_pool_id
    end
    
    # Initialize lazy expansion context
    init_lazy_context!(ensemble.expansion_manager, tree_id, root_id, UInt8(0))
end

"""
Select best node for expansion using UCB1.
"""
function select_node_for_expansion!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32
)
    current_id = ensemble.tree.root_nodes[tree_id]
    config = ensemble.tree_configs[tree_id]
    
    # Traverse tree using UCB1 until leaf
    while true
        current_node = ensemble.tree.nodes[current_id, tree_id]
        
        # Check if this node has children
        if get_child_count(current_node) == 0
            # This is a leaf node
            return current_id
        end
        
        # Select best child using UCB1
        best_child = select_best_child_ucb1!(ensemble, tree_id, current_id, config)
        
        if best_child == UInt16(0)
            # No valid child found
            return current_id
        end
        
        current_id = best_child
    end
end

"""
Select best child using UCB1 with lazy expansion.
"""
function select_best_child_ucb1!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    parent_id::UInt16,
    config::TreeConfig
)
    parent_node = ensemble.tree.nodes[parent_id, tree_id]
    parent_visits = get_visit_count(parent_node)
    
    if parent_visits == 0
        return UInt16(0)  # Cannot select from unvisited node
    end
    
    # Get number of potential children
    context = ensemble.expansion_manager.contexts[parent_id, tree_id]
    num_children = context.num_children
    
    if num_children == 0
        return UInt16(0)  # No children available
    end
    
    best_child = UInt16(0)
    best_score = -Inf32
    
    # Evaluate each potential child
    for child_idx in 1:num_children
        # Get or create child lazily
        child_id = get_or_create_child!(
            ensemble.expansion_manager,
            ensemble.tree,
            tree_id,
            parent_id,
            child_idx
        )
        
        if child_id == UInt16(0)
            continue  # Failed to create child
        end
        
        # Calculate UCB1 score
        child_node = ensemble.tree.nodes[child_id, tree_id]
        child_visits = get_visit_count(child_node)
        
        if child_visits == 0
            # Unvisited child gets highest priority
            return child_id
        end
        
        child_q = get_q_value(child_node)
        child_prior = get_prior(child_node)
        
        ucb1_score = ucb1_score(
            child_q * Float32(child_visits),
            child_visits,
            parent_visits,
            config.exploration_constant,
            child_prior
        )
        
        if ucb1_score > best_score
            best_score = ucb1_score
            best_child = child_id
        end
    end
    
    return best_child
end

"""
Expand a node by creating child descriptors.
"""
function expand_node!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    node_id::UInt16,
    available_actions::Vector{UInt16},
    prior_scores::Vector{Float32}
)
    if length(available_actions) != length(prior_scores)
        @error "Actions and prior scores must have same length"
        return false
    end
    
    num_children = min(length(available_actions), MAX_CHILDREN_PER_NODE)
    
    # Initialize lazy expansion context
    init_lazy_context!(ensemble.expansion_manager, tree_id, node_id, UInt8(num_children))
    
    # Add child descriptors
    for i in 1:num_children
        action_id = available_actions[i]
        prior_score = prior_scores[i]
        
        # Calculate feature hash for this action
        parent_pool_id = ensemble.tree.feature_pool_ids[node_id, tree_id]
        parent_features = get_feature_set(ensemble.feature_pool, parent_pool_id)
        child_features = create_child_features(ensemble.feature_pool, parent_pool_id, action_id)
        feature_hash = hash_feature_set(child_features)
        
        # Add child descriptor
        success = add_child_descriptor!(
            ensemble.expansion_manager,
            tree_id,
            node_id,
            UInt8(i),
            action_id,
            prior_score,
            feature_hash
        )
        
        if !success
            @warn "Failed to add child descriptor" tree_id node_id child_idx=i
        end
    end
    
    return true
end

"""
Simulate (evaluate) a node using metamodel.
"""
function simulate_node!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    node_id::UInt16,
    evaluation_func::Function
)
    # Get node features
    feature_pool_id = ensemble.tree.feature_pool_ids[node_id, tree_id]
    features = get_feature_set(ensemble.feature_pool, feature_pool_id)
    
    # Convert to feature indices for evaluation
    feature_indices = Int[]
    for chunk_idx in 1:FEATURE_CHUNKS
        chunk = features[chunk_idx]
        for bit_idx in 0:63
            if (chunk & (UInt64(1) << bit_idx)) != 0
                feature_idx = (chunk_idx - 1) * 64 + bit_idx + 1
                if feature_idx <= MAX_FEATURES
                    push!(feature_indices, feature_idx)
                end
            end
        end
    end
    
    # Evaluate using provided function
    score = evaluation_func(feature_indices)
    
    return Float32(score)
end

"""
Backpropagate score up the tree.
"""
function backpropagate!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    node_id::UInt16,
    score::Float32
)
    current_id = node_id
    
    while current_id != UInt16(0)
        current_node = ensemble.tree.nodes[current_id, tree_id]
        
        # Update visit count
        current_visits = get_visit_count(current_node)
        set_visit_count!(ensemble.tree, tree_id, current_id, current_visits + 1)
        
        # Update Q-value (incremental average)
        current_q = get_q_value(current_node)
        new_q = current_q + (score - current_q) / Float32(current_visits + 1)
        set_q_value!(ensemble.tree, tree_id, current_id, new_q)
        
        # Move to parent
        parent_id = get_parent_id(current_node)
        current_id = parent_id == UInt16(0) ? UInt16(0) : parent_id
    end
end

"""
Run MCTS simulation for a single tree.
"""
function run_mcts_simulation!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    evaluation_func::Function,
    available_actions::Vector{UInt16},
    prior_scores::Vector{Float32}
)
    # Selection: find leaf node
    leaf_id = select_node_for_expansion!(ensemble, tree_id)
    
    # Expansion: create child descriptors if needed
    leaf_node = ensemble.tree.nodes[leaf_id, tree_id]
    if get_child_count(leaf_node) == 0 && get_visit_count(leaf_node) > 0
        expand_node!(ensemble, tree_id, leaf_id, available_actions, prior_scores)
    end
    
    # Simulation: evaluate node
    score = simulate_node!(ensemble, tree_id, leaf_id, evaluation_func)
    
    # Backpropagation: update ancestors
    backpropagate!(ensemble, tree_id, leaf_id, score)
    
    return score
end

"""
Run ensemble MCTS for specified number of iterations.
"""
function run_ensemble_mcts!(
    ensemble::MemoryEfficientTreeEnsemble,
    iterations::Int,
    evaluation_func::Function,
    available_actions::Vector{UInt16},
    prior_scores::Vector{Float32}
)
    @info "Starting ensemble MCTS" iterations trees=ensemble.max_trees
    
    start_time = time()
    
    for iter in 1:iterations
        # Run simulation on each tree
        for tree_id in 1:ensemble.max_trees
            run_mcts_simulation!(
                ensemble,
                tree_id,
                evaluation_func,
                available_actions,
                prior_scores
            )
        end
        
        # Periodic updates
        if iter % 1000 == 0
            elapsed = time() - start_time
            rate = iter / elapsed
            
            @info "MCTS Progress" iteration=iter rate_per_sec=rate
            
            # Update memory statistics
            update_memory_statistics!(ensemble)
            
            # Trigger GC if needed
            if should_gc(ensemble.feature_pool)
                garbage_collect!(ensemble.feature_pool, ensemble.tree.feature_pool_ids)
            end
        end
    end
    
    @info "Ensemble MCTS completed" iterations total_time=time()-start_time
end

"""
Get best features from ensemble using voting.
"""
function get_best_features_ensemble(
    ensemble::MemoryEfficientTreeEnsemble,
    num_features::Int
)
    feature_votes = Dict{Int, Float32}()
    
    # Collect votes from all trees
    for tree_id in 1:ensemble.max_trees
        tree_features = get_tree_best_features(ensemble, tree_id, num_features * 2)
        
        for (feature_idx, weight) in tree_features
            feature_votes[feature_idx] = get(feature_votes, feature_idx, 0.0f0) + weight
        end
    end
    
    # Sort by vote count and return top features
    sorted_features = sort(collect(feature_votes), by = x -> x[2], rev = true)
    
    return [feat for (feat, _) in sorted_features[1:min(num_features, length(sorted_features))]]
end

"""
Get best features from a single tree.
"""
function get_tree_best_features(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    num_features::Int
)
    feature_weights = Dict{Int, Float32}()
    
    # Traverse tree and weight features by visit counts
    traverse_tree_for_features!(
        ensemble,
        tree_id,
        ensemble.tree.root_nodes[tree_id],
        feature_weights,
        1.0f0
    )
    
    # Sort and return top features
    sorted_features = sort(collect(feature_weights), by = x -> x[2], rev = true)
    
    return sorted_features[1:min(num_features, length(sorted_features))]
end

"""
Recursively traverse tree to collect feature weights.
"""
function traverse_tree_for_features!(
    ensemble::MemoryEfficientTreeEnsemble,
    tree_id::Int32,
    node_id::UInt16,
    feature_weights::Dict{Int, Float32},
    parent_weight::Float32
)
    node = ensemble.tree.nodes[node_id, tree_id]
    visits = get_visit_count(node)
    
    if visits == 0
        return
    end
    
    # Calculate node weight
    node_weight = parent_weight * Float32(visits)
    
    # Get features for this node
    feature_pool_id = ensemble.tree.feature_pool_ids[node_id, tree_id]
    features = get_feature_set(ensemble.feature_pool, feature_pool_id)
    
    # Add features to weight map
    for chunk_idx in 1:FEATURE_CHUNKS
        chunk = features[chunk_idx]
        for bit_idx in 0:63
            if (chunk & (UInt64(1) << bit_idx)) != 0
                feature_idx = (chunk_idx - 1) * 64 + bit_idx + 1
                if feature_idx <= MAX_FEATURES
                    feature_weights[feature_idx] = get(feature_weights, feature_idx, 0.0f0) + node_weight
                end
            end
        end
    end
    
    # Recurse to children
    child_count = get_child_count(node)
    first_child = get_first_child(node)
    
    for i in 1:child_count
        child_id = first_child + i - 1
        if child_id > 0 && child_id <= ensemble.max_nodes_per_tree
            traverse_tree_for_features!(
                ensemble,
                tree_id,
                child_id,
                feature_weights,
                node_weight
            )
        end
    end
end

"""
Update memory usage statistics.
"""
function update_memory_statistics!(ensemble::MemoryEfficientTreeEnsemble)
    # Compressed nodes
    node_size = sizeof(CompressedNode)
    total_nodes = sum(Array(ensemble.tree.total_nodes))
    ensemble.memory_stats.compressed_nodes_mb = 
        Float64(total_nodes * node_size) / (1024.0^2)
    
    # Feature pool
    pool_stats = get_pool_statistics(ensemble.feature_pool)
    ensemble.memory_stats.feature_pool_mb = 
        Float64(pool_stats["total_entries"] * sizeof(FeatureSetEntry)) / (1024.0^2)
    
    # Expansion contexts
    context_size = sizeof(LazyExpansionContext)
    total_contexts = ensemble.max_trees * ensemble.max_nodes_per_tree
    ensemble.memory_stats.expansion_contexts_mb = 
        Float64(total_contexts * context_size) / (1024.0^2)
    
    # Total memory
    ensemble.memory_stats.total_mb = 
        ensemble.memory_stats.compressed_nodes_mb +
        ensemble.memory_stats.feature_pool_mb +
        ensemble.memory_stats.expansion_contexts_mb
    
    # Calculate efficiency metrics
    uncompressed_size = Float64(total_nodes * 256) / (1024.0^2)  # Original node size
    ensemble.memory_stats.compression_ratio = 
        uncompressed_size / ensemble.memory_stats.compressed_nodes_mb
    
    # Lazy expansion savings
    lazy_stats = get_lazy_expansion_stats(ensemble.expansion_manager)
    ensemble.memory_stats.lazy_savings_mb = lazy_stats["memory_saved_mb"]
    
    # Update device memory
    CUDA.device!(ensemble.device) do
        ensemble.memory_stats.device_free_mb = CUDA.available_memory() / (1024.0^2)
    end
end

"""
Get comprehensive ensemble statistics.
"""
function get_ensemble_statistics(ensemble::MemoryEfficientTreeEnsemble)
    update_memory_statistics!(ensemble)
    
    pool_stats = get_pool_statistics(ensemble.feature_pool)
    lazy_stats = get_lazy_expansion_stats(ensemble.expansion_manager)
    
    return Dict{String, Any}(
        "memory_stats" => Dict(
            "compressed_nodes_mb" => ensemble.memory_stats.compressed_nodes_mb,
            "feature_pool_mb" => ensemble.memory_stats.feature_pool_mb,
            "expansion_contexts_mb" => ensemble.memory_stats.expansion_contexts_mb,
            "total_mb" => ensemble.memory_stats.total_mb,
            "compression_ratio" => ensemble.memory_stats.compression_ratio,
            "lazy_savings_mb" => ensemble.memory_stats.lazy_savings_mb,
            "device_free_mb" => ensemble.memory_stats.device_free_mb,
            "device_total_mb" => ensemble.memory_stats.device_total_mb
        ),
        "pool_stats" => pool_stats,
        "lazy_stats" => lazy_stats,
        "tree_stats" => Dict(
            "active_trees" => ensemble.max_trees,
            "total_nodes" => sum(Array(ensemble.tree.total_nodes)),
            "avg_nodes_per_tree" => mean(Array(ensemble.tree.total_nodes))
        )
    )
end

export MemoryEfficientTreeEnsemble, TreeConfig, MemoryStatistics
export initialize_ensemble!, init_tree!, select_node_for_expansion!
export expand_node!, simulate_node!, backpropagate!, run_mcts_simulation!
export run_ensemble_mcts!, get_best_features_ensemble, get_tree_best_features
export update_memory_statistics!, get_ensemble_statistics

end # module