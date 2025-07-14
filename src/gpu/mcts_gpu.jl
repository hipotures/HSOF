module MCTSGPU

using CUDA
using Statistics
using Dates

# Include kernel modules
include("kernels/mcts_types.jl")
include("kernels/memory_pool.jl")
include("kernels/persistent_kernel.jl")

using .MCTSTypes
using .MemoryPool
using .PersistentKernel

export MCTSGPUEngine, initialize!, start!, stop!, get_statistics
export select_features, get_best_features, reset_tree!

"""
Main MCTS GPU Engine managing the persistent kernel and tree operations
"""
mutable struct MCTSGPUEngine
    # Core components
    tree::MCTSTreeSoA
    config::PersistentKernelConfig
    memory_manager::MemoryPoolManager
    
    # Kernel management
    kernel_task::Union{Nothing, Task}
    kernel_state::Union{Nothing, KernelState}
    work_queue::Union{Nothing, WorkQueue}
    
    # Statistics
    stats::TreeStatistics
    start_time::DateTime
    
    # Device
    device::CuDevice
    
    # Status
    is_running::Bool
    
    function MCTSGPUEngine(;
        device::CuDevice = CUDA.device(),
        block_size = 256,
        grid_size = 108,
        max_iterations = 1_000_000,
        exploration_constant = 1.414f0,
        defrag_threshold = 0.5f0
    )
        # Create tree on specified device
        tree = MCTSTreeSoA(device)
        
        # Create configuration
        config = PersistentKernelConfig(
            block_size = block_size,
            grid_size = grid_size,
            max_iterations = max_iterations,
            exploration_constant = exploration_constant
        )
        
        # Create memory manager
        memory_manager = MemoryPoolManager(tree, defrag_threshold = defrag_threshold)
        
        # Initialize statistics
        stats = TreeStatistics()
        
        new(tree, config, memory_manager,
            nothing, nothing, nothing,
            stats, now(), device, false)
    end
end

"""
Initialize the MCTS engine with root node
"""
function initialize!(engine::MCTSGPUEngine, initial_features::Vector{Int} = Int[])
    CUDA.device!(engine.device) do
        # Reset tree to initial state
        fill!(engine.tree.next_free_node, 1)
        fill!(engine.tree.free_list_size, 0)
        fill!(engine.tree.total_nodes, 0)
        fill!(engine.tree.max_depth, 0)
        
        # Initialize root node directly
        CUDA.@allowscalar begin
            engine.tree.node_ids[1] = 1
            engine.tree.node_states[1] = NODE_ACTIVE
            engine.tree.parent_ids[1] = -1
            engine.tree.visit_counts[1] = 1
            engine.tree.total_scores[1] = 0.0f0
            engine.tree.prior_scores[1] = 0.0f0
            engine.tree.first_child_idx[1] = -1
            engine.tree.num_children[1] = 0
            engine.tree.total_nodes[1] = 1
            engine.tree.next_free_node[1] = 2  # Next available is 2
            
            # Clear feature mask for root
            for i in 1:FEATURE_CHUNKS
                engine.tree.feature_masks[i, 1] = UInt64(0)
            end
            
            # Set initial features
            for feature_idx in initial_features
                if 1 <= feature_idx <= MAX_FEATURES
                    chunk_idx = div(feature_idx - 1, 64) + 1
                    bit_idx = mod(feature_idx - 1, 64)
                    engine.tree.feature_masks[chunk_idx, 1] |= (UInt64(1) << bit_idx)
                end
            end
        end
        
        # Reset statistics
        engine.stats = TreeStatistics()
        engine.start_time = now()
    end
    
    return nothing
end

"""
Start the persistent MCTS kernel
"""
function start!(engine::MCTSGPUEngine)
    if engine.is_running
        @warn "Engine is already running"
        return
    end
    
    # Launch kernel asynchronously
    engine.kernel_task = @async begin
        CUDA.device!(engine.device) do
            kernel, state, queue = launch_persistent_kernel!(
                engine.tree, 
                engine.config,
                device = engine.device
            )
            
            engine.kernel_state = state
            engine.work_queue = queue
            
            # Wait for kernel completion
            CUDA.synchronize()
        end
    end
    
    engine.is_running = true
    
    return nothing
end

"""
Stop the persistent MCTS kernel
"""
function stop!(engine::MCTSGPUEngine)
    if !engine.is_running
        @warn "Engine is not running"
        return
    end
    
    # Signal kernel to stop
    if !isnothing(engine.kernel_state)
        stop_kernel!(engine.kernel_state)
    end
    
    # Wait for kernel task to complete
    if !isnothing(engine.kernel_task)
        wait(engine.kernel_task)
    end
    
    engine.is_running = false
    
    # Collect final statistics
    update_statistics!(engine)
    
    return nothing
end

"""
Select features using MCTS exploration
"""
function select_features(
    engine::MCTSGPUEngine,
    num_features::Int,
    iterations::Int;
    reset_tree::Bool = true
)
    if reset_tree
        initialize!(engine)
    end
    
    # Update max iterations
    engine.config = PersistentKernelConfig(
        block_size = engine.config.block_size,
        grid_size = engine.config.grid_size,
        max_iterations = iterations,
        exploration_constant = engine.config.exploration_constant
    )
    
    # Start kernel
    start!(engine)
    
    # Monitor progress
    last_update = time()
    update_interval = 1.0  # seconds
    
    while engine.is_running
        current_time = time()
        if current_time - last_update > update_interval
            update_statistics!(engine)
            progress = get_progress(engine)
            
            @info "MCTS Progress" progress iterations stats=engine.stats
            
            last_update = current_time
        end
        
        sleep(0.1)
    end
    
    # Get best features
    best_features = get_best_features(engine, num_features)
    
    return best_features
end

"""
Get the best features based on visit counts
"""
function get_best_features(engine::MCTSGPUEngine, num_features::Int)
    # Download tree data
    visit_counts = Array(engine.tree.visit_counts)
    node_states = Array(engine.tree.node_states)
    feature_masks = Array(engine.tree.feature_masks)
    
    # Find all leaf nodes with high visit counts
    leaf_nodes = findall(i -> node_states[i] == NODE_ACTIVE && visit_counts[i] > 0, 1:MAX_NODES)
    
    if isempty(leaf_nodes)
        @warn "No valid leaf nodes found"
        return Int[]
    end
    
    # Sort by visit count
    sorted_indices = sort(leaf_nodes, by = i -> visit_counts[i], rev = true)
    
    # Collect unique features from top nodes
    selected_features = Set{Int}()
    
    for node_idx in sorted_indices
        # Get features for this node
        for feature_idx in 1:MAX_FEATURES
            if has_feature(feature_masks, node_idx, feature_idx)
                push!(selected_features, feature_idx)
            end
        end
        
        if length(selected_features) >= num_features
            break
        end
    end
    
    # Convert to sorted array
    features = sort(collect(selected_features))[1:min(num_features, length(selected_features))]
    
    return features
end

"""
Update engine statistics
"""
function update_statistics!(engine::MCTSGPUEngine)
    # Download statistics from GPU
    CUDA.device!(engine.device) do
        CUDA.@allowscalar begin
            total_nodes = engine.tree.total_nodes[1]
            max_depth = engine.tree.max_depth[1]
            
            engine.stats.nodes_allocated = total_nodes
            engine.stats.max_depth_reached = max_depth
            
            # Calculate rates if kernel is running
            if !isnothing(engine.kernel_state)
                iterations = engine.kernel_state.iteration[1]
                
                elapsed = (now() - engine.start_time).value / 1000.0  # seconds
                if elapsed > 0 && iterations > 0
                    engine.stats.selection_time_ms = elapsed * 1000.0 / iterations
                    
                    # Estimate GPU utilization (simplified)
                    theoretical_ops = engine.config.block_size * engine.config.grid_size * iterations
                    actual_ops = total_nodes * 10  # Rough estimate
                    engine.stats.gpu_utilization = min(100.0, actual_ops / theoretical_ops * 100.0)
                end
            end
        end
        
        # Check if defragmentation needed
        if should_defragment(engine.memory_manager)
            @info "Defragmenting memory pool..."
            new_total = defragment!(engine.memory_manager)
            engine.stats.nodes_recycled = total_nodes - new_total
        end
    end
end

"""
Get current progress
"""
function get_progress(engine::MCTSGPUEngine)
    if isnothing(engine.kernel_state)
        return 0.0
    end
    
    current = Array(engine.kernel_state.iteration)[1]
    total = engine.config.max_iterations
    
    return current / total
end

"""
Reset the tree to initial state
"""
function reset_tree!(engine::MCTSGPUEngine)
    if engine.is_running
        stop!(engine)
    end
    
    initialize!(engine)
end

"""
Get detailed statistics
"""
function get_statistics(engine::MCTSGPUEngine)
    # Update statistics first
    update_statistics!(engine)
    
    stats_dict = Dict{String, Any}(
        "nodes_allocated" => engine.stats.nodes_allocated,
        "nodes_recycled" => engine.stats.nodes_recycled,
        "max_depth" => engine.stats.max_depth_reached,
        "gpu_utilization" => engine.stats.gpu_utilization,
        "elapsed_time" => (now() - engine.start_time).value / 1000.0,
        "device" => CUDA.name(engine.device),
        "memory_used_mb" => engine.stats.nodes_allocated * sizeof(Int32) * 10 / 1024^2  # Rough estimate
    )
    
    if !isnothing(engine.kernel_state)
        CUDA.@allowscalar stats_dict["iterations"] = engine.kernel_state.iteration[1]
    end
    
    return stats_dict
end

end # module