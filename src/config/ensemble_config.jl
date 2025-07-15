module EnsembleConfig

using JSON3
using YAML
using ArgParse

"""
Configuration schema for ensemble parameters with validation and defaults.
Supports JSON/YAML configuration files and command-line overrides.
"""

# Configuration parameter validation functions
function validate_positive_integer(value::Any, name::String)
    if !isa(value, Integer) || value <= 0
        throw(ArgumentError("$name must be a positive integer, got: $value"))
    end
    return Int(value)
end

function validate_positive_float(value::Any, name::String)
    if !isa(value, Real) || value <= 0
        throw(ArgumentError("$name must be a positive number, got: $value"))
    end
    return Float64(value)
end

function validate_float_range(value::Any, name::String, min_val::Float64, max_val::Float64)
    if !isa(value, Real) || value < min_val || value > max_val
        throw(ArgumentError("$name must be between $min_val and $max_val, got: $value"))
    end
    return Float64(value)
end

function validate_string_options(value::Any, name::String, options::Vector{String})
    if !isa(value, String) || !(value in options)
        throw(ArgumentError("$name must be one of $options, got: $value"))
    end
    return String(value)
end

function validate_file_path(value::Any, name::String)
    if !isa(value, String)
        throw(ArgumentError("$name must be a string path, got: $value"))
    end
    return String(value)
end

"""
Ensemble configuration structure with parameter validation.
"""
Base.@kwdef struct EnsembleConfiguration
    # Core ensemble parameters
    num_trees::Int = 100
    trees_per_gpu::Int = 50
    max_nodes_per_tree::Int = 20000
    max_depth::Int = 50
    
    # MCTS parameters
    exploration_constant_min::Float64 = 0.5
    exploration_constant_max::Float64 = 2.0
    virtual_loss::Int = 10
    max_iterations::Int = 1000000
    
    # Feature selection parameters
    initial_features::Int = 500
    target_features::Int = 50
    feature_subset_ratio::Float64 = 0.8
    
    # Diversity parameters
    diversity_threshold::Float64 = 0.7
    random_seed_base::Int = 12345
    
    # Memory management
    memory_pool_size::Float64 = 0.8
    gc_threshold::Float64 = 0.75
    defrag_threshold::Float64 = 0.5
    
    # Performance parameters
    batch_size::Int = 1024
    update_interval_ms::Int = 100
    sync_interval_iterations::Int = 1000
    
    # GPU configuration
    gpu_devices::Vector{Int} = [0, 1]
    memory_limit_gb::Float64 = 22.0  # RTX 4090 has 24GB
    
    # Convergence detection
    convergence_window::Int = 100
    convergence_threshold::Float64 = 0.01
    min_iterations::Int = 10000
    
    # File paths
    model_path::String = "models/metamodel.jl"
    data_path::String = "data/features.sqlite"
    output_path::String = "results/ensemble_results.json"
    log_path::String = "logs/ensemble.log"
    
    # Advanced parameters
    lazy_expansion::Bool = true
    shared_features::Bool = true
    compressed_nodes::Bool = true
    fault_tolerance::Bool = true
    
    # Monitoring
    enable_profiling::Bool = true
    enable_dashboard::Bool = true
    dashboard_refresh_ms::Int = 100
    
    # Performance targets
    target_gpu_utilization::Float64 = 0.85
    target_scaling_efficiency::Float64 = 0.85
    
    function EnsembleConfiguration(
        num_trees::Int,
        trees_per_gpu::Int,
        max_nodes_per_tree::Int,
        max_depth::Int,
        exploration_constant_min::Float64,
        exploration_constant_max::Float64,
        virtual_loss::Int,
        max_iterations::Int,
        initial_features::Int,
        target_features::Int,
        feature_subset_ratio::Float64,
        diversity_threshold::Float64,
        random_seed_base::Int,
        memory_pool_size::Float64,
        gc_threshold::Float64,
        defrag_threshold::Float64,
        batch_size::Int,
        update_interval_ms::Int,
        sync_interval_iterations::Int,
        gpu_devices::Vector{Int},
        memory_limit_gb::Float64,
        convergence_window::Int,
        convergence_threshold::Float64,
        min_iterations::Int,
        model_path::String,
        data_path::String,
        output_path::String,
        log_path::String,
        lazy_expansion::Bool,
        shared_features::Bool,
        compressed_nodes::Bool,
        fault_tolerance::Bool,
        enable_profiling::Bool,
        enable_dashboard::Bool,
        dashboard_refresh_ms::Int,
        target_gpu_utilization::Float64,
        target_scaling_efficiency::Float64
    )
        # Validate all parameters
        validate_positive_integer(num_trees, "num_trees")
        validate_positive_integer(trees_per_gpu, "trees_per_gpu")
        validate_positive_integer(max_nodes_per_tree, "max_nodes_per_tree")
        validate_positive_integer(max_depth, "max_depth")
        
        validate_float_range(exploration_constant_min, "exploration_constant_min", 0.1, 5.0)
        validate_float_range(exploration_constant_max, "exploration_constant_max", 0.1, 5.0)
        
        if exploration_constant_min >= exploration_constant_max
            throw(ArgumentError("exploration_constant_min must be less than exploration_constant_max"))
        end
        
        validate_positive_integer(virtual_loss, "virtual_loss")
        validate_positive_integer(max_iterations, "max_iterations")
        
        validate_positive_integer(initial_features, "initial_features")
        validate_positive_integer(target_features, "target_features")
        
        if target_features >= initial_features
            throw(ArgumentError("target_features must be less than initial_features"))
        end
        
        validate_float_range(feature_subset_ratio, "feature_subset_ratio", 0.1, 1.0)
        validate_float_range(diversity_threshold, "diversity_threshold", 0.1, 1.0)
        
        validate_positive_integer(random_seed_base, "random_seed_base")
        
        validate_float_range(memory_pool_size, "memory_pool_size", 0.1, 1.0)
        validate_float_range(gc_threshold, "gc_threshold", 0.1, 1.0)
        validate_float_range(defrag_threshold, "defrag_threshold", 0.1, 1.0)
        
        validate_positive_integer(batch_size, "batch_size")
        validate_positive_integer(update_interval_ms, "update_interval_ms")
        validate_positive_integer(sync_interval_iterations, "sync_interval_iterations")
        
        if length(gpu_devices) == 0
            throw(ArgumentError("gpu_devices cannot be empty"))
        end
        
        for gpu_id in gpu_devices
            if gpu_id < 0
                throw(ArgumentError("GPU device IDs must be non-negative"))
            end
        end
        
        validate_positive_float(memory_limit_gb, "memory_limit_gb")
        
        validate_positive_integer(convergence_window, "convergence_window")
        validate_float_range(convergence_threshold, "convergence_threshold", 0.001, 1.0)
        validate_positive_integer(min_iterations, "min_iterations")
        
        validate_file_path(model_path, "model_path")
        validate_file_path(data_path, "data_path")
        validate_file_path(output_path, "output_path")
        validate_file_path(log_path, "log_path")
        
        validate_positive_integer(dashboard_refresh_ms, "dashboard_refresh_ms")
        
        validate_float_range(target_gpu_utilization, "target_gpu_utilization", 0.1, 1.0)
        validate_float_range(target_scaling_efficiency, "target_scaling_efficiency", 0.1, 1.0)
        
        # Check logical constraints
        if num_trees % length(gpu_devices) != 0
            @warn "num_trees ($num_trees) is not evenly divisible by number of GPUs ($(length(gpu_devices)))"
        end
        
        expected_trees_per_gpu = div(num_trees, length(gpu_devices))
        if trees_per_gpu != expected_trees_per_gpu
            @warn "trees_per_gpu ($trees_per_gpu) doesn't match expected value ($expected_trees_per_gpu)"
        end
        
        if min_iterations >= max_iterations
            throw(ArgumentError("min_iterations must be less than max_iterations"))
        end
        
        new(
            num_trees, trees_per_gpu, max_nodes_per_tree, max_depth,
            exploration_constant_min, exploration_constant_max, virtual_loss, max_iterations,
            initial_features, target_features, feature_subset_ratio, diversity_threshold, random_seed_base,
            memory_pool_size, gc_threshold, defrag_threshold,
            batch_size, update_interval_ms, sync_interval_iterations,
            gpu_devices, memory_limit_gb,
            convergence_window, convergence_threshold, min_iterations,
            model_path, data_path, output_path, log_path,
            lazy_expansion, shared_features, compressed_nodes, fault_tolerance,
            enable_profiling, enable_dashboard, dashboard_refresh_ms,
            target_gpu_utilization, target_scaling_efficiency
        )
    end
end

"""
Load configuration from JSON file.
"""
function load_json_config(file_path::String)
    if !isfile(file_path)
        throw(ArgumentError("Configuration file not found: $file_path"))
    end
    
    try
        json_data = JSON3.read(file_path)
        return dict_to_config(json_data)
    catch e
        throw(ArgumentError("Failed to parse JSON configuration: $e"))
    end
end

"""
Load configuration from YAML file.
"""
function load_yaml_config(file_path::String)
    if !isfile(file_path)
        throw(ArgumentError("Configuration file not found: $file_path"))
    end
    
    try
        yaml_data = YAML.load_file(file_path)
        return dict_to_config(yaml_data)
    catch e
        throw(ArgumentError("Failed to parse YAML configuration: $e"))
    end
end

"""
Convert dictionary to EnsembleConfiguration with validation.
"""
function dict_to_config(data::Dict)
    # Extract parameters with defaults
    config_params = Dict()
    
    # Core parameters
    config_params[:num_trees] = get(data, "num_trees", 100)
    config_params[:trees_per_gpu] = get(data, "trees_per_gpu", 50)
    config_params[:max_nodes_per_tree] = get(data, "max_nodes_per_tree", 20000)
    config_params[:max_depth] = get(data, "max_depth", 50)
    
    # MCTS parameters
    config_params[:exploration_constant_min] = get(data, "exploration_constant_min", 0.5)
    config_params[:exploration_constant_max] = get(data, "exploration_constant_max", 2.0)
    config_params[:virtual_loss] = get(data, "virtual_loss", 10)
    config_params[:max_iterations] = get(data, "max_iterations", 1000000)
    
    # Feature selection
    config_params[:initial_features] = get(data, "initial_features", 500)
    config_params[:target_features] = get(data, "target_features", 50)
    config_params[:feature_subset_ratio] = get(data, "feature_subset_ratio", 0.8)
    
    # Diversity
    config_params[:diversity_threshold] = get(data, "diversity_threshold", 0.7)
    config_params[:random_seed_base] = get(data, "random_seed_base", 12345)
    
    # Memory management
    config_params[:memory_pool_size] = get(data, "memory_pool_size", 0.8)
    config_params[:gc_threshold] = get(data, "gc_threshold", 0.75)
    config_params[:defrag_threshold] = get(data, "defrag_threshold", 0.5)
    
    # Performance
    config_params[:batch_size] = get(data, "batch_size", 1024)
    config_params[:update_interval_ms] = get(data, "update_interval_ms", 100)
    config_params[:sync_interval_iterations] = get(data, "sync_interval_iterations", 1000)
    
    # GPU configuration
    config_params[:gpu_devices] = get(data, "gpu_devices", [0, 1])
    config_params[:memory_limit_gb] = get(data, "memory_limit_gb", 22.0)
    
    # Convergence
    config_params[:convergence_window] = get(data, "convergence_window", 100)
    config_params[:convergence_threshold] = get(data, "convergence_threshold", 0.01)
    config_params[:min_iterations] = get(data, "min_iterations", 10000)
    
    # File paths
    config_params[:model_path] = get(data, "model_path", "models/metamodel.jl")
    config_params[:data_path] = get(data, "data_path", "data/features.sqlite")
    config_params[:output_path] = get(data, "output_path", "results/ensemble_results.json")
    config_params[:log_path] = get(data, "log_path", "logs/ensemble.log")
    
    # Advanced features
    config_params[:lazy_expansion] = get(data, "lazy_expansion", true)
    config_params[:shared_features] = get(data, "shared_features", true)
    config_params[:compressed_nodes] = get(data, "compressed_nodes", true)
    config_params[:fault_tolerance] = get(data, "fault_tolerance", true)
    
    # Monitoring
    config_params[:enable_profiling] = get(data, "enable_profiling", true)
    config_params[:enable_dashboard] = get(data, "enable_dashboard", true)
    config_params[:dashboard_refresh_ms] = get(data, "dashboard_refresh_ms", 100)
    
    # Performance targets
    config_params[:target_gpu_utilization] = get(data, "target_gpu_utilization", 0.85)
    config_params[:target_scaling_efficiency] = get(data, "target_scaling_efficiency", 0.85)
    
    return EnsembleConfiguration(;config_params...)
end

"""
Convert EnsembleConfiguration to dictionary.
"""
function config_to_dict(config::EnsembleConfiguration)
    return Dict(
        "num_trees" => config.num_trees,
        "trees_per_gpu" => config.trees_per_gpu,
        "max_nodes_per_tree" => config.max_nodes_per_tree,
        "max_depth" => config.max_depth,
        "exploration_constant_min" => config.exploration_constant_min,
        "exploration_constant_max" => config.exploration_constant_max,
        "virtual_loss" => config.virtual_loss,
        "max_iterations" => config.max_iterations,
        "initial_features" => config.initial_features,
        "target_features" => config.target_features,
        "feature_subset_ratio" => config.feature_subset_ratio,
        "diversity_threshold" => config.diversity_threshold,
        "random_seed_base" => config.random_seed_base,
        "memory_pool_size" => config.memory_pool_size,
        "gc_threshold" => config.gc_threshold,
        "defrag_threshold" => config.defrag_threshold,
        "batch_size" => config.batch_size,
        "update_interval_ms" => config.update_interval_ms,
        "sync_interval_iterations" => config.sync_interval_iterations,
        "gpu_devices" => config.gpu_devices,
        "memory_limit_gb" => config.memory_limit_gb,
        "convergence_window" => config.convergence_window,
        "convergence_threshold" => config.convergence_threshold,
        "min_iterations" => config.min_iterations,
        "model_path" => config.model_path,
        "data_path" => config.data_path,
        "output_path" => config.output_path,
        "log_path" => config.log_path,
        "lazy_expansion" => config.lazy_expansion,
        "shared_features" => config.shared_features,
        "compressed_nodes" => config.compressed_nodes,
        "fault_tolerance" => config.fault_tolerance,
        "enable_profiling" => config.enable_profiling,
        "enable_dashboard" => config.enable_dashboard,
        "dashboard_refresh_ms" => config.dashboard_refresh_ms,
        "target_gpu_utilization" => config.target_gpu_utilization,
        "target_scaling_efficiency" => config.target_scaling_efficiency
    )
end

"""
Save configuration to JSON file.
"""
function save_json_config(config::EnsembleConfiguration, file_path::String)
    # Ensure directory exists
    dir = dirname(file_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    config_dict = config_to_dict(config)
    
    open(file_path, "w") do io
        JSON3.pretty(io, config_dict, 2)
    end
    
    @info "Configuration saved to $file_path"
end

"""
Save configuration to YAML file.
"""
function save_yaml_config(config::EnsembleConfiguration, file_path::String)
    # Ensure directory exists
    dir = dirname(file_path)
    if !isempty(dir) && !isdir(dir)
        mkpath(dir)
    end
    
    config_dict = config_to_dict(config)
    
    YAML.write_file(file_path, config_dict)
    
    @info "Configuration saved to $file_path"
end

"""
Create argument parser for command-line overrides.
"""
function create_argument_parser()
    s = ArgParseSettings(
        description = "Ensemble MCTS Configuration",
        version = "1.0.0",
        add_version = true
    )
    
    @add_arg_table! s begin
        "--config", "-c"
            help = "Configuration file path (JSON or YAML)"
            default = "config/ensemble.json"
        
        "--num-trees"
            help = "Number of trees in ensemble"
            arg_type = Int
        
        "--trees-per-gpu"
            help = "Number of trees per GPU"
            arg_type = Int
        
        "--max-iterations"
            help = "Maximum MCTS iterations"
            arg_type = Int
        
        "--initial-features"
            help = "Initial number of features"
            arg_type = Int
        
        "--target-features"
            help = "Target number of features"
            arg_type = Int
        
        "--exploration-constant-min"
            help = "Minimum exploration constant"
            arg_type = Float64
        
        "--exploration-constant-max"
            help = "Maximum exploration constant"
            arg_type = Float64
        
        "--gpu-devices"
            help = "GPU device IDs (comma-separated)"
            default = "0,1"
        
        "--memory-limit-gb"
            help = "Memory limit per GPU in GB"
            arg_type = Float64
        
        "--data-path"
            help = "Path to data file"
        
        "--output-path"
            help = "Path to output results"
        
        "--log-path"
            help = "Path to log file"
        
        "--enable-profiling"
            help = "Enable performance profiling"
            action = :store_true
        
        "--disable-profiling"
            help = "Disable performance profiling"
            action = :store_true
        
        "--enable-dashboard"
            help = "Enable dashboard"
            action = :store_true
        
        "--disable-dashboard"
            help = "Disable dashboard"
            action = :store_true
        
        "--lazy-expansion"
            help = "Enable lazy expansion"
            action = :store_true
        
        "--no-lazy-expansion"
            help = "Disable lazy expansion"
            action = :store_true
        
        "--shared-features"
            help = "Enable shared feature storage"
            action = :store_true
        
        "--no-shared-features"
            help = "Disable shared feature storage"
            action = :store_true
        
        "--compressed-nodes"
            help = "Enable compressed node storage"
            action = :store_true
        
        "--no-compressed-nodes"
            help = "Disable compressed node storage"
            action = :store_true
        
        "--fault-tolerance"
            help = "Enable fault tolerance"
            action = :store_true
        
        "--no-fault-tolerance"
            help = "Disable fault tolerance"
            action = :store_true
        
        "--save-config"
            help = "Save final configuration to file"
    end
    
    return s
end

"""
Parse command line arguments and apply overrides to configuration.
"""
function parse_args_and_load_config(args::Vector{String} = ARGS)
    parser = create_argument_parser()
    parsed_args = parse_args(args, parser)
    
    # Load base configuration
    config_file = parsed_args["config"]
    
    if isfile(config_file)
        if endswith(config_file, ".json")
            config = load_json_config(config_file)
        elseif endswith(config_file, ".yaml") || endswith(config_file, ".yml")
            config = load_yaml_config(config_file)
        else
            throw(ArgumentError("Unsupported configuration file format: $config_file"))
        end
        @info "Loaded configuration from $config_file"
    else
        @info "Configuration file not found, using defaults"
        config = EnsembleConfiguration()
    end
    
    # Apply command-line overrides
    config_dict = config_to_dict(config)
    
    # Override with command-line arguments
    if parsed_args["num-trees"] !== nothing
        config_dict["num_trees"] = parsed_args["num-trees"]
    end
    
    if parsed_args["trees-per-gpu"] !== nothing
        config_dict["trees_per_gpu"] = parsed_args["trees-per-gpu"]
    end
    
    if parsed_args["max-iterations"] !== nothing
        config_dict["max_iterations"] = parsed_args["max-iterations"]
    end
    
    if parsed_args["initial-features"] !== nothing
        config_dict["initial_features"] = parsed_args["initial-features"]
    end
    
    if parsed_args["target-features"] !== nothing
        config_dict["target_features"] = parsed_args["target-features"]
    end
    
    if parsed_args["exploration-constant-min"] !== nothing
        config_dict["exploration_constant_min"] = parsed_args["exploration-constant-min"]
    end
    
    if parsed_args["exploration-constant-max"] !== nothing
        config_dict["exploration_constant_max"] = parsed_args["exploration-constant-max"]
    end
    
    if parsed_args["gpu-devices"] !== nothing
        gpu_ids = parse.(Int, split(parsed_args["gpu-devices"], ","))
        config_dict["gpu_devices"] = gpu_ids
    end
    
    if parsed_args["memory-limit-gb"] !== nothing
        config_dict["memory_limit_gb"] = parsed_args["memory-limit-gb"]
    end
    
    if parsed_args["data-path"] !== nothing
        config_dict["data_path"] = parsed_args["data-path"]
    end
    
    if parsed_args["output-path"] !== nothing
        config_dict["output_path"] = parsed_args["output-path"]
    end
    
    if parsed_args["log-path"] !== nothing
        config_dict["log_path"] = parsed_args["log-path"]
    end
    
    # Boolean flags
    if parsed_args["enable-profiling"]
        config_dict["enable_profiling"] = true
    elseif parsed_args["disable-profiling"]
        config_dict["enable_profiling"] = false
    end
    
    if parsed_args["enable-dashboard"]
        config_dict["enable_dashboard"] = true
    elseif parsed_args["disable-dashboard"]
        config_dict["enable_dashboard"] = false
    end
    
    if parsed_args["lazy-expansion"]
        config_dict["lazy_expansion"] = true
    elseif parsed_args["no-lazy-expansion"]
        config_dict["lazy_expansion"] = false
    end
    
    if parsed_args["shared-features"]
        config_dict["shared_features"] = true
    elseif parsed_args["no-shared-features"]
        config_dict["shared_features"] = false
    end
    
    if parsed_args["compressed-nodes"]
        config_dict["compressed_nodes"] = true
    elseif parsed_args["no-compressed-nodes"]
        config_dict["compressed_nodes"] = false
    end
    
    if parsed_args["fault-tolerance"]
        config_dict["fault_tolerance"] = true
    elseif parsed_args["no-fault-tolerance"]
        config_dict["fault_tolerance"] = false
    end
    
    # Create final configuration
    final_config = dict_to_config(config_dict)
    
    # Save configuration if requested
    if parsed_args["save-config"] !== nothing
        save_path = parsed_args["save-config"]
        if endswith(save_path, ".json")
            save_json_config(final_config, save_path)
        elseif endswith(save_path, ".yaml") || endswith(save_path, ".yml")
            save_yaml_config(final_config, save_path)
        else
            throw(ArgumentError("Unsupported save format: $save_path"))
        end
    end
    
    return final_config
end

"""
Display configuration summary.
"""
function display_config_summary(config::EnsembleConfiguration)
    println("=== Ensemble Configuration Summary ===")
    println()
    
    println("Core Parameters:")
    println("  Trees: $(config.num_trees) ($(config.trees_per_gpu) per GPU)")
    println("  Max nodes per tree: $(config.max_nodes_per_tree)")
    println("  Max depth: $(config.max_depth)")
    println("  Max iterations: $(config.max_iterations)")
    println()
    
    println("Feature Selection:")
    println("  Initial features: $(config.initial_features)")
    println("  Target features: $(config.target_features)")
    println("  Feature subset ratio: $(config.feature_subset_ratio)")
    println("  Diversity threshold: $(config.diversity_threshold)")
    println()
    
    println("MCTS Parameters:")
    println("  Exploration constant: $(config.exploration_constant_min) - $(config.exploration_constant_max)")
    println("  Virtual loss: $(config.virtual_loss)")
    println("  Batch size: $(config.batch_size)")
    println()
    
    println("GPU Configuration:")
    println("  Devices: $(config.gpu_devices)")
    println("  Memory limit: $(config.memory_limit_gb) GB")
    println()
    
    println("Memory Management:")
    println("  Pool size: $(config.memory_pool_size)")
    println("  GC threshold: $(config.gc_threshold)")
    println("  Defrag threshold: $(config.defrag_threshold)")
    println()
    
    println("Advanced Features:")
    println("  Lazy expansion: $(config.lazy_expansion)")
    println("  Shared features: $(config.shared_features)")
    println("  Compressed nodes: $(config.compressed_nodes)")
    println("  Fault tolerance: $(config.fault_tolerance)")
    println()
    
    println("Monitoring:")
    println("  Profiling: $(config.enable_profiling)")
    println("  Dashboard: $(config.enable_dashboard)")
    println("  Refresh rate: $(config.dashboard_refresh_ms) ms")
    println()
    
    println("File Paths:")
    println("  Model: $(config.model_path)")
    println("  Data: $(config.data_path)")
    println("  Output: $(config.output_path)")
    println("  Log: $(config.log_path)")
    println()
end

export EnsembleConfiguration
export load_json_config, load_yaml_config, save_json_config, save_yaml_config
export parse_args_and_load_config, display_config_summary
export dict_to_config, config_to_dict

end # module