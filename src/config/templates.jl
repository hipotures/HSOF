module ConfigTemplates

using ..EnsembleConfig

"""
Configuration templates for common ensemble scenarios.
"""

"""
Development/Testing configuration with reduced resources.
"""
function development_config()
    return EnsembleConfiguration(
        # Reduced scale for development
        num_trees = 10,
        trees_per_gpu = 5,
        max_nodes_per_tree = 1000,
        max_depth = 20,
        max_iterations = 10000,
        
        # Standard MCTS parameters
        exploration_constant_min = 0.5,
        exploration_constant_max = 2.0,
        virtual_loss = 5,
        
        # Smaller feature sets for testing
        initial_features = 100,
        target_features = 20,
        feature_subset_ratio = 0.8,
        diversity_threshold = 0.7,
        random_seed_base = 12345,
        
        # Conservative memory settings
        memory_pool_size = 0.5,
        gc_threshold = 0.6,
        defrag_threshold = 0.4,
        
        # Smaller batches for development
        batch_size = 256,
        update_interval_ms = 500,
        sync_interval_iterations = 100,
        
        # Single GPU for development
        gpu_devices = [0],
        memory_limit_gb = 8.0,
        
        # Quick convergence for testing
        convergence_window = 20,
        convergence_threshold = 0.05,
        min_iterations = 1000,
        
        # Development paths
        model_path = "test/models/metamodel.jl",
        data_path = "test/data/test_features.sqlite",
        output_path = "test/results/dev_results.json",
        log_path = "test/logs/dev.log",
        
        # Enable all optimizations
        lazy_expansion = true,
        shared_features = true,
        compressed_nodes = true,
        fault_tolerance = false,  # Disabled for simpler debugging
        
        # Enable monitoring for development
        enable_profiling = true,
        enable_dashboard = true,
        dashboard_refresh_ms = 200,
        
        # Relaxed performance targets
        target_gpu_utilization = 0.7,
        target_scaling_efficiency = 0.7
    )
end

"""
Production configuration optimized for dual RTX 4090 setup.
"""
function production_config()
    return EnsembleConfiguration(
        # Full scale production
        num_trees = 100,
        trees_per_gpu = 50,
        max_nodes_per_tree = 20000,
        max_depth = 50,
        max_iterations = 1000000,
        
        # Optimized MCTS parameters
        exploration_constant_min = 0.5,
        exploration_constant_max = 2.0,
        virtual_loss = 10,
        
        # Full feature selection pipeline
        initial_features = 500,
        target_features = 50,
        feature_subset_ratio = 0.8,
        diversity_threshold = 0.7,
        random_seed_base = 12345,
        
        # Aggressive memory utilization
        memory_pool_size = 0.85,
        gc_threshold = 0.8,
        defrag_threshold = 0.6,
        
        # High-performance batching
        batch_size = 1024,
        update_interval_ms = 100,
        sync_interval_iterations = 1000,
        
        # Dual RTX 4090 GPUs
        gpu_devices = [0, 1],
        memory_limit_gb = 22.0,
        
        # Production convergence settings
        convergence_window = 100,
        convergence_threshold = 0.01,
        min_iterations = 10000,
        
        # Production paths
        model_path = "models/metamodel.jl",
        data_path = "data/features.sqlite",
        output_path = "results/production_results.json",
        log_path = "logs/production.log",
        
        # All optimizations enabled
        lazy_expansion = true,
        shared_features = true,
        compressed_nodes = true,
        fault_tolerance = true,
        
        # Production monitoring
        enable_profiling = true,
        enable_dashboard = false,  # Disabled for production
        dashboard_refresh_ms = 100,
        
        # High performance targets
        target_gpu_utilization = 0.85,
        target_scaling_efficiency = 0.85
    )
end

"""
High-memory configuration for large feature sets (>1000 features).
"""
function high_memory_config()
    return EnsembleConfiguration(
        # Increased capacity
        num_trees = 100,
        trees_per_gpu = 50,
        max_nodes_per_tree = 50000,  # Increased for large feature sets
        max_depth = 100,
        max_iterations = 2000000,
        
        # Balanced exploration for large spaces
        exploration_constant_min = 0.3,
        exploration_constant_max = 1.5,
        virtual_loss = 15,
        
        # Large feature selection
        initial_features = 2000,
        target_features = 100,
        feature_subset_ratio = 0.6,  # Reduced for diversity
        diversity_threshold = 0.6,
        random_seed_base = 12345,
        
        # Maximum memory utilization
        memory_pool_size = 0.9,
        gc_threshold = 0.85,
        defrag_threshold = 0.7,
        
        # Large batch processing
        batch_size = 2048,
        update_interval_ms = 200,
        sync_interval_iterations = 2000,
        
        # Dual GPUs with max memory
        gpu_devices = [0, 1],
        memory_limit_gb = 23.0,
        
        # Extended convergence for complex problems
        convergence_window = 200,
        convergence_threshold = 0.005,
        min_iterations = 50000,
        
        # High-memory paths
        model_path = "models/large_metamodel.jl",
        data_path = "data/large_features.sqlite",
        output_path = "results/large_results.json",
        log_path = "logs/large.log",
        
        # All optimizations essential for large problems
        lazy_expansion = true,
        shared_features = true,
        compressed_nodes = true,
        fault_tolerance = true,
        
        # Extended monitoring
        enable_profiling = true,
        enable_dashboard = true,
        dashboard_refresh_ms = 500,
        
        # High performance targets
        target_gpu_utilization = 0.9,
        target_scaling_efficiency = 0.8
    )
end

"""
Fast exploration configuration for quick preliminary results.
"""
function fast_exploration_config()
    return EnsembleConfiguration(
        # Moderate scale for speed
        num_trees = 50,
        trees_per_gpu = 25,
        max_nodes_per_tree = 5000,
        max_depth = 30,
        max_iterations = 100000,
        
        # Aggressive exploration
        exploration_constant_min = 1.0,
        exploration_constant_max = 3.0,
        virtual_loss = 20,
        
        # Standard feature selection
        initial_features = 500,
        target_features = 50,
        feature_subset_ratio = 0.9,  # Higher for speed
        diversity_threshold = 0.8,
        random_seed_base = 12345,
        
        # Fast memory management
        memory_pool_size = 0.7,
        gc_threshold = 0.6,
        defrag_threshold = 0.3,
        
        # High-throughput batching
        batch_size = 2048,
        update_interval_ms = 50,
        sync_interval_iterations = 500,
        
        # Dual GPUs
        gpu_devices = [0, 1],
        memory_limit_gb = 20.0,
        
        # Quick convergence
        convergence_window = 50,
        convergence_threshold = 0.02,
        min_iterations = 5000,
        
        # Fast paths
        model_path = "models/fast_metamodel.jl",
        data_path = "data/features.sqlite",
        output_path = "results/fast_results.json",
        log_path = "logs/fast.log",
        
        # Speed-optimized features
        lazy_expansion = true,
        shared_features = true,
        compressed_nodes = true,
        fault_tolerance = false,  # Disabled for speed
        
        # Minimal monitoring overhead
        enable_profiling = false,
        enable_dashboard = false,
        dashboard_refresh_ms = 1000,
        
        # Moderate performance targets
        target_gpu_utilization = 0.8,
        target_scaling_efficiency = 0.75
    )
end

"""
Benchmarking configuration for performance testing.
"""
function benchmark_config()
    return EnsembleConfiguration(
        # Standard benchmarking scale
        num_trees = 100,
        trees_per_gpu = 50,
        max_nodes_per_tree = 10000,
        max_depth = 40,
        max_iterations = 500000,
        
        # Balanced parameters
        exploration_constant_min = 0.5,
        exploration_constant_max = 2.0,
        virtual_loss = 10,
        
        # Standard feature selection
        initial_features = 500,
        target_features = 50,
        feature_subset_ratio = 0.8,
        diversity_threshold = 0.7,
        random_seed_base = 42,  # Fixed for reproducibility
        
        # Benchmark memory settings
        memory_pool_size = 0.8,
        gc_threshold = 0.75,
        defrag_threshold = 0.5,
        
        # Optimized batching
        batch_size = 1024,
        update_interval_ms = 100,
        sync_interval_iterations = 1000,
        
        # Dual GPUs
        gpu_devices = [0, 1],
        memory_limit_gb = 22.0,
        
        # Fixed convergence for consistency
        convergence_window = 100,
        convergence_threshold = 0.01,
        min_iterations = 20000,
        
        # Benchmark paths
        model_path = "benchmark/models/metamodel.jl",
        data_path = "benchmark/data/features.sqlite",
        output_path = "benchmark/results/benchmark_results.json",
        log_path = "benchmark/logs/benchmark.log",
        
        # All optimizations for fair comparison
        lazy_expansion = true,
        shared_features = true,
        compressed_nodes = true,
        fault_tolerance = true,
        
        # Comprehensive profiling
        enable_profiling = true,
        enable_dashboard = false,
        dashboard_refresh_ms = 100,
        
        # Target performance
        target_gpu_utilization = 0.85,
        target_scaling_efficiency = 0.85
    )
end

"""
Single GPU configuration for systems with limited hardware.
"""
function single_gpu_config()
    return EnsembleConfiguration(
        # Adjusted for single GPU
        num_trees = 50,
        trees_per_gpu = 50,
        max_nodes_per_tree = 15000,
        max_depth = 45,
        max_iterations = 750000,
        
        # Standard MCTS parameters
        exploration_constant_min = 0.5,
        exploration_constant_max = 2.0,
        virtual_loss = 10,
        
        # Standard feature selection
        initial_features = 500,
        target_features = 50,
        feature_subset_ratio = 0.8,
        diversity_threshold = 0.7,
        random_seed_base = 12345,
        
        # Single GPU memory management
        memory_pool_size = 0.85,
        gc_threshold = 0.8,
        defrag_threshold = 0.6,
        
        # Optimized for single GPU
        batch_size = 1024,
        update_interval_ms = 100,
        sync_interval_iterations = 1000,
        
        # Single GPU only
        gpu_devices = [0],
        memory_limit_gb = 22.0,
        
        # Standard convergence
        convergence_window = 100,
        convergence_threshold = 0.01,
        min_iterations = 10000,
        
        # Single GPU paths
        model_path = "models/metamodel.jl",
        data_path = "data/features.sqlite",
        output_path = "results/single_gpu_results.json",
        log_path = "logs/single_gpu.log",
        
        # All optimizations important for single GPU
        lazy_expansion = true,
        shared_features = true,
        compressed_nodes = true,
        fault_tolerance = false,  # Less critical for single GPU
        
        # Standard monitoring
        enable_profiling = true,
        enable_dashboard = true,
        dashboard_refresh_ms = 100,
        
        # Adjusted targets for single GPU
        target_gpu_utilization = 0.85,
        target_scaling_efficiency = 1.0  # No scaling expected
    )
end

"""
Get configuration template by name.
"""
function get_template(name::String)
    templates = Dict(
        "development" => development_config,
        "dev" => development_config,
        "production" => production_config,
        "prod" => production_config,
        "high-memory" => high_memory_config,
        "large" => high_memory_config,
        "fast" => fast_exploration_config,
        "quick" => fast_exploration_config,
        "benchmark" => benchmark_config,
        "bench" => benchmark_config,
        "single-gpu" => single_gpu_config,
        "single" => single_gpu_config
    )
    
    if haskey(templates, name)
        return templates[name]()
    else
        available = join(keys(templates), ", ")
        throw(ArgumentError("Unknown template '$name'. Available templates: $available"))
    end
end

"""
List all available configuration templates.
"""
function list_templates()
    templates = [
        ("development", "Reduced scale for development and testing"),
        ("production", "Full scale for dual RTX 4090 production setup"),
        ("high-memory", "Large feature sets with increased memory usage"),
        ("fast", "Quick exploration for preliminary results"),
        ("benchmark", "Standard configuration for performance testing"),
        ("single-gpu", "Single GPU configuration for limited hardware")
    ]
    
    println("Available Configuration Templates:")
    println("=" ^ 50)
    
    for (name, description) in templates
        println("  $name: $description")
    end
    
    println()
    println("Usage: get_template(\"template_name\")")
end

"""
Save all templates to configuration files.
"""
function save_all_templates(base_dir::String = "config/templates")
    mkpath(base_dir)
    
    templates = [
        ("development", development_config()),
        ("production", production_config()),
        ("high-memory", high_memory_config()),
        ("fast", fast_exploration_config()),
        ("benchmark", benchmark_config()),
        ("single-gpu", single_gpu_config())
    ]
    
    for (name, config) in templates
        json_path = joinpath(base_dir, "$name.json")
        yaml_path = joinpath(base_dir, "$name.yaml")
        
        save_json_config(config, json_path)
        save_yaml_config(config, yaml_path)
        
        @info "Saved template '$name' to $json_path and $yaml_path"
    end
    
    @info "All configuration templates saved to $base_dir"
end

export development_config, production_config, high_memory_config
export fast_exploration_config, benchmark_config, single_gpu_config
export get_template, list_templates, save_all_templates

end # module