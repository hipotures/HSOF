"""
Configuration loader for HSOF pipeline.
Loads and merges configuration parameters from YAML files.
"""

using YAML

"""
Load HSOF pipeline configuration with mode presets.
"""
function load_hsof_config(config_path::String="config/hsof.yaml")
    # Load base configuration
    config = YAML.load_file(config_path)
    
    # Apply mode preset if specified
    mode = get(config, "mode", "balanced")
    if haskey(config, "presets") && haskey(config["presets"], mode)
        preset = config["presets"][mode]
        # Only merge if preset is not empty
        if preset !== nothing && !isempty(preset)
            println("Applying '$mode' mode preset...")
            config = merge_configs(config, preset)
        end
    end
    
    return config
end

"""
Recursively merge configuration dictionaries.
"""
function merge_configs(base::Dict, override::Dict)
    result = copy(base)
    
    for (key, value) in override
        if haskey(result, key) && isa(result[key], Dict) && isa(value, Dict)
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else
            # Override value
            result[key] = value
        end
    end
    
    return result
end

"""
Get configuration value with default fallback.
"""
function get_config_value(config::Dict, keys::String...; default=nothing)
    current = config
    
    for key in keys
        if haskey(current, key)
            current = current[key]
        else
            return default
        end
    end
    
    return current
end

"""
Print current configuration settings.
"""
function print_config_summary(config::Dict)
    println("\n=== HSOF Configuration Summary ===")
    println("Mode: $(get(config, "mode", "custom"))")
    
    # Stage 1
    println("\nStage 1 (Correlation Filtering):")
    s1 = config["stage1"]
    println("  Correlation threshold: $(s1["correlation_threshold"])")
    println("  Min features: $(s1["min_features_to_keep"])")
    
    # Stage 2
    println("\nStage 2 (MCTS + Metamodel):")
    s2 = config["stage2"]
    println("  Total iterations: $(s2["total_iterations"])")
    println("  Trees: $(s2["n_trees"])")
    println("  Pretraining samples: $(s2["pretraining_samples"])")
    println("  Architecture: $(s2["hidden_sizes"])")
    
    # Stage 3
    println("\nStage 3 (Model Evaluation):")
    s3 = config["stage3"]
    println("  Candidate subsets: $(s3["n_candidate_subsets"])")
    println("  CV folds: $(s3["cv_folds"])")
    println("  Final features: $(s3["min_features_final"])-$(s3["max_features_final"])")
    
    println("\n" * "="^40)
end