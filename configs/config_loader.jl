# Configuration Loader Module
module ConfigLoader

using TOML
using Logging
using Dates

# Configuration structure
struct HSFOConfig
    gpu_config::Dict{String, Any}
    algorithm_config::Dict{String, Any}
    data_config::Dict{String, Any}
    environment::String
end

# Global configuration instance
const CONFIG = Ref{HSFOConfig}()

"""
    load_configs(env::String = "dev"; config_dir::String = "configs")

Load all configuration files for the specified environment.
"""
function load_configs(env::String = "dev"; config_dir::String = "configs")
    @info "Loading configurations for environment: $env"
    
    # Load base configurations
    gpu_config = load_and_merge_config(
        joinpath(config_dir, "gpu_config.toml"),
        joinpath(config_dir, "gpu_config.$env.toml")
    )
    
    algorithm_config = load_and_merge_config(
        joinpath(config_dir, "algorithm_config.toml"),
        joinpath(config_dir, "algorithm_config.$env.toml")
    )
    
    data_config = load_and_merge_config(
        joinpath(config_dir, "data_config.toml"),
        joinpath(config_dir, "data_config.$env.toml")
    )
    
    # Create configuration instance
    CONFIG[] = HSFOConfig(
        gpu_config,
        algorithm_config,
        data_config,
        env
    )
    
    @info "Configurations loaded successfully"
    return CONFIG[]
end

"""
    load_and_merge_config(base_path::String, env_path::String)

Load base configuration and merge with environment-specific overrides.
"""
function load_and_merge_config(base_path::String, env_path::String)
    # Load base configuration
    if !isfile(base_path)
        error("Base configuration file not found: $base_path")
    end
    
    config = TOML.parsefile(base_path)
    
    # Merge with environment-specific config if it exists
    if isfile(env_path)
        env_config = TOML.parsefile(env_path)
        merge_configs!(config, env_config)
        @info "Merged environment config from: $env_path"
    end
    
    return config
end

"""
    merge_configs!(base::Dict, override::Dict)

Recursively merge override configuration into base configuration.
"""
function merge_configs!(base::Dict, override::Dict)
    for (key, value) in override
        if haskey(base, key) && isa(base[key], Dict) && isa(value, Dict)
            merge_configs!(base[key], value)
        else
            base[key] = value
        end
    end
end

"""
    get_config()

Get the current configuration instance.
"""
function get_config()
    if !isassigned(CONFIG)
        error("Configuration not loaded. Call load_configs() first.")
    end
    return CONFIG[]
end

"""
    get_gpu_config()

Get GPU configuration.
"""
function get_gpu_config()
    return get_config().gpu_config
end

"""
    get_algorithm_config()

Get algorithm configuration.
"""
function get_algorithm_config()
    return get_config().algorithm_config
end

"""
    get_data_config()

Get data configuration.
"""
function get_data_config()
    return get_config().data_config
end

"""
    get_config_value(path::String, default = nothing)

Get a configuration value by dot-separated path.
Example: get_config_value("algorithm.mcts.max_iterations")
"""
function get_config_value(path::String, default = nothing)
    parts = split(path, ".")
    
    # Determine which config to use
    config_map = Dict(
        "gpu" => get_gpu_config(),
        "algorithm" => get_algorithm_config(),
        "data" => get_data_config()
    )
    
    if isempty(parts) || !haskey(config_map, parts[1])
        return default
    end
    
    # Navigate through the configuration
    current = config_map[parts[1]]
    
    for part in parts[2:end]
        if isa(current, Dict) && haskey(current, part)
            current = current[part]
        else
            return default
        end
    end
    
    return current
end

"""
    validate_config()

Validate the loaded configuration for required fields and valid values.
"""
function validate_config()
    config = get_config()
    errors = String[]
    
    # Validate GPU config
    gpu_cfg = config.gpu_config
    if !haskey(gpu_cfg, "cuda") || !haskey(gpu_cfg["cuda"], "memory_limit_gb")
        push!(errors, "Missing required GPU memory limit configuration")
    end
    
    # Validate algorithm config
    algo_cfg = config.algorithm_config
    if !haskey(algo_cfg, "mcts") || !haskey(algo_cfg["mcts"], "max_iterations")
        push!(errors, "Missing required MCTS configuration")
    end
    
    if haskey(algo_cfg, "filtering") && algo_cfg["filtering"]["target_features"] <= 0
        push!(errors, "Invalid target_features value: must be positive")
    end
    
    # Validate data config
    data_cfg = config.data_config
    if !haskey(data_cfg, "database") || !haskey(data_cfg["database"], "connection_pool_size")
        push!(errors, "Missing required database configuration")
    end
    
    if !isempty(errors)
        error("Configuration validation failed:\n" * join(errors, "\n"))
    end
    
    @info "Configuration validation passed"
    return true
end

"""
    save_runtime_config(path::String)

Save the current runtime configuration to a file.
"""
function save_runtime_config(path::String)
    config = get_config()
    
    runtime_config = Dict(
        "environment" => config.environment,
        "timestamp" => string(now()),
        "gpu" => config.gpu_config,
        "algorithm" => config.algorithm_config,
        "data" => config.data_config
    )
    
    open(path, "w") do io
        TOML.print(io, runtime_config)
    end
    
    @info "Saved runtime configuration to: $path"
end

"""
    override_config!(path::String, value)

Override a specific configuration value at runtime.
"""
function override_config!(path::String, value)
    parts = split(path, ".")
    
    if length(parts) < 2
        error("Invalid configuration path: $path")
    end
    
    # Get the appropriate config dict
    config_map = Dict(
        "gpu" => get_gpu_config(),
        "algorithm" => get_algorithm_config(),
        "data" => get_data_config()
    )
    
    if !haskey(config_map, parts[1])
        error("Invalid configuration section: $(parts[1])")
    end
    
    # Navigate to the parent dict
    current = config_map[parts[1]]
    for part in parts[2:end-1]
        if !isa(current, Dict) || !haskey(current, part)
            error("Invalid configuration path: $path")
        end
        current = current[part]
    end
    
    # Set the value
    current[parts[end]] = value
    @info "Configuration override: $path = $value"
end

end # module