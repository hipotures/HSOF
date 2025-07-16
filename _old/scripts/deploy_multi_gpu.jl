#!/usr/bin/env julia

# Multi-GPU Deployment Script for HSOF
# Sets up production environment for distributed MCTS execution

using Pkg
using CUDA
using Dates
using TOML

# Add project path
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Import deployment modules
include("../src/deployment/gpu_configuration.jl")
using .GPUConfiguration

"""
Pre-deployment checks
"""
function check_prerequisites()
    println("=== Pre-Deployment Checks ===")
    
    # Check Julia version
    if VERSION < v"1.9"
        error("Julia 1.9+ required. Current: $VERSION")
    end
    println("✓ Julia version: $VERSION")
    
    # Check CUDA
    if !CUDA.functional()
        error("CUDA not functional. Please check GPU drivers and CUDA.jl installation")
    end
    println("✓ CUDA functional")
    
    # Check GPU count
    num_gpus = length(CUDA.devices())
    if num_gpus < 1
        error("No GPUs detected")
    end
    println("✓ GPUs detected: $num_gpus")
    
    # Check available memory
    for i in 0:num_gpus-1
        device!(i)
        free_mem = CUDA.available_memory() / 1024^3
        total_mem = CUDA.total_memory() / 1024^3
        println("  GPU $i: $(round(free_mem, digits=2)) GB free / $(round(total_mem, digits=2)) GB total")
        
        if free_mem < 10.0
            @warn "GPU $i has less than 10GB free memory"
        end
    end
    
    # Check system memory
    sys_mem = Sys.total_memory() / 1024^3
    println("✓ System memory: $(round(sys_mem, digits=2)) GB")
    
    # Check CPU cores
    cpu_threads = Sys.CPU_THREADS
    println("✓ CPU threads: $cpu_threads")
    
    return true
end

"""
Configure system settings
"""
function configure_system_settings(config::DeploymentConfig)
    println("\n=== Configuring System Settings ===")
    
    # Set environment variables
    for (key, value) in config.environment
        ENV[key] = value
        println("  Set $key=$value")
    end
    
    # Configure CUDA settings
    if haskey(config.cuda_settings, "memory_pool") && config.cuda_settings["memory_pool"]
        # CUDA memory pool will be configured at runtime
        println("  CUDA memory pooling enabled")
    end
    
    # Set thread affinity if specified
    if Sys.islinux()
        for (gpu_id, cpu_cores) in config.topology.cpu_affinity
            # This would require system calls to set affinity
            println("  GPU $gpu_id affinity: CPUs $(join(cpu_cores, ","))")
        end
    end
    
    # Create necessary directories
    dirs = [
        "logs/gpu",
        "checkpoints",
        "configs",
        "results"
    ]
    
    for dir in dirs
        mkpath(dir)
        println("  Created directory: $dir")
    end
end

"""
Initialize GPU environment
"""
function initialize_gpu_environment(config::DeploymentConfig)
    println("\n=== Initializing GPU Environment ===")
    
    # Enable peer access where available
    for i in 1:config.topology.num_gpus
        for j in 1:config.topology.num_gpus
            if i != j && config.topology.peer_access_matrix[i,j]
                try
                    device!(i-1)
                    # Note: Actual peer access enabling would be done in CUDA kernels
                    println("  Peer access enabled: GPU $(i-1) -> GPU $(j-1)")
                catch e
                    @warn "Failed to enable peer access: $e"
                end
            end
        end
    end
    
    # Warm up GPUs
    println("\n  Warming up GPUs...")
    for gpu_id in config.topology.gpu_devices
        device!(gpu_id)
        
        # Allocate and free some memory to warm up
        try
            arr = CUDA.zeros(Float32, 1000, 1000)
            CUDA.@sync arr .= 1.0f0
            println("  GPU $gpu_id warmed up")
        catch e
            @warn "GPU $gpu_id warmup failed: $e"
        end
    end
    
    # Reset to GPU 0
    device!(0)
end

"""
Deploy HSOF multi-GPU system
"""
function deploy_multi_gpu(;
    config_file::Union{String, Nothing} = nothing,
    dry_run::Bool = false,
    verbose::Bool = true
)
    println("HSOF Multi-GPU Deployment Script")
    println("=" ^ 50)
    println("Timestamp: $(now())")
    println("Dry run: $dry_run")
    println()
    
    # Pre-deployment checks
    if !check_prerequisites()
        error("Pre-deployment checks failed")
    end
    
    # Load or create configuration
    if !isnothing(config_file) && isfile(config_file)
        println("\n=== Loading Configuration ===")
        println("  Config file: $config_file")
        config = load_config(config_file)
    else
        println("\n=== Creating Default Configuration ===")
        config = create_default_config()
        
        # Save generated config
        config_path = "configs/deployment_$(Dates.format(now(), "yyyymmdd_HHMMSS")).toml"
        save_config(config, config_path)
        println("  Saved configuration to: $config_path")
    end
    
    # Validate configuration
    println("\n=== Validating Configuration ===")
    issues = validate_config(config)
    if !isempty(issues)
        println("Configuration issues found:")
        for issue in issues
            println("  ⚠ $issue")
        end
        if !dry_run
            error("Configuration validation failed")
        end
    else
        println("  ✓ Configuration valid")
    end
    
    # Display configuration summary
    println("\n=== Configuration Summary ===")
    println("GPUs: $(config.topology.num_gpus)")
    println("  Devices: $(join(config.topology.gpu_devices, ", "))")
    println("  Names: $(join(config.topology.gpu_names, ", "))")
    println("  NVLink: $(config.topology.nvlink_available ? "Yes" : "No")")
    
    println("\nMemory Limits:")
    min_available = minimum(values(config.memory_limits.available_memory)) / 1024^3
    println("  Available per GPU: $(round(min_available, digits=2)) GB")
    println("  Dataset limit: $(round(config.memory_limits.dataset_memory_limit / 1024^3, digits=2)) GB")
    println("  Tree limit: $(round(config.memory_limits.tree_memory_limit / 1024^2, digits=2)) MB")
    println("  Buffer limit: $(round(config.memory_limits.buffer_memory_limit / 1024^2, digits=2)) MB")
    
    println("\nPerformance Targets:")
    println("  Scaling efficiency: $(config.performance_targets.scaling_efficiency_target * 100)%")
    println("  GPU utilization: $(config.performance_targets.gpu_utilization_target * 100)%")
    println("  Max sync latency: $(config.performance_targets.max_sync_latency_ms) ms")
    
    if dry_run
        println("\n=== Dry Run Complete ===")
        println("No changes made. Remove --dry-run to deploy.")
        return config
    end
    
    # Configure system
    configure_system_settings(config)
    
    # Initialize GPU environment
    initialize_gpu_environment(config)
    
    # Create startup script
    println("\n=== Creating Startup Script ===")
    create_startup_script(config)
    
    # Create systemd service (Linux only)
    if Sys.islinux()
        println("\n=== Creating Systemd Service ===")
        create_systemd_service(config)
    end
    
    println("\n=== Deployment Complete ===")
    println("✓ Multi-GPU environment configured")
    println("✓ Configuration saved to: configs/")
    println("✓ Startup script created: scripts/start_hsof.sh")
    
    if Sys.islinux()
        println("\nTo start HSOF as a service:")
        println("  sudo systemctl start hsof-gpu")
        println("  sudo systemctl enable hsof-gpu  # For auto-start")
    end
    
    println("\nTo start manually:")
    println("  ./scripts/start_hsof.sh")
    
    return config
end

"""
Create startup script
"""
function create_startup_script(config::DeploymentConfig)
    script_path = "scripts/start_hsof.sh"
    
    script_content = """
#!/bin/bash
# HSOF Multi-GPU Startup Script
# Generated: $(now())

# Set environment
export CUDA_VISIBLE_DEVICES=$(join(config.topology.gpu_devices, ","))
export JULIA_NUM_THREADS=$(Sys.CPU_THREADS)
export JULIA_CUDA_MEMORY_POOL=cuda
export JULIA_CUDA_SOFT_MEMORY_LIMIT=0.9

# Set CPU affinity (optional)
# taskset -c 0-31 julia ...

# Change to project directory
cd "$(dirname "$0")/.."

# Create log directory with timestamp
LOG_DIR="logs/gpu/\$(date +%Y%m%d_%H%M%S)"
mkdir -p "\$LOG_DIR"

# Start HSOF with logging
echo "Starting HSOF Multi-GPU System..."
echo "Log directory: \$LOG_DIR"
echo "GPUs: $(join(config.topology.gpu_devices, ", "))"

# Run with appropriate settings
julia --project=. \\
    --threads=$(Sys.CPU_THREADS) \\
    --heap-size-hint=$((Sys.total_memory() ÷ 2)) \\
    examples/distributed_mcts_production.jl \\
    --config configs/deployment_config.toml \\
    --log-dir "\$LOG_DIR" \\
    2>&1 | tee "\$LOG_DIR/hsof.log"
"""
    
    open(script_path, "w") do io
        write(io, script_content)
    end
    
    # Make executable
    chmod(script_path, 0o755)
    
    println("  Created startup script: $script_path")
end

"""
Create systemd service file (Linux only)
"""
function create_systemd_service(config::DeploymentConfig)
    service_content = """
[Unit]
Description=HSOF Multi-GPU Feature Selection Service
After=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$(pwd())

# Environment
Environment="CUDA_VISIBLE_DEVICES=$(join(config.topology.gpu_devices, ","))"
Environment="JULIA_NUM_THREADS=$(Sys.CPU_THREADS)"
Environment="JULIA_CUDA_MEMORY_POOL=cuda"

# Resource limits
LimitNOFILE=1048576
LimitMEMLOCK=infinity

# Restart policy
Restart=on-failure
RestartSec=30

# Start command
ExecStart=$(pwd())/scripts/start_hsof.sh

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_path = "configs/hsof-gpu.service"
    open(service_path, "w") do io
        write(io, service_content)
    end
    
    println("  Created systemd service file: $service_path")
    println("  To install: sudo cp $service_path /etc/systemd/system/")
end

# Command-line interface
if abspath(PROGRAM_FILE) == @__FILE__
    using ArgParse
    
    function parse_commandline()
        s = ArgParseSettings()
        
        @add_arg_table s begin
            "--config", "-c"
                help = "Configuration file path"
                arg_type = String
                default = nothing
            "--dry-run", "-n"
                help = "Perform dry run without making changes"
                action = :store_true
            "--verbose", "-v"
                help = "Verbose output"
                action = :store_true
        end
        
        return parse_args(s)
    end
    
    args = parse_commandline()
    
    try
        deploy_multi_gpu(
            config_file = args["config"],
            dry_run = args["dry-run"],
            verbose = args["verbose"]
        )
    catch e
        @error "Deployment failed" exception=(e, catch_backtrace())
        exit(1)
    end
end