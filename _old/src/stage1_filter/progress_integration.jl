module ProgressIntegration

using CUDA
using Printf
using ..ProgressTracking
using ..VarianceCalculation
using ..MutualInformation
using ..CorrelationMatrix

export FastFilteringWithProgress, run_with_progress!, create_progress_bar
export ProgressConfig, DefaultProgressBar, SilentProgressBar

"""
Configuration for progress tracking behavior
"""
struct ProgressConfig
    enable_progress::Bool
    update_frequency::Int32  # Update every N features
    callback_frequency::Float64  # Seconds between callbacks
    show_eta::Bool
    show_rate::Bool
    cancellable::Bool
end

function ProgressConfig(;
    enable_progress::Bool = true,
    update_frequency::Int32 = Int32(100),
    callback_frequency::Float64 = 0.5,
    show_eta::Bool = true,
    show_rate::Bool = true,
    cancellable::Bool = true
)
    return ProgressConfig(
        enable_progress,
        update_frequency,
        callback_frequency,
        show_eta,
        show_rate,
        cancellable
    )
end

"""
Fast filtering operations with integrated progress tracking
"""
struct FastFilteringWithProgress
    config::ProgressConfig
    tracker::Union{ProgressTracker, Nothing}
end

"""
Default console progress bar
"""
function DefaultProgressBar(info::Dict)
    bar_width = 30
    percentage = info[:percentage]
    filled = Int(round(bar_width * percentage / 100))
    bar = "█"^filled * "░"^(bar_width - filled)
    
    output = @sprintf("\r[%s] %.1f%% - %s: %d/%d",
        bar,
        percentage,
        info[:description],
        info[:processed],
        info[:total]
    )
    
    if info[:rate] > 0 && haskey(info, :show_rate) && info[:show_rate]
        output *= @sprintf(" (%.0f items/sec)", info[:rate])
    end
    
    if info[:eta_seconds] < Inf && haskey(info, :show_eta) && info[:show_eta]
        eta_min = Int(floor(info[:eta_seconds] / 60))
        eta_sec = Int(info[:eta_seconds] % 60)
        output *= @sprintf(" ETA: %02d:%02d", eta_min, eta_sec)
    end
    
    print(output)
    flush(stdout)
end

"""
Silent progress bar (no output)
"""
function SilentProgressBar(info::Dict)
    # No output
end

"""
Create appropriate progress bar based on config
"""
function create_progress_bar(config::ProgressConfig)
    if !config.enable_progress
        return SilentProgressBar
    else
        return function(info)
            info[:show_eta] = config.show_eta
            info[:show_rate] = config.show_rate
            DefaultProgressBar(info)
        end
    end
end

"""
Modified variance kernel with progress integration
"""
function variance_kernel_progress!(
    variances::CuDeviceVector{Float32},
    X::CuDeviceMatrix{Float32},
    n_features::Int32,
    n_samples::Int32,
    progress::GPUProgress,
    update_frequency::Int32
)
    feat_idx = Int32(blockIdx().x)
    tid = Int32(threadIdx().x)
    block_size = Int32(blockDim().x)
    
    if feat_idx > n_features
        return
    end
    
    # Check cancellation
    if progress.cancelled[1] != Int32(0)
        return
    end
    
    # Shared memory for reduction
    shared_sum = @cuDynamicSharedMem(Float32, block_size)
    shared_sum_sq = @cuDynamicSharedMem(Float32, block_size, block_size * sizeof(Float32))
    
    # Initialize
    local_sum = 0.0f0
    local_sum_sq = 0.0f0
    
    # Process samples
    for idx in tid:block_size:n_samples
        if idx <= n_samples
            val = X[feat_idx, idx]
            local_sum += val
            local_sum_sq += val * val
        end
    end
    
    # Store in shared memory
    shared_sum[tid] = local_sum
    shared_sum_sq[tid] = local_sum_sq
    sync_threads()
    
    # Reduction
    stride = block_size ÷ 2
    while stride > 0
        if tid <= stride
            shared_sum[tid] += shared_sum[tid + stride]
            shared_sum_sq[tid] += shared_sum_sq[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    
    # Final calculation and progress update
    if tid == 1
        mean = shared_sum[1] / Float32(n_samples)
        variance = (shared_sum_sq[1] / Float32(n_samples)) - mean * mean
        variances[feat_idx] = max(variance, 0.0f0)
        
        # Update progress
        if feat_idx % update_frequency == 0 || feat_idx == n_features
            items = feat_idx % update_frequency
            if items == 0
                items = update_frequency
            end
            CUDA.@atomic progress.processed_items[1] += items
        end
    end
    
    return nothing
end

"""
Run variance calculation with progress tracking
"""
function compute_variance_with_progress(
    X::CuArray{Float32, 2},
    config::ProgressConfig = ProgressConfig()
)
    n_features, n_samples = size(X)
    variances = CUDA.zeros(Float32, n_features)
    
    if !config.enable_progress
        # Run without progress tracking
        variances = compute_variance(X)
        return variances
    end
    
    # Create progress tracker
    tracker = create_progress_tracker(
        n_features;
        description = "Computing variances",
        callback = create_progress_bar(config),
        callback_frequency = config.callback_frequency
    )
    
    # Launch kernel with progress
    shared_mem = 2 * 256 * sizeof(Float32)
    @cuda threads=256 blocks=n_features shmem=shared_mem variance_kernel_progress!(
        variances, X, Int32(n_features), Int32(n_samples),
        tracker.gpu_progress, config.update_frequency
    )
    
    # Monitor progress
    monitor_progress(tracker, config)
    
    return variances
end

"""
Monitor progress until completion or cancellation
"""
function monitor_progress(tracker::ProgressTracker, config::ProgressConfig)
    while true
        update_progress!(tracker)
        
        progress = get_progress(tracker)
        if progress.processed >= progress.total
            update_progress!(tracker, force_callback=true)
            println()  # New line after progress
            break
        end
        
        if config.cancellable && is_cancelled(tracker)
            println("\nOperation cancelled!")
            break
        end
        
        sleep(0.05)
    end
    
    CUDA.synchronize()
end

"""
Run complete Stage 1 filtering with progress tracking
"""
function run_with_progress!(
    X::CuArray{Float32, 2},
    y::CuArray{Int32, 1},
    config::GPUConfig;
    progress_config::ProgressConfig = ProgressConfig()
)
    n_features, n_samples = size(X)
    
    # Create batch tracker for all operations
    operations = [
        ("Computing variances", n_features),
        ("Computing MI scores", n_features),
        ("Computing correlations", div(n_features * (n_features - 1), 2))
    ]
    
    batch_tracker = create_batch_tracker(operations)
    
    results = Dict{Symbol, Any}()
    
    # 1. Variance calculation
    println("Stage 1/3: Variance Calculation")
    variances = compute_variance_with_progress(X, progress_config)
    results[:variances] = variances
    
    # 2. MI calculation
    println("\nStage 2/3: Mutual Information")
    mi_tracker = create_progress_tracker(
        n_features;
        description = "Computing MI scores",
        callback = create_progress_bar(progress_config),
        callback_frequency = progress_config.callback_frequency
    )
    
    mi_scores = compute_mutual_information_with_progress(
        X, y, config, mi_tracker, progress_config
    )
    results[:mi_scores] = mi_scores
    
    # 3. Correlation matrix
    println("\nStage 3/3: Correlation Matrix")
    corr_tracker = create_progress_tracker(
        div(n_features * (n_features - 1), 2);
        description = "Computing correlations",
        callback = create_progress_bar(progress_config),
        callback_frequency = progress_config.callback_frequency
    )
    
    corr_matrix = compute_correlation_with_progress(
        X, config, corr_tracker, progress_config
    )
    results[:correlation_matrix] = corr_matrix
    
    return results
end

"""
MI calculation with progress
"""
function compute_mutual_information_with_progress(
    X::CuArray{Float32, 2},
    y::CuArray{Int32, 1},
    config::GPUConfig,
    tracker::ProgressTracker,
    progress_config::ProgressConfig
)
    n_features, n_samples = size(X)
    mi_scores = CUDA.zeros(Float32, n_features)
    
    # Use existing MI kernel with progress wrapper
    # (Implementation would integrate with actual MI kernel)
    
    # For now, use simplified version
    @cuda threads=256 blocks=n_features mi_kernel_with_progress!(
        mi_scores, X, y, Int32(n_features), Int32(n_samples),
        Int32(10), Int32(length(unique(Array(y)))),
        tracker.gpu_progress, progress_config.update_frequency
    )
    
    monitor_progress(tracker, progress_config)
    
    return mi_scores
end

"""
Correlation calculation with progress
"""
function compute_correlation_with_progress(
    X::CuArray{Float32, 2},
    config::GPUConfig,
    tracker::ProgressTracker,
    progress_config::ProgressConfig
)
    # Simplified implementation
    corr_matrix = compute_correlation_matrix(X, config)
    
    # Update progress to 100%
    total = Array(tracker.gpu_progress.total_items)[1]
    tracker.gpu_progress.processed_items[1] = Int32(total)
    update_progress!(tracker, force_callback=true)
    println()
    
    return corr_matrix
end

"""
Example usage with cancellation
"""
function example_with_cancellation()
    # Create test data
    X = CUDA.randn(Float32, 5000, 10000)
    y = CuArray(Int32.(rand(1:3, 10000)))
    
    config = ProgressConfig(
        enable_progress = true,
        show_eta = true,
        show_rate = true,
        cancellable = true
    )
    
    # Create tracker
    tracker = create_progress_tracker(
        5000;
        description = "Processing features",
        callback = create_progress_bar(config)
    )
    
    # Launch async task
    task = @async begin
        variances = CUDA.zeros(Float32, 5000)
        @cuda threads=256 blocks=5000 variance_kernel_progress!(
            variances, X, Int32(5000), Int32(10000),
            tracker.gpu_progress, config.update_frequency
        )
        CUDA.synchronize()
    end
    
    # Simulate cancellation after 2 seconds
    @async begin
        sleep(2.0)
        println("\n\nTriggering cancellation...")
        cancel_operation!(tracker)
    end
    
    # Monitor until done or cancelled
    monitor_progress(tracker, config)
    
    # Wait for task
    wait(task)
    
    # Check final status
    if is_cancelled(tracker)
        println("Operation was successfully cancelled")
    else
        println("Operation completed normally")
    end
end

end # module