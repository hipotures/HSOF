using Test
using CUDA
using Printf

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, skipping progress tracking tests"
    exit(0)
end

# Include progress tracking modules
include("../../src/stage1_filter/progress_tracking.jl")
include("../../src/stage1_filter/progress_kernels.jl")
using .ProgressTracking
using .ProgressKernels

println("Testing Progress Tracking System (Simplified)...")
println("="^60)

# Test 1: Basic progress tracker creation
println("\n1. Basic Progress Tracker")
tracker = create_progress_tracker(
    1000;
    description = "Test operation"
)

progress = get_progress(tracker)
println("  Initial state: $(progress.processed)/$(progress.total) ($(round(progress.percentage, digits=1))%)")
@test progress.processed == 0 && progress.total == 1000

# Test 2: Manual progress updates
println("\n2. Manual Progress Updates")
# Use copyto! to avoid scalar indexing
copyto!(tracker.gpu_progress.processed_items, [Int32(250)])
progress = get_progress(tracker)
println("  After update: $(progress.processed)/$(progress.total) ($(round(progress.percentage, digits=1))%)")
@test progress.processed == 250

# Test 3: Cancellation
println("\n3. Cancellation Mechanism")
println("  Cancelled: $(is_cancelled(tracker))")
cancel_operation!(tracker)
println("  After cancel: $(is_cancelled(tracker))")
@test is_cancelled(tracker)

# Test 4: Simple kernel with progress
println("\n4. GPU Kernel with Progress")
function simple_progress_kernel!(output, n, progress_items)
    idx = blockIdx().x
    tid = threadIdx().x
    
    if idx <= n && tid == 1
        output[idx] = Float32(idx)
        
        # Update progress every 10 items
        if idx % 10 == 0
            CUDA.@atomic progress_items[1] += Int32(10)
        elseif idx == n
            # Final update
            remaining = n % 10
            if remaining > 0
                CUDA.@atomic progress_items[1] += remaining
            end
        end
    end
    return nothing
end

# Create test data
n_items = 100
output = CUDA.zeros(Float32, n_items)
tracker2 = create_progress_tracker(n_items; description="GPU test")

# Launch kernel - pass the CuDeviceVector directly
@cuda threads=32 blocks=n_items simple_progress_kernel!(
    output, Int32(n_items), tracker2.gpu_progress.processed_items
)
CUDA.synchronize()

# Check progress
progress2 = get_progress(tracker2)
println("  GPU progress: $(progress2.processed)/$(progress2.total) ($(round(progress2.percentage, digits=1))%)")
@test progress2.processed == n_items

# Test 5: Progress with callback
println("\n5. Progress with Callback")
callback_count = Ref(0)
last_percentage = Ref(0.0)

tracker3 = create_progress_tracker(
    50;
    description = "Callback test",
    callback = function(info)
        callback_count[] += 1
        last_percentage[] = info[:percentage]
        print("\r  Progress: $(round(info[:percentage], digits=1))%")
        flush(stdout)
    end,
    callback_frequency = 0.0  # Immediate callbacks
)

# Simulate progress
for i in 1:5
    copyto!(tracker3.gpu_progress.processed_items, [Int32(i * 10)])
    update_progress!(tracker3)
    sleep(0.1)
end
println()

println("  Callbacks made: $(callback_count[])")
println("  Final percentage: $(round(last_percentage[], digits=1))%")
@test callback_count[] > 0

# Test 6: Time estimation
println("\n6. Time Estimation")
tracker4 = create_progress_tracker(1000)
tracker4.processing_rate = 100.0  # 100 items/sec
copyto!(tracker4.gpu_progress.processed_items, [Int32(200)])

eta = estimate_time_remaining(tracker4)
println("  Processing rate: 100 items/sec")
println("  Progress: 200/1000")
println("  ETA: $(round(eta, digits=1)) seconds")
@test abs(eta - 8.0) < 0.1  # Should be (1000-200)/100 = 8 seconds

# Test 7: Real variance calculation with progress
println("\n7. Variance Calculation with Progress")
X = CUDA.randn(Float32, 500, 1000)
variances = CUDA.zeros(Float32, 500)

tracker5 = create_progress_tracker(
    500;
    description = "Computing variances",
    callback = function(info)
        print("\r  $(info[:description]): $(info[:processed])/$(info[:total]) ($(round(info[:percentage], digits=1))%)")
        flush(stdout)
    end
)

# Launch variance kernel - use the v2 version
@cuda threads=256 blocks=500 variance_kernel_progress_v2!(
    variances, X, Int32(500), Int32(1000),
    tracker5.gpu_progress.processed_items,
    tracker5.gpu_progress.cancelled,
    Int32(50)  # Update every 50 features
)
CUDA.synchronize()

# Final update
update_progress!(tracker5, force_callback=true)
println()

# Verify
progress5 = get_progress(tracker5)
println("  Final: $(progress5.processed)/$(progress5.total)")
@test all(isfinite.(Array(variances)))

println("\n" * "="^60)
println("PROGRESS TRACKING TEST RESULTS")
println("="^60)
println("✓ Progress tracker creation: PASSED")
println("✓ Manual updates: PASSED")
println("✓ Cancellation: PASSED")
println("✓ GPU kernel integration: PASSED")
println("✓ Callbacks: PASSED")
println("✓ Time estimation: PASSED")
println("✓ Real computation: PASSED")
println("="^60)
println("\nAll tests passed successfully!")