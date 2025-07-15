module GPU

using CUDA

# GPU Management and Configuration
include("gpu_manager.jl")
include("memory_manager.jl")
include("stream_manager.jl")

# Re-export main functionality from included modules
using .GPUManager
using .MemoryManager
using .StreamManager

# Now include device_manager which depends on the above modules
include("device_manager.jl")
using .DeviceManager

# MCTS GPU Implementation
include("mcts_gpu.jl")
using .MCTSGPU

# Work Distribution for Multi-GPU
include("work_distribution.jl")
using .WorkDistribution

# PCIe Communication for Multi-GPU
include("pcie_communication.jl")
using .PCIeCommunication

# GPU Synchronization for Multi-GPU Coordination
include("gpu_synchronization.jl")
using .GPUSynchronization

# Dataset Storage for Multi-GPU
include("dataset_storage.jl")
using .DatasetStorage

# Fault Tolerance for Multi-GPU
include("fault_tolerance.jl")
using .FaultTolerance

# Performance Monitoring for Multi-GPU
include("performance_monitoring.jl")
using .PerformanceMonitoring

# Dynamic Rebalancing for Multi-GPU
include("dynamic_rebalancing.jl")
using .DynamicRebalancing

# Result Aggregation for Multi-GPU
include("result_aggregation.jl")
using .ResultAggregation

# GPU Management exports
export initialize_devices, get_device_info, validate_gpu_environment
export GPUManager, DeviceManager, StreamManager, MemoryManager

# MCTS GPU exports
export MCTSGPUEngine, initialize!, start!, stop!, get_statistics
export select_features, get_best_features, reset_tree!

# Work Distribution exports
export WorkDistributor, GPUWorkAssignment, WorkloadMetrics
export create_work_distributor, assign_tree_work, assign_metamodel_work
export get_gpu_for_tree, get_tree_range, get_load_balance_ratio
export update_metrics!, rebalance_if_needed!, get_work_summary
export set_gpu_affinity, execute_on_gpu

# PCIe Communication exports
export PCIeTransferManager, CandidateData, TransferBuffer, TransferStats
export create_transfer_manager, select_top_candidates, transfer_candidates
export enable_peer_access!, can_access_peer, get_transfer_stats
export reset_buffer!, should_transfer, compress_candidates, decompress_candidates
export add_candidates!, broadcast_candidates

# GPU Synchronization exports
export SyncManager, SyncState, SyncPhase, SyncBarrier, SyncEvent
export create_sync_manager, set_phase!, wait_for_phase, signal_event!
export wait_for_event, enter_barrier!, reset_barrier!, get_sync_state
export acquire_lock!, release_lock!, with_lock, is_phase_complete
export set_gpu_result!, get_gpu_result, get_all_results, clear_results!
export register_gpu!, unregister_gpu!, get_active_gpus
export reset_event!, update_sync_stats!
# Export phase enum values
export PHASE_INIT, PHASE_READY, PHASE_RUNNING, PHASE_SYNCING, PHASE_DONE, PHASE_ERROR

# Dataset Storage exports
export DatasetReplica, DatasetManager, DatasetVersion, MemoryStats
export create_dataset_manager, replicate_dataset!, get_dataset_replica
export update_dataset!, get_memory_usage, has_sufficient_memory
export load_dataset_to_gpu!, clear_dataset!, sync_datasets!
export get_feature_column, get_sample_batch, get_dataset_info

# Fault Tolerance exports
export GPUHealthMonitor, GPUStatus, FailureMode, CheckpointManager
export create_health_monitor, start_monitoring!, stop_monitoring!
export check_gpu_health, is_gpu_healthy, get_gpu_status
export register_error_callback!, handle_gpu_error
export create_checkpoint_manager, save_checkpoint, restore_checkpoint
export redistribute_work!, enable_graceful_degradation!
export get_failure_statistics, reset_failure_count
# Export status enum values
export GPU_HEALTHY, GPU_DEGRADED, GPU_FAILING, GPU_FAILED, GPU_RECOVERING
# Export failure mode enum values
export NO_FAILURE, CUDA_ERROR, MEMORY_ERROR, TIMEOUT_ERROR
export HEARTBEAT_FAILURE, THERMAL_THROTTLE, POWER_LIMIT

# Performance Monitoring exports
export PerformanceMonitor, GPUMetrics, KernelProfile, MemoryMetrics
export create_performance_monitor, start_monitoring!, stop_monitoring!
export record_kernel_start!, record_kernel_end!, get_kernel_stats
export update_gpu_metrics!, get_gpu_metrics, get_all_metrics
export detect_anomalies, get_performance_summary
export log_performance_data, set_log_level!
export reset_metrics!, export_metrics
# Export log level enum values
export LOG_NONE, LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG, LOG_TRACE

# Dynamic Rebalancing exports
export RebalancingManager, RebalancingDecision, MigrationPlan, RebalancingMetrics
export create_rebalancing_manager, start_rebalancing!, stop_rebalancing!
export check_imbalance, should_rebalance, create_migration_plan
export execute_migration!, update_workload_metrics!
export get_rebalancing_history, get_current_distribution
export set_rebalancing_threshold!, enable_auto_rebalancing!
# Export rebalancing state enum values
export BALANCED, MONITORING, PLANNING, MIGRATING, STABILIZING

# Result Aggregation exports
export ResultAggregator, TreeResult, FeatureScore, EnsembleResult
export create_result_aggregator, submit_tree_result!, aggregate_results
export get_ensemble_consensus, get_feature_rankings, get_aggregated_results
export clear_results!, reset_cache!, get_cache_stats
export set_consensus_threshold!, enable_caching!

# Module initialization
function __init__()
    if CUDA.functional()
        # Initialize GPU subsystem
        try
            gpu_manager = GPUManager.initialize()
            device_count = GPUManager.device_count()
            println("HSOF GPU Module initialized with $(device_count) GPU(s)")
            for i in 0:device_count-1
                dev = GPUManager.get_device(i)
                println("  GPU $i: $(dev.name) ($(round(dev.total_memory/1024^3, digits=2))GB)")
            end
        catch e
            @warn "Failed to initialize GPU manager: $e"
        end
    else
        @warn "CUDA.jl not functional - GPU features disabled"
    end
end

end # module GPU