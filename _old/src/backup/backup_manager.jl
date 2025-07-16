module BackupManager

using Dates
using Logging

# Include all backup modules
include("backup_types.jl")
include("orchestrator/backup_scheduler.jl")
include("storage/s3_storage.jl")
include("compression/backup_compression.jl")
include("monitoring/backup_monitor.jl")
include("restore/backup_restore.jl")

using .BackupTypes
using .BackupScheduler
using .S3Storage
using .BackupCompression
using .BackupMonitor
using .BackupRestore

export BackupManagerService, start_backup_system, stop_backup_system
export configure_backup_policies, get_system_status, create_manual_backup
export restore_from_backup, list_backups, test_backup_system

# Main backup manager service
mutable struct BackupManagerService
    running::Bool
    orchestrator::BackupOrchestrator
    monitor::BackupMonitoringService
    s3_client::S3Client
    
    function BackupManagerService()
        new(false, BackupScheduler.ORCHESTRATOR, BackupMonitor.MONITOR_SERVICE, S3Storage.DEFAULT_S3_CLIENT)
    end
end

# Global backup manager instance
const BACKUP_MANAGER = BackupManagerService()

"""
Start the complete backup system
"""
function start_backup_system(manager::BackupManagerService = BACKUP_MANAGER)
    if manager.running
        @warn "Backup system is already running"
        return false
    end
    
    @info "Starting HSOF Backup System..."
    
    try
        # Start backup orchestrator
        @info "Starting backup orchestrator..."
        start_scheduler(manager.orchestrator)
        
        # Start monitoring service
        @info "Starting backup monitoring..."
        start_monitoring(manager.monitor)
        
        # Test S3 connection
        @info "Testing S3 storage connection..."
        if !test_connection(manager.s3_client)
            @warn "S3 storage connection failed - backups will be stored locally only"
        else
            @info "S3 storage connection successful"
        end
        
        manager.running = true
        @info "✅ HSOF Backup System started successfully"
        
        # Log system configuration
        log_system_configuration(manager)
        
        return true
        
    catch e
        @error "Failed to start backup system" exception=e
        
        # Cleanup on failure
        try
            stop_scheduler(manager.orchestrator)
            stop_monitoring(manager.monitor)
        catch cleanup_error
            @debug "Error during cleanup" exception=cleanup_error
        end
        
        return false
    end
end

"""
Stop the complete backup system
"""
function stop_backup_system(manager::BackupManagerService = BACKUP_MANAGER)
    if !manager.running
        @info "Backup system is not running"
        return
    end
    
    @info "Stopping HSOF Backup System..."
    
    try
        # Stop monitoring first
        @info "Stopping backup monitoring..."
        stop_monitoring(manager.monitor)
        
        # Stop orchestrator
        @info "Stopping backup orchestrator..."
        stop_scheduler(manager.orchestrator)
        
        manager.running = false
        @info "✅ HSOF Backup System stopped successfully"
        
    catch e
        @error "Error stopping backup system" exception=e
    end
end

"""
Configure backup policies
"""
function configure_backup_policies(
    manager::BackupManagerService = BACKUP_MANAGER;
    enable_hourly::Bool = true,
    enable_daily::Bool = true,
    enable_weekly::Bool = true,
    enable_monthly::Bool = false,
    custom_policies::Vector{BackupPolicy} = BackupPolicy[]
)
    @info "Configuring backup policies..."
    
    # Clear existing policies
    empty!(manager.orchestrator.policies)
    
    # Add default policies based on configuration
    if enable_hourly
        hourly_policy = BackupPolicy("hourly_checkpoints", INCREMENTAL, "0 * * * *", 24)
        add_policy!(manager.orchestrator, hourly_policy)
        @info "Added hourly incremental backup policy"
    end
    
    if enable_daily
        daily_policy = BackupPolicy("daily_models", FULL, "0 2 * * *", 168)
        add_policy!(manager.orchestrator, daily_policy)
        @info "Added daily full backup policy"
    end
    
    if enable_weekly
        weekly_policy = BackupPolicy("weekly_archive", FULL, "0 3 * * 0", 672)
        add_policy!(manager.orchestrator, weekly_policy)
        @info "Added weekly archive backup policy"
    end
    
    if enable_monthly
        monthly_policy = BackupPolicy("monthly_archive", FULL, "0 4 1 * *", 2160)
        add_policy!(manager.orchestrator, monthly_policy)
        @info "Added monthly archive backup policy"
    end
    
    # Add custom policies
    for policy in custom_policies
        add_policy!(manager.orchestrator, policy)
        @info "Added custom backup policy: $(policy.name)"
    end
    
    @info "✅ Backup policies configured successfully ($(length(manager.orchestrator.policies)) policies)"
end

"""
Create manual backup
"""
function create_manual_backup(
    name::String,
    source_paths::Vector{String},
    backup_type::BackupType = FULL;
    manager::BackupManagerService = BACKUP_MANAGER
)::String
    @info "Creating manual backup: $name"
    
    # Generate backup ID
    backup_id = generate_backup_id("manual", backup_type)
    
    # Create temporary policy for manual backup
    temp_policy = BackupPolicy(
        "manual_$name",
        backup_type,
        "manual",
        24,  # 24 hour retention for manual backups
        compression_level = 6,
        storage_location = "s3://hsof-backups/manual/"
    )
    
    # Create backup metadata
    storage_path = joinpath(temp_policy.storage_location, backup_id)
    metadata = BackupMetadata(
        backup_id, backup_type, temp_policy.name,
        source_paths, storage_path,
        tags = Dict("manual" => "true", "name" => name, "created_by" => "user")
    )
    
    # Save metadata
    save_backup_metadata(manager.orchestrator, metadata)
    
    # Execute backup immediately
    try
        execute_backup(manager.orchestrator, backup_id)
        @info "✅ Manual backup created successfully: $backup_id"
        return backup_id
    catch e
        @error "Failed to create manual backup" backup_id=backup_id exception=e
        rethrow(e)
    end
end

"""
Restore from backup
"""
function restore_from_backup(
    backup_id::String,
    target_directory::String;
    restore_type::String = "full",
    selected_paths::Vector{String} = String[],
    verify_integrity::Bool = true,
    overwrite_existing::Bool = false,
    manager::BackupManagerService = BACKUP_MANAGER
)::RestoreResult
    
    @info "Restoring from backup" backup_id=backup_id target=target_directory type=restore_type
    
    request = RestoreRequest(
        backup_id, target_directory,
        restore_type = restore_type,
        selected_paths = selected_paths,
        verify_integrity = verify_integrity,
        overwrite_existing = overwrite_existing
    )
    
    return restore_backup(request)
end

"""
List available backups
"""
function list_backups(
    policy_filter::Union{String, Nothing} = nothing;
    manager::BackupManagerService = BACKUP_MANAGER
)::Vector{Dict{String, Any}}
    
    return list_available_backups(policy_filter)
end

"""
Get system status
"""
function get_system_status(manager::BackupManagerService = BACKUP_MANAGER)::Dict{String, Any}
    @info "Getting backup system status..."
    
    system_status = Dict{String, Any}(
        "timestamp" => now(),
        "system_running" => manager.running,
        "orchestrator_status" => get_backup_status(manager.orchestrator),
        "monitoring_status" => get_backup_health(manager.monitor),
        "storage_status" => get_storage_stats(manager.s3_client),
        "recent_backups" => get_recent_backups(now() - Hour(24))
    )
    
    return system_status
end

"""
Test backup system
"""
function test_backup_system(manager::BackupManagerService = BACKUP_MANAGER)::Dict{String, Any}
    @info "Testing backup system..."
    
    test_results = Dict{String, Any}(
        "timestamp" => now(),
        "tests_passed" => 0,
        "tests_failed" => 0,
        "test_results" => Dict{String, Any}()
    )
    
    # Test 1: Orchestrator functionality
    @info "Test 1: Testing orchestrator functionality..."
    try
        status = get_backup_status(manager.orchestrator)
        if status["running"]
            test_results["tests_passed"] += 1
            test_results["test_results"]["orchestrator"] = "✅ PASS"
        else
            test_results["tests_failed"] += 1
            test_results["test_results"]["orchestrator"] = "❌ FAIL - Not running"
        end
    catch e
        test_results["tests_failed"] += 1
        test_results["test_results"]["orchestrator"] = "❌ ERROR - $(string(e))"
    end
    
    # Test 2: Storage connectivity
    @info "Test 2: Testing S3 storage connectivity..."
    try
        if test_connection(manager.s3_client)
            test_results["tests_passed"] += 1
            test_results["test_results"]["storage"] = "✅ PASS"
        else
            test_results["tests_failed"] += 1
            test_results["test_results"]["storage"] = "❌ FAIL - Connection failed"
        end
    catch e
        test_results["tests_failed"] += 1
        test_results["test_results"]["storage"] = "❌ ERROR - $(string(e))"
    end
    
    # Test 3: Monitoring service
    @info "Test 3: Testing monitoring service..."
    try
        health = get_backup_health(manager.monitor)
        if health["status"] != "no_data"
            test_results["tests_passed"] += 1
            test_results["test_results"]["monitoring"] = "✅ PASS"
        else
            test_results["tests_failed"] += 1
            test_results["test_results"]["monitoring"] = "❌ FAIL - No monitoring data"
        end
    catch e
        test_results["tests_failed"] += 1
        test_results["test_results"]["monitoring"] = "❌ ERROR - $(string(e))"
    end
    
    # Test 4: Compression functionality
    @info "Test 4: Testing compression functionality..."
    try
        # Create a small test directory
        test_dir = mktempdir()
        test_file = joinpath(test_dir, "test.txt")
        write(test_file, "This is a test file for compression testing.")
        
        # Test compression
        compressed = compress_directory(test_dir, tempname())
        if compressed !== nothing && compressed.success
            test_results["tests_passed"] += 1
            test_results["test_results"]["compression"] = "✅ PASS"
            
            # Clean up
            rm(compressed.compressed_path, force=true)
        else
            test_results["tests_failed"] += 1
            test_results["test_results"]["compression"] = "❌ FAIL - Compression failed"
        end
        
        # Clean up test directory
        rm(test_dir, recursive=true, force=true)
        
    catch e
        test_results["tests_failed"] += 1
        test_results["test_results"]["compression"] = "❌ ERROR - $(string(e))"
    end
    
    # Test 5: Policy validation
    @info "Test 5: Testing policy validation..."
    try
        policy_count = length(manager.orchestrator.policies)
        if policy_count > 0
            test_results["tests_passed"] += 1
            test_results["test_results"]["policies"] = "✅ PASS - $policy_count policies configured"
        else
            test_results["tests_failed"] += 1
            test_results["test_results"]["policies"] = "❌ FAIL - No policies configured"
        end
    catch e
        test_results["tests_failed"] += 1
        test_results["test_results"]["policies"] = "❌ ERROR - $(string(e))"
    end
    
    test_results["total_tests"] = test_results["tests_passed"] + test_results["tests_failed"]
    test_results["success_rate"] = test_results["total_tests"] > 0 ? 
        test_results["tests_passed"] / test_results["total_tests"] : 0.0
    test_results["overall_status"] = test_results["tests_failed"] == 0 ? "PASS" : "FAIL"
    
    @info "Backup system test completed" status=test_results["overall_status"] passed=test_results["tests_passed"] failed=test_results["tests_failed"]
    
    return test_results
end

"""
Log system configuration
"""
function log_system_configuration(manager::BackupManagerService)
    @info "Backup System Configuration:" 
    @info "  Policies: $(length(manager.orchestrator.policies))"
    
    for policy in manager.orchestrator.policies
        @info "    - $(policy.name): $(policy.backup_type) every $(policy.schedule_cron), retain $(policy.retention_hours)h"
    end
    
    @info "  Max Concurrent Backups: $(manager.orchestrator.max_concurrent_backups)"
    @info "  Metadata Store: $(manager.orchestrator.metadata_store)"
    @info "  S3 Bucket: $(manager.s3_client.config.bucket)"
    @info "  S3 Region: $(manager.s3_client.config.region)"
    @info "  Monitoring Interval: $(manager.monitor.check_interval_seconds)s"
end

"""
Generate backup system report
"""
function generate_system_report(manager::BackupManagerService = BACKUP_MANAGER)::String
    status = get_system_status(manager)
    
    report = """
    # HSOF Backup System Report
    
    **Generated**: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    **System Status**: $(status["system_running"] ? "🟢 RUNNING" : "🔴 STOPPED")
    
    ## Orchestrator Status
    - Service Running: $(status["orchestrator_status"]["running"] ? "✅" : "❌")
    - Active Backups: $(status["orchestrator_status"]["active_backups"])
    - Queued Backups: $(status["orchestrator_status"]["queued_backups"])
    - Configured Policies: $(status["orchestrator_status"]["policies_count"])
    
    ## Storage Status
    - Storage Accessible: $(get(status["storage_status"], "storage_accessible", false) ? "✅" : "❌")
    - Total Backups: $(get(status["storage_status"], "total_backups", 0))
    - Storage Used: $(round(get(status["storage_status"], "total_size_bytes", 0) / 1024^3, digits=2)) GB
    - Bucket: $(get(status["storage_status"], "bucket", "unknown"))
    
    ## Monitoring Status
    - Health Status: $(get(status["monitoring_status"], "status", "unknown"))
    - Last Check: $(get(status["monitoring_status"], "timestamp", "never"))
    - Recent Success Rate: $(round(get(status["monitoring_status"], "recent_success_rate", 0.0) * 100, digits=1))%
    
    ## Recent Activity
    - Recent Backups (24h): $(length(status["recent_backups"]))
    """
    
    if !isempty(status["recent_backups"])
        report *= "\n    - Latest Backup: $(status["recent_backups"][1]["backup_id"]) ($(status["recent_backups"][1]["status"]))"
    end
    
    return report
end

"""
Cleanup old backups based on retention policies
"""
function cleanup_old_backups(manager::BackupManagerService = BACKUP_MANAGER)::Dict{String, Int}
    @info "Starting cleanup of old backups..."
    
    cleanup_results = Dict{String, Int}()
    
    for policy in manager.orchestrator.policies
        try
            @info "Cleaning up backups for policy: $(policy.name)"
            
            # Cleanup local backups
            local_cleaned = cleanup_local_backups(policy)
            
            # Cleanup S3 backups
            s3_cleaned = cleanup_old_backups(manager.s3_client, policy.retention_hours)
            
            cleanup_results[policy.name] = local_cleaned + s3_cleaned
            
            @info "Cleaned up $(cleanup_results[policy.name]) old backups for policy $(policy.name)"
            
        catch e
            @error "Error cleaning up backups for policy $(policy.name)" exception=e
            cleanup_results[policy.name] = 0
        end
    end
    
    total_cleaned = sum(values(cleanup_results))
    @info "✅ Cleanup completed: $total_cleaned total backups removed"
    
    return cleanup_results
end

"""
Cleanup local backups
"""
function cleanup_local_backups(policy::BackupPolicy)::Int
    cleaned_count = 0
    cutoff_time = now() - Hour(policy.retention_hours)
    
    metadata_store = "backups/metadata"
    if !isdir(metadata_store)
        return 0
    end
    
    for file in readdir(metadata_store)
        if endswith(file, ".json")
            try
                metadata_path = joinpath(metadata_store, file)
                json_content = read(metadata_path, String)
                data = JSON3.read(json_content)
                
                if data["policy_name"] == policy.name
                    backup_time = DateTime(data["timestamp"])
                    if backup_time < cutoff_time
                        # Remove metadata file
                        rm(metadata_path, force=true)
                        
                        # Remove backup data if it exists locally
                        local_backup_dir = joinpath("backups", "data", data["backup_id"])
                        if isdir(local_backup_dir)
                            rm(local_backup_dir, recursive=true, force=true)
                        end
                        
                        cleaned_count += 1
                        @debug "Removed old backup: $(data["backup_id"])"
                    end
                end
            catch e
                @debug "Error processing metadata file $file" exception=e
            end
        end
    end
    
    return cleaned_count
end

end  # module BackupManager