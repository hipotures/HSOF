module BackupMonitor

using Dates
using Logging
using JSON3

include("../backup_types.jl")
include("../orchestrator/backup_scheduler.jl")
include("../storage/s3_storage.jl")

using .BackupTypes
using .BackupScheduler
using .S3Storage

export BackupMonitoringService, start_monitoring, stop_monitoring, get_backup_health
export BackupAlert, AlertLevel, send_alert, check_backup_health

# Alert levels
@enum AlertLevel begin
    INFO = 1
    WARNING = 2
    CRITICAL = 3
end

# Backup alert structure
struct BackupAlert
    level::AlertLevel
    title::String
    message::String
    timestamp::DateTime
    backup_id::Union{String, Nothing}
    policy_name::Union{String, Nothing}
    
    function BackupAlert(
        level::AlertLevel,
        title::String,
        message::String;
        backup_id::Union{String, Nothing} = nothing,
        policy_name::Union{String, Nothing} = nothing
    )
        new(level, title, message, now(), backup_id, policy_name)
    end
end

# Monitoring service state
mutable struct BackupMonitoringService
    running::Bool
    monitor_task::Union{Task, Nothing}
    check_interval_seconds::Int
    alert_handlers::Vector{Function}
    health_history::Vector{Dict{String, Any}}
    max_history_size::Int
    thresholds::Dict{String, Any}
    
    function BackupMonitoringService(;
        check_interval_seconds::Int = 300,  # 5 minutes
        max_history_size::Int = 288  # 24 hours at 5min intervals
    )
        # Default monitoring thresholds
        thresholds = Dict{String, Any}(
            "max_backup_duration_hours" => 12,
            "max_failed_backups_per_day" => 3,
            "max_storage_usage_percent" => 85,
            "min_successful_backups_per_week" => 7,
            "max_backup_queue_size" => 10
        )
        
        new(false, nothing, check_interval_seconds, Function[], 
            Dict{String, Any}[], max_history_size, thresholds)
    end
end

# Global monitoring service
const MONITOR_SERVICE = BackupMonitoringService()

"""
Start backup monitoring service
"""
function start_monitoring(service::BackupMonitoringService = MONITOR_SERVICE)
    if service.running
        @warn "Backup monitoring is already running"
        return
    end
    
    service.running = true
    
    service.monitor_task = @async begin
        @info "Backup monitoring service started"
        
        while service.running
            try
                perform_health_check(service)
                sleep(service.check_interval_seconds)
            catch e
                @error "Error in backup monitoring loop" exception=e
                sleep(30)  # Brief pause before retry
            end
        end
        
        @info "Backup monitoring service stopped"
    end
end

"""
Stop backup monitoring service
"""
function stop_monitoring(service::BackupMonitoringService = MONITOR_SERVICE)
    if !service.running
        return
    end
    
    service.running = false
    
    if service.monitor_task !== nothing
        try
            wait(service.monitor_task)
        catch e
            @debug "Error waiting for monitoring task" exception=e
        end
        service.monitor_task = nothing
    end
    
    @info "Backup monitoring service stopped"
end

"""
Perform comprehensive backup health check
"""
function perform_health_check(service::BackupMonitoringService)
    @debug "Performing backup health check"
    
    health_data = Dict{String, Any}(
        "timestamp" => now(),
        "backup_service_status" => check_backup_service_health(),
        "storage_status" => check_storage_health(),
        "recent_backups" => check_recent_backup_status(),
        "retention_compliance" => check_retention_compliance(),
        "queue_status" => check_backup_queue_health()
    )
    
    # Add to history
    push!(service.health_history, health_data)
    
    # Maintain history size limit
    if length(service.health_history) > service.max_history_size
        splice!(service.health_history, 1:(length(service.health_history) - service.max_history_size))
    end
    
    # Check for alerts
    check_for_alerts(service, health_data)
end

"""
Check backup service health
"""
function check_backup_service_health()::Dict{String, Any}
    orchestrator_status = get_backup_status()
    
    return Dict{String, Any}(
        "service_running" => orchestrator_status["running"],
        "active_backups" => orchestrator_status["active_backups"],
        "queued_backups" => orchestrator_status["queued_backups"],
        "policies_configured" => orchestrator_status["policies_count"],
        "max_concurrent" => orchestrator_status["max_concurrent"]
    )
end

"""
Check storage health
"""
function check_storage_health()::Dict{String, Any}
    s3_client = S3Storage.DEFAULT_S3_CLIENT
    
    storage_accessible = test_connection(s3_client)
    storage_stats = storage_accessible ? get_storage_stats(s3_client) : Dict{String, Any}()
    
    return Dict{String, Any}(
        "storage_accessible" => storage_accessible,
        "total_backups" => get(storage_stats, "total_backups", 0),
        "total_size_bytes" => get(storage_stats, "total_size_bytes", 0),
        "last_backup" => get(storage_stats, "last_backup", nothing),
        "bucket" => get(storage_stats, "bucket", "unknown"),
        "region" => get(storage_stats, "region", "unknown")
    )
end

"""
Check recent backup status
"""
function check_recent_backup_status()::Dict{String, Any}
    # Get backup metadata from the last 24 hours
    cutoff_time = now() - Hour(24)
    
    recent_backups = get_recent_backups(cutoff_time)
    
    successful_count = count(b -> b["status"] == "COMPLETED", recent_backups)
    failed_count = count(b -> b["status"] == "FAILED", recent_backups)
    in_progress_count = count(b -> b["status"] == "IN_PROGRESS", recent_backups)
    
    return Dict{String, Any}(
        "total_recent_backups" => length(recent_backups),
        "successful_backups" => successful_count,
        "failed_backups" => failed_count,
        "in_progress_backups" => in_progress_count,
        "success_rate" => length(recent_backups) > 0 ? successful_count / length(recent_backups) : 1.0,
        "recent_backups" => recent_backups[1:min(5, length(recent_backups))]  # Last 5 backups
    )
end

"""
Get recent backup metadata
"""
function get_recent_backups(since::DateTime)::Vector{Dict{String, Any}}
    backups = Dict{String, Any}[]
    
    metadata_store = "backups/metadata"
    if !isdir(metadata_store)
        return backups
    end
    
    for file in readdir(metadata_store)
        if endswith(file, ".json")
            try
                metadata_path = joinpath(metadata_store, file)
                json_content = read(metadata_path, String)
                data = JSON3.read(json_content)
                
                backup_time = DateTime(data["timestamp"])
                if backup_time >= since
                    push!(backups, Dict{String, Any}(
                        "backup_id" => data["backup_id"],
                        "backup_type" => data["backup_type"],
                        "policy_name" => data["policy_name"],
                        "timestamp" => data["timestamp"],
                        "status" => data["status"],
                        "size_bytes" => data["size_bytes"],
                        "duration_seconds" => data["end_time"] !== nothing && data["start_time"] !== nothing ?
                            (DateTime(data["end_time"]) - DateTime(data["start_time"])).value / 1000 : nothing,
                        "error_message" => data["error_message"]
                    ))
                end
            catch e
                @debug "Error reading backup metadata file $file" exception=e
            end
        end
    end
    
    # Sort by timestamp, newest first
    sort!(backups, by = b -> b["timestamp"], rev = true)
    return backups
end

"""
Check retention policy compliance
"""
function check_retention_compliance()::Dict{String, Any}
    policies = ORCHESTRATOR.policies
    compliance_results = Dict{String, Any}()
    
    for policy in policies
        compliance_results[policy.name] = check_policy_retention_compliance(policy)
    end
    
    overall_compliant = all(r -> r["compliant"], values(compliance_results))
    
    return Dict{String, Any}(
        "overall_compliant" => overall_compliant,
        "policy_compliance" => compliance_results
    )
end

"""
Check retention compliance for a specific policy
"""
function check_policy_retention_compliance(policy::BackupPolicy)::Dict{String, Any}
    cutoff_time = now() - Hour(policy.retention_hours)
    
    # Get all backups for this policy
    policy_backups = get_backups_for_policy(policy.name)
    
    # Count backups within retention period
    recent_backups = filter(b -> DateTime(b["timestamp"]) >= cutoff_time, policy_backups)
    old_backups = filter(b -> DateTime(b["timestamp"]) < cutoff_time, policy_backups)
    
    # Check if we have expected number of backups
    expected_frequency = get_expected_backup_frequency(policy.schedule_cron)
    expected_count = max(1, div(policy.retention_hours, expected_frequency))
    
    compliant = length(recent_backups) >= expected_count * 0.8  # Allow 20% tolerance
    
    return Dict{String, Any}(
        "compliant" => compliant,
        "recent_backups_count" => length(recent_backups),
        "old_backups_count" => length(old_backups),
        "expected_count" => expected_count,
        "retention_hours" => policy.retention_hours
    )
end

"""
Get backups for a specific policy
"""
function get_backups_for_policy(policy_name::String)::Vector{Dict{String, Any}}
    backups = Dict{String, Any}[]
    
    metadata_store = "backups/metadata"
    if !isdir(metadata_store)
        return backups
    end
    
    for file in readdir(metadata_store)
        if endswith(file, ".json")
            try
                metadata_path = joinpath(metadata_store, file)
                json_content = read(metadata_path, String)
                data = JSON3.read(json_content)
                
                if data["policy_name"] == policy_name
                    push!(backups, Dict{String, Any}(
                        "backup_id" => data["backup_id"],
                        "timestamp" => data["timestamp"],
                        "status" => data["status"],
                        "size_bytes" => data["size_bytes"]
                    ))
                end
            catch e
                @debug "Error reading backup metadata file $file" exception=e
            end
        end
    end
    
    return backups
end

"""
Get expected backup frequency from cron expression (in hours)
"""
function get_expected_backup_frequency(cron_expr::String)::Int
    # Simplified cron parsing
    parts = split(cron_expr)
    if length(parts) >= 2
        minute, hour = parts[1:2]
        
        if hour == "*"
            return 1  # Hourly
        else
            return 24  # Daily
        end
    end
    
    return 24  # Default to daily
end

"""
Check backup queue health
"""
function check_backup_queue_health()::Dict{String, Any}
    orchestrator_status = get_backup_status()
    
    queue_size = orchestrator_status["queued_backups"]
    active_count = orchestrator_status["active_backups"]
    max_concurrent = orchestrator_status["max_concurrent"]
    
    queue_healthy = queue_size < 10  # Arbitrary threshold
    capacity_available = active_count < max_concurrent
    
    return Dict{String, Any}(
        "queue_size" => queue_size,
        "active_backups" => active_count,
        "max_concurrent" => max_concurrent,
        "queue_healthy" => queue_healthy,
        "capacity_available" => capacity_available
    )
end

"""
Check for alerts based on health data
"""
function check_for_alerts(service::BackupMonitoringService, health_data::Dict{String, Any})
    alerts = BackupAlert[]
    
    # Check backup service status
    service_status = health_data["backup_service_status"]
    if !service_status["service_running"]
        push!(alerts, BackupAlert(CRITICAL, "Backup Service Down", 
                                "Backup orchestrator service is not running"))
    end
    
    # Check storage accessibility
    storage_status = health_data["storage_status"]
    if !storage_status["storage_accessible"]
        push!(alerts, BackupAlert(CRITICAL, "Storage Inaccessible", 
                                "Cannot access backup storage"))
    end
    
    # Check recent backup failures
    recent_status = health_data["recent_backups"]
    if recent_status["failed_backups"] >= service.thresholds["max_failed_backups_per_day"]
        push!(alerts, BackupAlert(WARNING, "High Backup Failure Rate", 
                                "$(recent_status["failed_backups"]) backups failed in the last 24 hours"))
    end
    
    # Check queue size
    queue_status = health_data["queue_status"]
    if queue_status["queue_size"] >= service.thresholds["max_backup_queue_size"]
        push!(alerts, BackupAlert(WARNING, "Large Backup Queue", 
                                "$(queue_status["queue_size"]) backups waiting in queue"))
    end
    
    # Check retention compliance
    retention_status = health_data["retention_compliance"]
    if !retention_status["overall_compliant"]
        push!(alerts, BackupAlert(WARNING, "Retention Policy Violation", 
                                "One or more backup policies are not meeting retention requirements"))
    end
    
    # Send alerts
    for alert in alerts
        send_alert(service, alert)
    end
end

"""
Send alert through configured handlers
"""
function send_alert(service::BackupMonitoringService, alert::BackupAlert)
    @info "Backup Alert [$(alert.level)]: $(alert.title) - $(alert.message)"
    
    # Call all registered alert handlers
    for handler in service.alert_handlers
        try
            handler(alert)
        catch e
            @error "Error in alert handler" exception=e
        end
    end
end

"""
Add alert handler
"""
function add_alert_handler!(service::BackupMonitoringService, handler::Function)
    push!(service.alert_handlers, handler)
end

"""
Get current backup health summary
"""
function get_backup_health(service::BackupMonitoringService = MONITOR_SERVICE)::Dict{String, Any}
    if isempty(service.health_history)
        return Dict{String, Any}("status" => "no_data", "message" => "No health data available")
    end
    
    latest_health = service.health_history[end]
    
    # Determine overall health status
    service_running = latest_health["backup_service_status"]["service_running"]
    storage_accessible = latest_health["storage_status"]["storage_accessible"]
    recent_success_rate = latest_health["recent_backups"]["success_rate"]
    retention_compliant = latest_health["retention_compliance"]["overall_compliant"]
    
    overall_status = if !service_running || !storage_accessible
        "critical"
    elseif recent_success_rate < 0.8 || !retention_compliant
        "warning"
    else
        "healthy"
    end
    
    return Dict{String, Any}(
        "status" => overall_status,
        "timestamp" => latest_health["timestamp"],
        "service_running" => service_running,
        "storage_accessible" => storage_accessible,
        "recent_success_rate" => recent_success_rate,
        "retention_compliant" => retention_compliant,
        "active_backups" => latest_health["backup_service_status"]["active_backups"],
        "queued_backups" => latest_health["backup_service_status"]["queued_backups"],
        "total_backups_stored" => latest_health["storage_status"]["total_backups"],
        "last_backup" => latest_health["storage_status"]["last_backup"]
    )
end

"""
Get backup health history
"""
function get_health_history(service::BackupMonitoringService = MONITOR_SERVICE, hours::Int = 24)::Vector{Dict{String, Any}}
    cutoff_time = now() - Hour(hours)
    
    filtered_history = filter(h -> DateTime(h["timestamp"]) >= cutoff_time, service.health_history)
    return filtered_history
end

"""
Generate backup health report
"""
function generate_health_report(service::BackupMonitoringService = MONITOR_SERVICE)::String
    health = get_backup_health(service)
    
    report = """
    # HSOF Backup System Health Report
    
    **Generated**: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    **Overall Status**: $(uppercase(health["status"]))
    
    ## Service Status
    - Backup Service Running: $(health["service_running"] ? "✅" : "❌")
    - Storage Accessible: $(health["storage_accessible"] ? "✅" : "❌")
    - Active Backups: $(health["active_backups"])
    - Queued Backups: $(health["queued_backups"])
    
    ## Recent Performance
    - Success Rate (24h): $(round(health["recent_success_rate"] * 100, digits=1))%
    - Retention Compliant: $(health["retention_compliant"] ? "✅" : "❌")
    - Total Backups Stored: $(health["total_backups_stored"])
    - Last Backup: $(health["last_backup"])
    
    ## Recommendations
    """
    
    if health["status"] == "critical"
        report *= "\n- 🚨 **IMMEDIATE ACTION REQUIRED**: Critical backup system issues detected"
    elseif health["status"] == "warning"
        report *= "\n- ⚠️ **ATTENTION NEEDED**: Backup system has warnings that should be addressed"
    else
        report *= "\n- ✅ **SYSTEM HEALTHY**: Backup system is operating normally"
    end
    
    if health["recent_success_rate"] < 0.9
        report *= "\n- Consider investigating recent backup failures"
    end
    
    if health["queued_backups"] > 5
        report *= "\n- Consider increasing backup concurrency or investigating queue delays"
    end
    
    return report
end

# Default alert handlers
"""
Log alert handler
"""
function log_alert_handler(alert::BackupAlert)
    level_str = string(alert.level)
    @info "BACKUP_ALERT [$level_str] $(alert.title): $(alert.message)" backup_id=alert.backup_id policy=alert.policy_name
end

# Register default alert handler
add_alert_handler!(MONITOR_SERVICE, log_alert_handler)

end  # module BackupMonitor