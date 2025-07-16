module BackupScheduler

using Dates
using Logging
# Custom cron-like scheduling implementation

include("../backup_types.jl")
using .BackupTypes

export BackupOrchestrator, start_scheduler, stop_scheduler, schedule_backup, execute_backup

# Default backup policies
const DEFAULT_POLICIES = BackupPolicy[]

# Default backup categories - what data to include in different backup types
const BACKUP_CATEGORIES = Dict{String, Vector{String}}(
    "models" => ["models/", "trained_models/", "*.jl", "*.toml"],
    "checkpoints" => ["checkpoints/", "cache/", "temp_results/"],
    "results" => ["results/", "outputs/", "reports/"],
    "configuration" => ["configs/", "*.toml", "*.json", "*.yaml"],
    "logs" => ["logs/", "*.log"]
)

# Backup orchestrator state
mutable struct BackupOrchestrator
    policies::Vector{BackupPolicy}
    active_backups::Dict{String, Task}
    scheduler_task::Union{Task, Nothing}
    running::Bool
    max_concurrent_backups::Int
    backup_queue::Vector{String}  # Queue of backup IDs waiting to execute
    metadata_store::String  # Path to metadata storage
    
    function BackupOrchestrator(;
        max_concurrent_backups::Int = 3,
        metadata_store::String = "backups/metadata"
    )
        new(copy(DEFAULT_POLICIES), Dict{String, Task}(), nothing, false,
            max_concurrent_backups, String[], metadata_store)
    end
end

# Global orchestrator instance
const ORCHESTRATOR = BackupOrchestrator()

"""
Add or update a backup policy
"""
function add_policy!(orchestrator::BackupOrchestrator, policy::BackupPolicy)
    # Remove existing policy with same name
    filter!(p -> p.name != policy.name, orchestrator.policies)
    push!(orchestrator.policies, policy)
    @info "Added backup policy: $(policy.name)"
end

"""
Start the backup scheduler
"""
function start_scheduler(orchestrator::BackupOrchestrator = ORCHESTRATOR)
    if orchestrator.running
        @warn "Backup scheduler is already running"
        return
    end
    
    orchestrator.running = true
    
    # Create metadata directory
    mkpath(orchestrator.metadata_store)
    
    orchestrator.scheduler_task = @async begin
        @info "Backup scheduler started"
        
        while orchestrator.running
            try
                check_scheduled_backups(orchestrator)
                process_backup_queue(orchestrator)
                cleanup_completed_backups(orchestrator)
                
                # Check every minute
                sleep(60)
            catch e
                @error "Error in backup scheduler loop" exception=e
                sleep(10)  # Brief pause before retry
            end
        end
        
        @info "Backup scheduler stopped"
    end
end

"""
Stop the backup scheduler
"""
function stop_scheduler(orchestrator::BackupOrchestrator = ORCHESTRATOR)
    if !orchestrator.running
        return
    end
    
    orchestrator.running = false
    
    # Cancel active backups
    for (backup_id, task) in orchestrator.active_backups
        @info "Cancelling active backup: $backup_id"
        try
            Base.throwto(task, InterruptException())
        catch e
            @debug "Error cancelling backup $backup_id" exception=e
        end
    end
    
    # Wait for scheduler task to complete
    if orchestrator.scheduler_task !== nothing
        try
            wait(orchestrator.scheduler_task)
        catch e
            @debug "Error waiting for scheduler task" exception=e
        end
        orchestrator.scheduler_task = nothing
    end
    
    empty!(orchestrator.active_backups)
    empty!(orchestrator.backup_queue)
    
    @info "Backup scheduler stopped and all active backups cancelled"
end

"""
Check for scheduled backups that need to run
"""
function check_scheduled_backups(orchestrator::BackupOrchestrator)
    current_time = now()
    
    for policy in orchestrator.policies
        try
            # Parse cron expression and check if backup should run
            if should_run_backup(policy.schedule_cron, current_time)
                @info "Scheduling backup for policy: $(policy.name)"
                schedule_backup(orchestrator, policy)
            end
        catch e
            @error "Error checking schedule for policy $(policy.name)" exception=e
        end
    end
end

"""
Check if backup should run based on cron schedule
"""
function should_run_backup(cron_expr::String, current_time::DateTime)::Bool
    # Simplified cron checking - in production would use proper cron library
    # For now, check basic patterns
    
    parts = split(cron_expr)
    if length(parts) != 5
        @warn "Invalid cron expression: $cron_expr"
        return false
    end
    
    minute, hour, day, month, weekday = parts
    
    # Check if current time matches cron expression
    current_minute = Dates.minute(current_time)
    current_hour = Dates.hour(current_time)
    current_day = Dates.day(current_time)
    current_month = Dates.month(current_time)
    current_weekday = Dates.dayofweek(current_time)
    
    # Simple matching logic (would be more sophisticated in production)
    return (minute == "*" || parse(Int, minute) == current_minute) &&
           (hour == "*" || parse(Int, hour) == current_hour) &&
           (day == "*" || parse(Int, day) == current_day) &&
           (month == "*" || parse(Int, month) == current_month)
end

"""
Schedule a backup based on policy
"""
function schedule_backup(orchestrator::BackupOrchestrator, policy::BackupPolicy)
    # Generate backup ID
    backup_id = generate_backup_id("scheduled", policy.backup_type)
    
    # Determine source paths based on backup type and policy
    source_paths = get_source_paths_for_policy(policy)
    
    # Create backup metadata
    storage_path = joinpath(policy.storage_location, backup_id)
    metadata = BackupMetadata(
        backup_id, policy.backup_type, policy.name,
        source_paths, storage_path,
        tags = Dict("scheduled" => "true", "policy" => policy.name)
    )
    
    # Save metadata
    save_backup_metadata(orchestrator, metadata)
    
    # Add to queue
    push!(orchestrator.backup_queue, backup_id)
    
    @info "Scheduled backup $backup_id for policy $(policy.name)"
end

"""
Get source paths for a backup policy
"""
function get_source_paths_for_policy(policy::BackupPolicy)::Vector{String}
    # Determine which data categories to backup based on policy name
    if contains(policy.name, "checkpoint")
        return BACKUP_CATEGORIES["checkpoints"]
    elseif contains(policy.name, "model")
        return vcat(BACKUP_CATEGORIES["models"], BACKUP_CATEGORIES["configuration"])
    elseif contains(policy.name, "archive")
        # Full archive includes everything except logs
        return vcat(
            BACKUP_CATEGORIES["models"],
            BACKUP_CATEGORIES["checkpoints"], 
            BACKUP_CATEGORIES["results"],
            BACKUP_CATEGORIES["configuration"]
        )
    else
        # Default to models and checkpoints
        return vcat(BACKUP_CATEGORIES["models"], BACKUP_CATEGORIES["checkpoints"])
    end
end

"""
Process backup queue and start backups if capacity allows
"""
function process_backup_queue(orchestrator::BackupOrchestrator)
    while !isempty(orchestrator.backup_queue) && 
          length(orchestrator.active_backups) < orchestrator.max_concurrent_backups
        
        backup_id = popfirst!(orchestrator.backup_queue)
        start_backup_execution(orchestrator, backup_id)
    end
end

"""
Start executing a backup
"""
function start_backup_execution(orchestrator::BackupOrchestrator, backup_id::String)
    @info "Starting backup execution: $backup_id"
    
    backup_task = @async begin
        try
            execute_backup(orchestrator, backup_id)
        catch e
            @error "Backup $backup_id failed" exception=e
            
            # Update metadata with failure
            metadata = load_backup_metadata(orchestrator, backup_id)
            if metadata !== nothing
                metadata.status = FAILED
                metadata.end_time = now()
                metadata.error_message = string(e)
                save_backup_metadata(orchestrator, metadata)
            end
        finally
            # Remove from active backups
            delete!(orchestrator.active_backups, backup_id)
        end
    end
    
    orchestrator.active_backups[backup_id] = backup_task
end

"""
Execute a backup operation
"""
function execute_backup(orchestrator::BackupOrchestrator, backup_id::String)
    metadata = load_backup_metadata(orchestrator, backup_id)
    if metadata === nothing
        @error "Cannot find metadata for backup: $backup_id"
        return
    end
    
    # Update status to in progress
    metadata.status = IN_PROGRESS
    metadata.start_time = now()
    save_backup_metadata(orchestrator, metadata)
    
    @info "Executing backup: $backup_id (type: $(metadata.backup_type))"
    
    # Find the policy for this backup
    policy = nothing
    for p in orchestrator.policies
        if p.name == metadata.policy_name
            policy = p
            break
        end
    end
    
    if policy === nothing
        throw(ErrorException("Cannot find policy $(metadata.policy_name) for backup $backup_id"))
    end
    
    # Perform the actual backup based on type
    if metadata.backup_type == FULL
        execute_full_backup(orchestrator, metadata, policy)
    elseif metadata.backup_type == INCREMENTAL
        execute_incremental_backup(orchestrator, metadata, policy)
    else
        throw(ErrorException("Unsupported backup type: $(metadata.backup_type)"))
    end
    
    # Update metadata on completion
    metadata.status = COMPLETED
    metadata.end_time = now()
    save_backup_metadata(orchestrator, metadata)
    
    @info "Backup completed successfully: $backup_id"
end

"""
Execute a full backup
"""
function execute_full_backup(orchestrator::BackupOrchestrator, metadata::BackupMetadata, policy::BackupPolicy)
    total_size = 0
    files_count = 0
    
    # Create backup directory
    backup_dir = joinpath("backups", "data", metadata.backup_id)
    mkpath(backup_dir)
    
    # Backup each source path
    for source_path in metadata.source_paths
        if isdir(source_path)
            @info "Backing up directory: $source_path"
            size, count = backup_directory(source_path, backup_dir, metadata.backup_id)
            total_size += size
            files_count += count
        elseif isfile(source_path)
            @info "Backing up file: $source_path"
            size = backup_file(source_path, backup_dir, metadata.backup_id)
            total_size += size
            files_count += 1
        else
            @warn "Source path does not exist: $source_path"
        end
    end
    
    # Update metadata with size information
    metadata.size_bytes = total_size
    
    # Compress backup if required
    if policy.compression_level > 0
        compressed_size = compress_backup(backup_dir, policy.compression_level)
        metadata.compressed_size_bytes = compressed_size
    else
        metadata.compressed_size_bytes = total_size
    end
    
    # Calculate checksum
    metadata.checksum = calculate_backup_checksum(backup_dir)
    
    @info "Full backup completed: $(files_count) files, $(round(total_size / 1024^2, digits=2)) MB"
end

"""
Execute an incremental backup
"""
function execute_incremental_backup(orchestrator::BackupOrchestrator, metadata::BackupMetadata, policy::BackupPolicy)
    # Find the most recent full backup for this policy
    parent_backup = find_latest_full_backup(orchestrator, policy.name)
    
    if parent_backup === nothing
        @warn "No full backup found for incremental backup, performing full backup instead"
        metadata.backup_type = FULL
        execute_full_backup(orchestrator, metadata, policy)
        return
    end
    
    metadata.parent_backup_id = parent_backup.backup_id
    
    @info "Performing incremental backup based on $(parent_backup.backup_id)"
    
    # Compare files with parent backup and only backup changed files
    total_size = 0
    files_count = 0
    
    backup_dir = joinpath("backups", "data", metadata.backup_id)
    mkpath(backup_dir)
    
    for source_path in metadata.source_paths
        if isdir(source_path)
            size, count = backup_directory_incremental(source_path, backup_dir, metadata.backup_id, parent_backup.timestamp)
            total_size += size
            files_count += count
        end
    end
    
    metadata.size_bytes = total_size
    metadata.compressed_size_bytes = total_size  # Simplified for incremental
    metadata.checksum = calculate_backup_checksum(backup_dir)
    
    @info "Incremental backup completed: $(files_count) files, $(round(total_size / 1024^2, digits=2)) MB"
end

"""
Backup a directory recursively
"""
function backup_directory(source_dir::String, backup_dir::String, backup_id::String)::Tuple{Int64, Int}
    total_size = 0
    file_count = 0
    
    if !isdir(source_dir)
        return (total_size, file_count)
    end
    
    for (root, dirs, files) in walkdir(source_dir)
        for file in files
            source_file = joinpath(root, file)
            
            # Create relative path for backup
            rel_path = relpath(source_file, source_dir)
            backup_file_path = joinpath(backup_dir, rel_path)
            
            # Create directory structure
            mkpath(dirname(backup_file_path))
            
            # Copy file
            try
                cp(source_file, backup_file_path)
                total_size += filesize(source_file)
                file_count += 1
            catch e
                @warn "Failed to backup file $source_file" exception=e
            end
        end
    end
    
    return (total_size, file_count)
end

"""
Backup a single file
"""
function backup_file(source_file::String, backup_dir::String, backup_id::String)::Int64
    if !isfile(source_file)
        return 0
    end
    
    backup_file_path = joinpath(backup_dir, basename(source_file))
    
    try
        cp(source_file, backup_file_path)
        return filesize(source_file)
    catch e
        @warn "Failed to backup file $source_file" exception=e
        return 0
    end
end

"""
Backup directory incrementally (only changed files)
"""
function backup_directory_incremental(source_dir::String, backup_dir::String, backup_id::String, since::DateTime)::Tuple{Int64, Int}
    total_size = 0
    file_count = 0
    
    if !isdir(source_dir)
        return (total_size, file_count)
    end
    
    for (root, dirs, files) in walkdir(source_dir)
        for file in files
            source_file = joinpath(root, file)
            
            # Check if file was modified since the last backup
            if stat(source_file).mtime > Dates.datetime2unix(since)
                rel_path = relpath(source_file, source_dir)
                backup_file_path = joinpath(backup_dir, rel_path)
                
                mkpath(dirname(backup_file_path))
                
                try
                    cp(source_file, backup_file_path)
                    total_size += filesize(source_file)
                    file_count += 1
                catch e
                    @warn "Failed to backup file $source_file" exception=e
                end
            end
        end
    end
    
    return (total_size, file_count)
end

"""
Compress backup directory
"""
function compress_backup(backup_dir::String, compression_level::Int)::Int64
    # Simplified compression - in production would use tar.gz or similar
    @info "Compressing backup directory: $backup_dir (level: $compression_level)"
    
    # For now, return original size (compression would be implemented here)
    total_size = 0
    for (root, dirs, files) in walkdir(backup_dir)
        for file in files
            total_size += filesize(joinpath(root, file))
        end
    end
    
    return total_size
end

"""
Calculate checksum for backup directory
"""
function calculate_backup_checksum(backup_dir::String)::String
    # Simplified checksum - in production would use SHA256 or similar
    return string(hash(backup_dir))
end

"""
Find the latest full backup for a policy
"""
function find_latest_full_backup(orchestrator::BackupOrchestrator, policy_name::String)::Union{BackupMetadata, Nothing}
    metadata_files = readdir(orchestrator.metadata_store, join=false)
    
    latest_backup = nothing
    latest_timestamp = DateTime(1900)
    
    for file in metadata_files
        if endswith(file, ".json")
            try
                metadata_path = joinpath(orchestrator.metadata_store, file)
                json_content = read(metadata_path, String)
                metadata = from_json(json_content)
                
                if metadata.policy_name == policy_name && 
                   metadata.backup_type == FULL &&
                   metadata.status == COMPLETED &&
                   metadata.timestamp > latest_timestamp
                    
                    latest_backup = metadata
                    latest_timestamp = metadata.timestamp
                end
            catch e
                @debug "Error reading metadata file $file" exception=e
            end
        end
    end
    
    return latest_backup
end

"""
Save backup metadata to storage
"""
function save_backup_metadata(orchestrator::BackupOrchestrator, metadata::BackupMetadata)
    metadata_file = joinpath(orchestrator.metadata_store, "$(metadata.backup_id).json")
    
    try
        write(metadata_file, to_json(metadata))
    catch e
        @error "Failed to save backup metadata for $(metadata.backup_id)" exception=e
    end
end

"""
Load backup metadata from storage
"""
function load_backup_metadata(orchestrator::BackupOrchestrator, backup_id::String)::Union{BackupMetadata, Nothing}
    metadata_file = joinpath(orchestrator.metadata_store, "$(backup_id).json")
    
    if !isfile(metadata_file)
        return nothing
    end
    
    try
        json_content = read(metadata_file, String)
        return from_json(json_content)
    catch e
        @error "Failed to load backup metadata for $backup_id" exception=e
        return nothing
    end
end

"""
Cleanup completed backup tasks
"""
function cleanup_completed_backups(orchestrator::BackupOrchestrator)
    completed_backups = String[]
    
    for (backup_id, task) in orchestrator.active_backups
        if istaskdone(task)
            push!(completed_backups, backup_id)
        end
    end
    
    for backup_id in completed_backups
        delete!(orchestrator.active_backups, backup_id)
        @debug "Cleaned up completed backup task: $backup_id"
    end
end

"""
Get status of all active backups
"""
function get_backup_status(orchestrator::BackupOrchestrator = ORCHESTRATOR)::Dict{String, Any}
    return Dict(
        "running" => orchestrator.running,
        "active_backups" => length(orchestrator.active_backups),
        "queued_backups" => length(orchestrator.backup_queue),
        "policies_count" => length(orchestrator.policies),
        "max_concurrent" => orchestrator.max_concurrent_backups
    )
end

end  # module BackupScheduler