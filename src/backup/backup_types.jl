module BackupTypes

using Dates
using JSON3

export BackupType, BackupPolicy, BackupMetadata, BackupStatus, BackupResult

# Backup types enumeration
@enum BackupType begin
    FULL = 1
    INCREMENTAL = 2
    DIFFERENTIAL = 3
end

# Backup status enumeration
@enum BackupStatus begin
    PENDING = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5
end

# Backup policy configuration
struct BackupPolicy
    name::String
    backup_type::BackupType
    schedule_cron::String  # Cron expression for scheduling
    retention_hours::Int   # Hours to keep this backup type
    compression_level::Int # 0-9, 0=no compression, 9=max compression
    encryption_enabled::Bool
    storage_location::String
    max_concurrent_backups::Int
    
    function BackupPolicy(
        name::String,
        backup_type::BackupType,
        schedule_cron::String,
        retention_hours::Int;
        compression_level::Int = 6,
        encryption_enabled::Bool = true,
        storage_location::String = "s3://hsof-backups/",
        max_concurrent_backups::Int = 2
    )
        new(name, backup_type, schedule_cron, retention_hours, 
            compression_level, encryption_enabled, storage_location, max_concurrent_backups)
    end
end

# Backup metadata
mutable struct BackupMetadata
    backup_id::String
    backup_type::BackupType
    policy_name::String
    timestamp::DateTime
    source_paths::Vector{String}
    size_bytes::Int64
    compressed_size_bytes::Int64
    checksum::String
    encryption_key_id::Union{String, Nothing}
    storage_path::String
    parent_backup_id::Union{String, Nothing}  # For incremental backups
    status::BackupStatus
    start_time::DateTime
    end_time::Union{DateTime, Nothing}
    error_message::Union{String, Nothing}
    tags::Dict{String, String}
    
    function BackupMetadata(
        backup_id::String,
        backup_type::BackupType,
        policy_name::String,
        source_paths::Vector{String},
        storage_path::String;
        parent_backup_id::Union{String, Nothing} = nothing,
        tags::Dict{String, String} = Dict{String, String}()
    )
        new(backup_id, backup_type, policy_name, now(), source_paths,
            0, 0, "", nothing, storage_path, parent_backup_id,
            PENDING, now(), nothing, nothing, tags)
    end
end

# Backup operation result
struct BackupResult
    success::Bool
    backup_id::String
    size_bytes::Int64
    duration_seconds::Float64
    files_backed_up::Int
    files_skipped::Int
    error_message::Union{String, Nothing}
    warnings::Vector{String}
    
    function BackupResult(
        success::Bool,
        backup_id::String,
        size_bytes::Int64,
        duration_seconds::Float64;
        files_backed_up::Int = 0,
        files_skipped::Int = 0,
        error_message::Union{String, Nothing} = nothing,
        warnings::Vector{String} = String[]
    )
        new(success, backup_id, size_bytes, duration_seconds,
            files_backed_up, files_skipped, error_message, warnings)
    end
end

# Default backup policies for HSOF
const DEFAULT_POLICIES = [
    BackupPolicy("hourly_checkpoints", INCREMENTAL, "0 * * * *", 24),  # Every hour, keep 24
    BackupPolicy("daily_models", FULL, "0 2 * * *", 168),              # Daily at 2 AM, keep 7 days  
    BackupPolicy("weekly_archive", FULL, "0 3 * * 0", 672),            # Weekly on Sunday, keep 4 weeks
    BackupPolicy("monthly_archive", FULL, "0 4 1 * *", 2160)           # Monthly, keep 3 months
]

# Data categories for backup
const BACKUP_CATEGORIES = Dict(
    "models" => [
        "models/metamodel/",
        "models/stage1/",
        "models/stage2/", 
        "models/stage3/"
    ],
    "checkpoints" => [
        "checkpoints/mcts/",
        "checkpoints/pipeline/",
        "checkpoints/sessions/"
    ],
    "results" => [
        "results/feature_selection/",
        "results/benchmarks/",
        "results/experiments/"
    ],
    "configuration" => [
        "configs/",
        ".taskmaster/",
        "k8s/"
    ],
    "logs" => [
        "logs/application/",
        "logs/gpu/",
        "logs/performance/"
    ]
)

"""
Generate unique backup ID with timestamp and category
"""
function generate_backup_id(category::String, backup_type::BackupType)::String
    timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    type_str = lowercase(string(backup_type))
    return "$(category)-$(type_str)-$(timestamp)-$(randstring(6))"
end

"""
Convert backup metadata to JSON for storage
"""
function to_json(metadata::BackupMetadata)::String
    return JSON3.write(Dict(
        "backup_id" => metadata.backup_id,
        "backup_type" => string(metadata.backup_type),
        "policy_name" => metadata.policy_name,
        "timestamp" => string(metadata.timestamp),
        "source_paths" => metadata.source_paths,
        "size_bytes" => metadata.size_bytes,
        "compressed_size_bytes" => metadata.compressed_size_bytes,
        "checksum" => metadata.checksum,
        "encryption_key_id" => metadata.encryption_key_id,
        "storage_path" => metadata.storage_path,
        "parent_backup_id" => metadata.parent_backup_id,
        "status" => string(metadata.status),
        "start_time" => string(metadata.start_time),
        "end_time" => metadata.end_time !== nothing ? string(metadata.end_time) : nothing,
        "error_message" => metadata.error_message,
        "tags" => metadata.tags
    ))
end

"""
Create backup metadata from JSON
"""
function from_json(json_str::String)::BackupMetadata
    data = JSON3.read(json_str)
    
    metadata = BackupMetadata(
        data["backup_id"],
        BackupType(parse(Int, split(string(data["backup_type"]), "::")[end])),
        data["policy_name"],
        data["source_paths"],
        data["storage_path"],
        parent_backup_id = data["parent_backup_id"],
        tags = data["tags"]
    )
    
    metadata.timestamp = DateTime(data["timestamp"])
    metadata.size_bytes = data["size_bytes"]
    metadata.compressed_size_bytes = data["compressed_size_bytes"]
    metadata.checksum = data["checksum"]
    metadata.encryption_key_id = data["encryption_key_id"]
    metadata.status = BackupStatus(parse(Int, split(string(data["status"]), "::")[end]))
    metadata.start_time = DateTime(data["start_time"])
    
    if data["end_time"] !== nothing
        metadata.end_time = DateTime(data["end_time"])
    end
    
    metadata.error_message = data["error_message"]
    
    return metadata
end

end  # module BackupTypes