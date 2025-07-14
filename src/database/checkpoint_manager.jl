module CheckpointManager

using SQLite
using DataFrames
using JSON
using Dates
using CodecZlib
using Serialization

# Include dependencies
include("connection_pool.jl")
include("result_writer.jl")
using .ConnectionPool
using .ResultWriter

export CheckpointManager, Checkpoint, save_checkpoint!, load_checkpoint,
       list_checkpoints, delete_old_checkpoints!, get_checkpoint_stats

"""
Checkpoint data structure
"""
struct Checkpoint
    iteration::Int
    tree_state::Any  # MCTS tree state
    feature_selection::Vector{Int}
    scores::Vector{Float64}
    metadata::Dict{String, Any}
    timestamp::DateTime
end

"""
Checkpoint manager with compression and retention
"""
mutable struct CheckpointManager
    pool::SQLitePool
    table_name::String
    compression_level::Int
    retention_count::Int
    retention_days::Float64
    auto_checkpoint_interval::Int
    auto_checkpoint_time::Float64
    last_checkpoint_iteration::Int
    last_checkpoint_time::Float64
end

"""
    CheckpointManager(pool::SQLitePool; kwargs...)

Create a checkpoint manager for MCTS state persistence.

# Arguments
- `pool`: Database connection pool
- `table_name`: Checkpoint table name (default: "mcts_checkpoints")
- `compression_level`: Zlib compression level 0-9 (default: 6)
- `retention_count`: Number of checkpoints to keep (default: 10)
- `retention_days`: Days to keep checkpoints (default: 7.0)
- `auto_checkpoint_interval`: Iterations between auto checkpoints (default: 10000)
- `auto_checkpoint_time`: Minutes between auto checkpoints (default: 5.0)
"""
function CheckpointManager(
    pool::SQLitePool;
    table_name::String = "mcts_checkpoints",
    compression_level::Int = 6,
    retention_count::Int = 10,
    retention_days::Float64 = 7.0,
    auto_checkpoint_interval::Int = 10_000,
    auto_checkpoint_time::Float64 = 5.0
)
    # Validate compression level
    compression_level = clamp(compression_level, 0, 9)
    
    manager = CheckpointManager(
        pool,
        table_name,
        compression_level,
        retention_count,
        retention_days,
        auto_checkpoint_interval,
        auto_checkpoint_time * 60,  # Convert to seconds
        0,
        time()
    )
    
    # Ensure table exists
    create_checkpoint_table(manager)
    
    return manager
end

"""
Create checkpoint table if it doesn't exist
"""
function create_checkpoint_table(manager::CheckpointManager)
    with_connection(manager.pool) do conn
        SQLite.execute(conn.db, """
            CREATE TABLE IF NOT EXISTS $(manager.table_name) (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER NOT NULL,
                checkpoint_data BLOB NOT NULL,
                compressed INTEGER DEFAULT 1,
                compression_ratio REAL,
                size_bytes INTEGER NOT NULL,
                original_size_bytes INTEGER NOT NULL,
                checksum TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices
        SQLite.execute(conn.db, """
            CREATE INDEX IF NOT EXISTS idx_$(manager.table_name)_iteration 
            ON $(manager.table_name)(iteration DESC)
        """)
        
        SQLite.execute(conn.db, """
            CREATE INDEX IF NOT EXISTS idx_$(manager.table_name)_timestamp 
            ON $(manager.table_name)(timestamp DESC)
        """)
    end
end

"""
    save_checkpoint!(manager::CheckpointManager, checkpoint::Checkpoint; force::Bool = false)

Save a checkpoint with compression.
"""
function save_checkpoint!(manager::CheckpointManager, checkpoint::Checkpoint; force::Bool = false)
    # Check if we should checkpoint
    if !force && !should_checkpoint(manager, checkpoint.iteration)
        return false
    end
    
    # Serialize checkpoint data
    io = IOBuffer()
    serialize(io, checkpoint)
    original_data = take!(io)
    original_size = length(original_data)
    
    # Compress if enabled
    compressed_data, compressed_size, compression_ratio = if manager.compression_level > 0
        compressed = transcode(ZlibCompressor(level=manager.compression_level), original_data)
        length(compressed), original_size / length(compressed)
    else
        original_data, original_size, 1.0
    end
    
    # Calculate checksum
    checksum = bytes2hex(sha256(compressed_data))
    
    # Save to database
    with_connection(manager.pool) do conn
        SQLite.execute(conn.db, """
            INSERT INTO $(manager.table_name)
            (iteration, checkpoint_data, compressed, compression_ratio, 
             size_bytes, original_size_bytes, checksum, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            checkpoint.iteration,
            compressed_data,
            manager.compression_level > 0 ? 1 : 0,
            compression_ratio,
            compressed_size,
            original_size,
            checksum,
            JSON.json(checkpoint.metadata),
            string(checkpoint.timestamp)
        ])
    end
    
    # Update tracking
    manager.last_checkpoint_iteration = checkpoint.iteration
    manager.last_checkpoint_time = time()
    
    # Clean up old checkpoints
    delete_old_checkpoints!(manager)
    
    return true
end

"""
Check if we should create a checkpoint
"""
function should_checkpoint(manager::CheckpointManager, iteration::Int)
    # Check iteration interval
    if iteration - manager.last_checkpoint_iteration >= manager.auto_checkpoint_interval
        return true
    end
    
    # Check time interval
    if time() - manager.last_checkpoint_time >= manager.auto_checkpoint_time
        return true
    end
    
    return false
end

"""
    load_checkpoint(manager::CheckpointManager, iteration::Union{Int, Nothing} = nothing)

Load a checkpoint. If iteration is not specified, loads the latest.
"""
function load_checkpoint(manager::CheckpointManager, iteration::Union{Int, Nothing} = nothing)
    checkpoint_row = with_connection(manager.pool) do conn
        if isnothing(iteration)
            # Load latest
            query = """
                SELECT * FROM $(manager.table_name)
                ORDER BY iteration DESC
                LIMIT 1
            """
            result = DBInterface.execute(conn.db, query) |> DataFrame
        else
            # Load specific iteration
            query = """
                SELECT * FROM $(manager.table_name)
                WHERE iteration = ?
                LIMIT 1
            """
            result = DBInterface.execute(conn.db, query, [iteration]) |> DataFrame
        end
        
        if isempty(result)
            return nothing
        end
        
        return result[1, :]
    end
    
    if isnothing(checkpoint_row)
        return nothing
    end
    
    # Verify checksum
    stored_checksum = checkpoint_row.checksum
    data_checksum = bytes2hex(sha256(checkpoint_row.checkpoint_data))
    
    if stored_checksum != data_checksum
        @warn "Checkpoint checksum mismatch" iteration=checkpoint_row.iteration
        return nothing
    end
    
    # Decompress if needed
    decompressed_data = if checkpoint_row.compressed == 1
        transcode(ZlibDecompressor(), checkpoint_row.checkpoint_data)
    else
        checkpoint_row.checkpoint_data
    end
    
    # Deserialize
    io = IOBuffer(decompressed_data)
    checkpoint = deserialize(io)
    
    return checkpoint
end

"""
    list_checkpoints(manager::CheckpointManager; limit::Int = 20)

List available checkpoints.
"""
function list_checkpoints(manager::CheckpointManager; limit::Int = 20)
    with_connection(manager.pool) do conn
        query = """
            SELECT 
                id,
                iteration,
                size_bytes,
                original_size_bytes,
                compression_ratio,
                timestamp,
                created_at
            FROM $(manager.table_name)
            ORDER BY iteration DESC
            LIMIT ?
        """
        
        return DBInterface.execute(conn.db, query, [limit]) |> DataFrame
    end
end

"""
    delete_old_checkpoints!(manager::CheckpointManager)

Delete old checkpoints based on retention policy.
"""
function delete_old_checkpoints!(manager::CheckpointManager)
    with_connection(manager.pool) do conn
        # Keep most recent N checkpoints
        SQLite.execute(conn.db, """
            DELETE FROM $(manager.table_name)
            WHERE id NOT IN (
                SELECT id FROM $(manager.table_name)
                ORDER BY iteration DESC
                LIMIT ?
            )
        """, [manager.retention_count])
        
        # Delete checkpoints older than retention days
        cutoff_date = now() - Day(manager.retention_days)
        SQLite.execute(conn.db, """
            DELETE FROM $(manager.table_name)
            WHERE timestamp < ?
        """, [string(cutoff_date)])
    end
end

"""
Get checkpoint statistics
"""
function get_checkpoint_stats(manager::CheckpointManager)
    with_connection(manager.pool) do conn
        stats_query = """
            SELECT 
                COUNT(*) as total_checkpoints,
                SUM(size_bytes) as total_size_bytes,
                SUM(original_size_bytes) as total_original_size_bytes,
                AVG(compression_ratio) as avg_compression_ratio,
                MIN(iteration) as min_iteration,
                MAX(iteration) as max_iteration,
                MIN(timestamp) as oldest_checkpoint,
                MAX(timestamp) as newest_checkpoint
            FROM $(manager.table_name)
        """
        
        stats = DBInterface.execute(conn.db, stats_query) |> DataFrame
        
        if isempty(stats) || stats.total_checkpoints[1] == 0
            return Dict(
                "total_checkpoints" => 0,
                "total_size_mb" => 0.0,
                "space_saved_mb" => 0.0,
                "avg_compression_ratio" => 0.0
            )
        end
        
        row = stats[1, :]
        
        return Dict(
            "total_checkpoints" => row.total_checkpoints,
            "total_size_mb" => row.total_size_bytes / 1e6,
            "total_original_size_mb" => row.total_original_size_bytes / 1e6,
            "space_saved_mb" => (row.total_original_size_bytes - row.total_size_bytes) / 1e6,
            "avg_compression_ratio" => row.avg_compression_ratio,
            "iteration_range" => (row.min_iteration, row.max_iteration),
            "time_range" => (row.oldest_checkpoint, row.newest_checkpoint)
        )
    end
end

"""
Create incremental checkpoint (delta from previous)
"""
function save_incremental_checkpoint!(manager::CheckpointManager, 
                                    checkpoint::Checkpoint,
                                    base_iteration::Int)
    # Load base checkpoint
    base = load_checkpoint(manager, base_iteration)
    
    if isnothing(base)
        # Fall back to full checkpoint
        return save_checkpoint!(manager, checkpoint)
    end
    
    # Create delta checkpoint
    delta = create_delta_checkpoint(base, checkpoint)
    
    # Add metadata about base
    delta.metadata["base_iteration"] = base_iteration
    delta.metadata["is_incremental"] = true
    
    # Save delta
    return save_checkpoint!(manager, delta, force=true)
end

"""
Create a delta between two checkpoints
"""
function create_delta_checkpoint(base::Checkpoint, current::Checkpoint)
    # This is a simplified version - real implementation would
    # compute actual differences in tree structure
    
    delta_metadata = copy(current.metadata)
    delta_metadata["delta_from"] = base.iteration
    
    # For now, just store the full state
    # A real implementation would store only changes
    return Checkpoint(
        current.iteration,
        current.tree_state,
        current.feature_selection,
        current.scores,
        delta_metadata,
        current.timestamp
    )
end

"""
Apply retention policy manually
"""
function apply_retention_policy!(manager::CheckpointManager)
    delete_old_checkpoints!(manager)
    
    # Get current stats
    stats = get_checkpoint_stats(manager)
    
    @info "Checkpoint retention applied" stats
end

"""
Export checkpoint to file
"""
function export_checkpoint(manager::CheckpointManager, 
                         iteration::Int,
                         filepath::String)
    checkpoint = load_checkpoint(manager, iteration)
    
    if isnothing(checkpoint)
        throw(ErrorException("Checkpoint not found for iteration $iteration"))
    end
    
    # Save to file
    open(filepath, "w") do f
        serialize(f, checkpoint)
    end
    
    # Also save metadata
    metadata_path = replace(filepath, r"\.[^.]+$" => "_metadata.json")
    open(metadata_path, "w") do f
        JSON.print(f, Dict(
            "iteration" => checkpoint.iteration,
            "timestamp" => string(checkpoint.timestamp),
            "metadata" => checkpoint.metadata,
            "feature_count" => length(checkpoint.feature_selection),
            "file_size" => filesize(filepath)
        ), 4)
    end
    
    return filepath, metadata_path
end

"""
Import checkpoint from file
"""
function import_checkpoint!(manager::CheckpointManager, filepath::String)
    # Load from file
    checkpoint = open(filepath, "r") do f
        deserialize(f)
    end
    
    # Save to database
    save_checkpoint!(manager, checkpoint, force=true)
    
    return checkpoint.iteration
end

# SHA256 implementation placeholder (would use a proper crypto library)
function sha256(data::Vector{UInt8})
    # This is a placeholder - use a real SHA256 implementation
    return UInt8[x ⊻ 0x5A for x in data[1:min(32, length(data))]]
end

end # module