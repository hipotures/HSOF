module MetadataParser

using SQLite
using JSON
using DataFrames
using Dates
using Base: @kwdef

# Include connection pool
include("connection_pool.jl")
using .ConnectionPool

export DatasetMetadata, parse_metadata, MetadataCache, get_cached_metadata, clear_cache

"""
Dataset metadata structure
"""
@kwdef struct DatasetMetadata
    table_name::String
    excluded_columns::Vector{String} = String[]
    id_columns::Vector{String} = String[]
    target_column::Union{String, Nothing} = nothing
    feature_count::Int = 0
    row_count::Int = 0
    created_at::DateTime = now()
    metadata_version::String = "1.0"
    additional_info::Dict{String, Any} = Dict{String, Any}()
end

"""
Metadata cache entry
"""
mutable struct CacheEntry
    metadata::DatasetMetadata
    timestamp::DateTime
    access_count::Int
end

"""
Thread-safe metadata cache with TTL
"""
struct MetadataCache
    cache::Dict{String, CacheEntry}
    ttl_seconds::Float64
    lock::ReentrantLock
end

# Global cache instance
const METADATA_CACHE = MetadataCache(
    Dict{String, CacheEntry}(),
    300.0,  # 5 minutes TTL
    ReentrantLock()
)

"""
    parse_metadata(pool::SQLitePool, table_name::String) -> DatasetMetadata

Parse metadata from dataset_metadata table for a specific dataset.
"""
function parse_metadata(pool::SQLitePool, table_name::String)
    # Check cache first
    cached = get_cached_metadata(table_name)
    if !isnothing(cached)
        return cached
    end
    
    metadata = with_connection(pool) do conn
        # Query metadata table
        query = """
            SELECT 
                excluded_columns,
                id_columns,
                target_column,
                metadata_version,
                additional_info,
                created_at
            FROM dataset_metadata
            WHERE table_name = ?
            LIMIT 1
        """
        
        result = DBInterface.execute(conn.db, query, [table_name]) |> DataFrame
        
        if size(result, 1) == 0
            throw(ErrorException("No metadata found for table: $table_name"))
        end
        
        row = result[1, :]
        
        # Parse JSON fields
        excluded_columns = parse_json_array(row.excluded_columns, "excluded_columns")
        id_columns = parse_json_array(row.id_columns, "id_columns")
        target_column = parse_optional_string(row.target_column)
        
        # Parse additional info if present
        additional_info = if !ismissing(row.additional_info) && !isnothing(row.additional_info)
            try
                JSON.parse(row.additional_info)
            catch
                Dict{String, Any}()
            end
        else
            Dict{String, Any}()
        end
        
        # Get metadata version
        metadata_version = if !ismissing(row.metadata_version)
            string(row.metadata_version)
        else
            "1.0"
        end
        
        # Get table info
        feature_count, row_count = get_table_stats(conn.db, table_name, excluded_columns, id_columns, target_column)
        
        # Parse created_at timestamp
        created_at = if !ismissing(row.created_at)
            try
                DateTime(row.created_at)
            catch
                now()
            end
        else
            now()
        end
        
        DatasetMetadata(
            table_name = table_name,
            excluded_columns = excluded_columns,
            id_columns = id_columns,
            target_column = target_column,
            feature_count = feature_count,
            row_count = row_count,
            created_at = created_at,
            metadata_version = metadata_version,
            additional_info = additional_info
        )
    end
    
    # Cache the metadata
    cache_metadata(table_name, metadata)
    
    return metadata
end

"""
Parse JSON array field with validation
"""
function parse_json_array(value, field_name::String)
    if ismissing(value) || isnothing(value)
        return String[]
    end
    
    try
        parsed = JSON.parse(value)
        if !(parsed isa Vector)
            throw(ErrorException("$field_name must be a JSON array"))
        end
        return String[string(x) for x in parsed]
    catch e
        throw(ErrorException("Failed to parse $field_name: $(e.msg)"))
    end
end

"""
Parse optional string field
"""
function parse_optional_string(value)
    if ismissing(value) || isnothing(value) || value == ""
        return nothing
    end
    return string(value)
end

"""
Get table statistics (feature count and row count)
"""
function get_table_stats(db::SQLite.DB, table_name::String, 
                        excluded_cols::Vector{String}, 
                        id_cols::Vector{String}, 
                        target_col::Union{String, Nothing})
    # Get all columns
    table_info = DBInterface.execute(db, "PRAGMA table_info($table_name)") |> DataFrame
    all_columns = table_info.name
    
    # Calculate feature columns
    feature_columns = filter(col -> !(col in excluded_cols || col in id_cols || col == target_col), all_columns)
    feature_count = length(feature_columns)
    
    # Get row count with timeout
    row_count = try
        result = DBInterface.execute(db, "SELECT COUNT(*) as count FROM $table_name") |> DataFrame
        result.count[1]
    catch
        # If COUNT(*) is too slow, estimate from sqlite_stat1
        try
            result = DBInterface.execute(db, 
                "SELECT stat FROM sqlite_stat1 WHERE tbl = ?", [table_name]) |> DataFrame
            if size(result, 1) > 0
                # stat format is "rows indexes..."
                parse(Int, split(result.stat[1])[1])
            else
                -1  # Unknown
            end
        catch
            -1  # Unknown
        end
    end
    
    return feature_count, row_count
end

"""
    validate_metadata(metadata::DatasetMetadata, db::SQLite.DB)

Validate metadata against actual table structure.
"""
function validate_metadata(metadata::DatasetMetadata, db::SQLite.DB)
    # Get actual table columns
    table_info = DBInterface.execute(db, "PRAGMA table_info($(metadata.table_name))") |> DataFrame
    
    if size(table_info, 1) == 0
        throw(ErrorException("Table $(metadata.table_name) does not exist"))
    end
    
    actual_columns = Set(table_info.name)
    
    # Validate excluded columns exist
    for col in metadata.excluded_columns
        if !(col in actual_columns)
            throw(ErrorException("Excluded column '$col' does not exist in table"))
        end
    end
    
    # Validate id columns exist
    for col in metadata.id_columns
        if !(col in actual_columns)
            throw(ErrorException("ID column '$col' does not exist in table"))
        end
    end
    
    # Validate target column exists
    if !isnothing(metadata.target_column) && !(metadata.target_column in actual_columns)
        throw(ErrorException("Target column '$(metadata.target_column)' does not exist in table"))
    end
    
    return true
end

"""
Cache metadata with TTL
"""
function cache_metadata(table_name::String, metadata::DatasetMetadata)
    lock(METADATA_CACHE.lock) do
        METADATA_CACHE.cache[table_name] = CacheEntry(metadata, now(), 0)
    end
end

"""
    get_cached_metadata(table_name::String) -> Union{DatasetMetadata, Nothing}

Retrieve metadata from cache if valid.
"""
function get_cached_metadata(table_name::String)
    lock(METADATA_CACHE.lock) do
        if haskey(METADATA_CACHE.cache, table_name)
            entry = METADATA_CACHE.cache[table_name]
            
            # Check TTL
            age_seconds = (now() - entry.timestamp).value / 1000
            if age_seconds <= METADATA_CACHE.ttl_seconds
                entry.access_count += 1
                return entry.metadata
            else
                # Expired, remove from cache
                delete!(METADATA_CACHE.cache, table_name)
            end
        end
        
        return nothing
    end
end

"""
    clear_cache(table_name::Union{String, Nothing} = nothing)

Clear metadata cache for specific table or all tables.
"""
function clear_cache(table_name::Union{String, Nothing} = nothing)
    lock(METADATA_CACHE.lock) do
        if isnothing(table_name)
            empty!(METADATA_CACHE.cache)
        else
            delete!(METADATA_CACHE.cache, table_name)
        end
    end
end

"""
    get_feature_columns(metadata::DatasetMetadata, db::SQLite.DB) -> Vector{String}

Get list of feature columns based on metadata exclusions.
"""
function get_feature_columns(metadata::DatasetMetadata, db::SQLite.DB)
    # Get all columns
    table_info = DBInterface.execute(db, "PRAGMA table_info($(metadata.table_name))") |> DataFrame
    all_columns = String.(table_info.name)
    
    # Filter out excluded, id, and target columns
    excluded_set = Set([metadata.excluded_columns; metadata.id_columns])
    if !isnothing(metadata.target_column)
        push!(excluded_set, metadata.target_column)
    end
    
    return filter(col -> !(col in excluded_set), all_columns)
end

"""
    create_metadata_table(db::SQLite.DB)

Create the dataset_metadata table if it doesn't exist.
"""
function create_metadata_table(db::SQLite.DB)
    SQLite.execute(db, """
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            table_name TEXT PRIMARY KEY,
            excluded_columns TEXT NOT NULL DEFAULT '[]',
            id_columns TEXT NOT NULL DEFAULT '[]',
            target_column TEXT,
            metadata_version TEXT DEFAULT '1.0',
            additional_info TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
end

"""
    insert_metadata(db::SQLite.DB, metadata::DatasetMetadata)

Insert or update metadata for a dataset.
"""
function insert_metadata(db::SQLite.DB, metadata::DatasetMetadata)
    stmt = SQLite.Stmt(db, """
        INSERT OR REPLACE INTO dataset_metadata 
        (table_name, excluded_columns, id_columns, target_column, 
         metadata_version, additional_info, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """)
    
    SQLite.execute(stmt, [
        metadata.table_name,
        JSON.json(metadata.excluded_columns),
        JSON.json(metadata.id_columns),
        metadata.target_column,
        metadata.metadata_version,
        JSON.json(metadata.additional_info)
    ])
end

end # module