module DataLoader

using SQLite
using DataFrames
using Base: @kwdef
using Statistics

# Include dependencies
include("connection_pool.jl")
include("metadata_parser.jl")
using .ConnectionPool
using .MetadataParser

export ChunkIterator, DataChunk, LoaderConfig, create_chunk_iterator, 
       estimate_memory_usage, adaptive_chunk_size

"""
Configuration for data loading
"""
@kwdef struct LoaderConfig
    chunk_size::Int = 10_000
    prefetch_chunks::Int = 2
    memory_limit_gb::Float64 = 8.0
    adaptive_sizing::Bool = true
    order_by::Union{String, Nothing} = nothing
    where_clause::Union{String, Nothing} = nothing
end

"""
Data chunk with metadata
"""
struct DataChunk
    data::DataFrame
    chunk_index::Int
    total_chunks::Int
    row_offset::Int
    is_last::Bool
end

"""
Lazy chunk iterator for streaming large datasets
"""
mutable struct ChunkIterator
    pool::SQLitePool
    metadata::DatasetMetadata
    config::LoaderConfig
    feature_columns::Vector{String}
    total_rows::Int
    current_chunk::Int
    total_chunks::Int
    prefetch_buffer::Channel{DataChunk}
    prefetch_task::Union{Task, Nothing}
    closed::Bool
end

"""
    create_chunk_iterator(pool::SQLitePool, table_name::String; kwargs...) -> ChunkIterator

Create a lazy iterator for streaming data in chunks.
"""
function create_chunk_iterator(
    pool::SQLitePool, 
    table_name::String;
    chunk_size::Int = 10_000,
    prefetch_chunks::Int = 2,
    memory_limit_gb::Float64 = 8.0,
    adaptive_sizing::Bool = true,
    order_by::Union{String, Nothing} = nothing,
    where_clause::Union{String, Nothing} = nothing
)
    # Parse metadata
    metadata = parse_metadata(pool, table_name)
    
    # Get feature columns
    feature_columns = with_connection(pool) do conn
        MetadataParser.get_feature_columns(metadata, conn.db)
    end
    
    # Get total row count
    total_rows = get_row_count(pool, table_name, where_clause)
    
    # Adjust chunk size if adaptive
    if adaptive_sizing
        chunk_size = adaptive_chunk_size(
            length(feature_columns), 
            total_rows, 
            memory_limit_gb
        )
    end
    
    config = LoaderConfig(
        chunk_size = chunk_size,
        prefetch_chunks = prefetch_chunks,
        memory_limit_gb = memory_limit_gb,
        adaptive_sizing = adaptive_sizing,
        order_by = order_by,
        where_clause = where_clause
    )
    
    # Calculate total chunks
    total_chunks = cld(total_rows, chunk_size)
    
    # Create prefetch buffer
    buffer_size = min(prefetch_chunks, total_chunks)
    prefetch_buffer = Channel{DataChunk}(buffer_size)
    
    iterator = ChunkIterator(
        pool,
        metadata,
        config,
        feature_columns,
        total_rows,
        0,
        total_chunks,
        prefetch_buffer,
        nothing,
        false
    )
    
    # Start prefetching
    iterator.prefetch_task = start_prefetching(iterator)
    
    return iterator
end

"""
Get row count with optional WHERE clause
"""
function get_row_count(pool::SQLitePool, table_name::String, where_clause::Union{String, Nothing})
    with_connection(pool) do conn
        query = if isnothing(where_clause)
            "SELECT COUNT(*) as count FROM $table_name"
        else
            "SELECT COUNT(*) as count FROM $table_name WHERE $where_clause"
        end
        
        result = DBInterface.execute(conn.db, query) |> DataFrame
        return result.count[1]
    end
end

"""
Calculate adaptive chunk size based on available memory and data characteristics
"""
function adaptive_chunk_size(n_features::Int, n_rows::Int, memory_limit_gb::Float64)
    # Estimate bytes per row (8 bytes per Float64 + overhead)
    bytes_per_row = n_features * 8 + 100  # 100 bytes overhead
    
    # Target using 10% of memory limit per chunk
    target_memory = memory_limit_gb * 1e9 * 0.1
    
    # Calculate chunk size
    chunk_size = floor(Int, target_memory / bytes_per_row)
    
    # Apply bounds
    chunk_size = clamp(chunk_size, 1000, 100_000)
    
    # Don't make chunks smaller than necessary
    if chunk_size > n_rows
        chunk_size = n_rows
    end
    
    return chunk_size
end

"""
Estimate memory usage for a chunk
"""
function estimate_memory_usage(n_rows::Int, n_features::Int)
    # DataFrame overhead + data
    bytes_per_row = n_features * 8 + 100
    total_bytes = n_rows * bytes_per_row
    return total_bytes / 1e9  # Convert to GB
end

"""
Start background prefetching
"""
function start_prefetching(iterator::ChunkIterator)
    @async begin
        try
            chunk_index = 1
            while chunk_index <= iterator.total_chunks && !iterator.closed
                chunk = load_chunk(iterator, chunk_index)
                put!(iterator.prefetch_buffer, chunk)
                chunk_index += 1
            end
        catch e
            if !iterator.closed
                @error "Prefetch error" exception=e
            end
        finally
            close(iterator.prefetch_buffer)
        end
    end
end

"""
Load a single chunk from database
"""
function load_chunk(iterator::ChunkIterator, chunk_index::Int)
    with_connection(iterator.pool) do conn
        offset = (chunk_index - 1) * iterator.config.chunk_size
        
        # Build column list
        columns = []
        
        # Add ID columns if any
        if !isempty(iterator.metadata.id_columns)
            append!(columns, iterator.metadata.id_columns)
        end
        
        # Add feature columns
        append!(columns, iterator.feature_columns)
        
        # Add target column if exists
        if !isnothing(iterator.metadata.target_column)
            push!(columns, iterator.metadata.target_column)
        end
        
        column_list = join(columns, ", ")
        
        # Build query
        query = "SELECT $column_list FROM $(iterator.metadata.table_name)"
        
        # Add WHERE clause if specified
        if !isnothing(iterator.config.where_clause)
            query *= " WHERE $(iterator.config.where_clause)"
        end
        
        # Add ORDER BY for consistency
        if !isnothing(iterator.config.order_by)
            query *= " ORDER BY $(iterator.config.order_by)"
        elseif !isempty(iterator.metadata.id_columns)
            # Default to ordering by first ID column for consistency
            query *= " ORDER BY $(iterator.metadata.id_columns[1])"
        end
        
        # Add LIMIT and OFFSET
        query *= " LIMIT $(iterator.config.chunk_size) OFFSET $offset"
        
        # Execute query
        data = DBInterface.execute(conn.db, query) |> DataFrame
        
        # Create chunk
        is_last = chunk_index == iterator.total_chunks
        
        return DataChunk(
            data,
            chunk_index,
            iterator.total_chunks,
            offset,
            is_last
        )
    end
end

"""
Iterator interface implementation
"""
Base.iterate(iterator::ChunkIterator) = iterate(iterator, nothing)

function Base.iterate(iterator::ChunkIterator, state)
    if iterator.closed
        return nothing
    end
    
    # Try to get next chunk from buffer
    chunk = try
        take!(iterator.prefetch_buffer)
    catch e
        if e isa InvalidStateException
            # Channel closed, no more data
            return nothing
        else
            rethrow(e)
        end
    end
    
    iterator.current_chunk = chunk.chunk_index
    
    return (chunk, nothing)
end

Base.length(iterator::ChunkIterator) = iterator.total_chunks

Base.eltype(::Type{ChunkIterator}) = DataChunk

"""
Close the iterator and clean up resources
"""
function Base.close(iterator::ChunkIterator)
    iterator.closed = true
    
    # Close the channel
    close(iterator.prefetch_buffer)
    
    # Wait for prefetch task to complete
    if !isnothing(iterator.prefetch_task)
        try
            wait(iterator.prefetch_task)
        catch
            # Ignore errors during shutdown
        end
    end
end

"""
    stream_features(pool::SQLitePool, table_name::String, process_fn::Function; kwargs...)

Stream features from a table, applying a processing function to each chunk.

# Example
```julia
results = stream_features(pool, "features_table", chunk_size=5000) do chunk
    # Process chunk
    return mean(Matrix(chunk.data[:, 2:end]), dims=1)
end
```
"""
function stream_features(
    process_fn::Function,
    pool::SQLitePool, 
    table_name::String;
    kwargs...
)
    iterator = create_chunk_iterator(pool, table_name; kwargs...)
    results = []
    
    try
        for chunk in iterator
            result = process_fn(chunk)
            push!(results, result)
        end
    finally
        close(iterator)
    end
    
    return results
end

"""
    validate_chunk_integrity(chunks::Vector{DataChunk})

Validate that chunks cover all data without gaps or overlaps.
"""
function validate_chunk_integrity(chunks::Vector{DataChunk})
    if isempty(chunks)
        return true
    end
    
    # Check continuity
    expected_offset = 0
    for (i, chunk) in enumerate(chunks)
        if chunk.row_offset != expected_offset
            return false
        end
        expected_offset += size(chunk.data, 1)
        
        # Check chunk index
        if chunk.chunk_index != i
            return false
        end
    end
    
    # Check last chunk flag
    if !chunks[end].is_last
        return false
    end
    
    return true
end

"""
Get memory statistics for the current process
"""
function get_memory_stats()
    # Get RSS (Resident Set Size) in bytes
    pid = getpid()
    
    if Sys.islinux()
        try
            status = read("/proc/$pid/status", String)
            for line in split(status, '\n')
                if startswith(line, "VmRSS:")
                    # Extract value in kB and convert to GB
                    kb = parse(Int, split(line)[2])
                    return kb / 1e6  # Convert to GB
                end
            end
        catch
            return 0.0
        end
    end
    
    return 0.0
end

"""
Monitor memory usage during chunk processing
"""
struct MemoryMonitor
    initial_memory::Float64
    limit_gb::Float64
    
    function MemoryMonitor(limit_gb::Float64)
        new(get_memory_stats(), limit_gb)
    end
end

function check_memory(monitor::MemoryMonitor)
    current = get_memory_stats()
    used = current - monitor.initial_memory
    
    if used > monitor.limit_gb
        @warn "Memory usage exceeds limit" used_gb=used limit_gb=monitor.limit_gb
        return false
    end
    
    return true
end

end # module