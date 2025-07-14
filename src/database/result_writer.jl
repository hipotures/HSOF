module ResultWriter

using SQLite
using DataFrames
using JSON
using Dates
using Base.Threads: SpinLock, lock, unlock

# Include dependencies
include("connection_pool.jl")
using .ConnectionPool

export ResultWriter, MCTSResult, FeatureImportance, write_results!, 
       create_result_tables, batch_write_results!, write_feature_importance!

"""
MCTS result structure
"""
struct MCTSResult
    iteration::Int
    selected_features::Vector{Int}
    feature_scores::Vector{Float64}
    objective_value::Float64
    metadata::Dict{String, Any}
    timestamp::DateTime
end

"""
Feature importance record
"""
struct FeatureImportance
    feature_index::Int
    feature_name::String
    importance_score::Float64
    selection_count::Int
    average_contribution::Float64
end

"""
Result writer with batching and transaction management
"""
mutable struct ResultWriter
    pool::SQLitePool
    table_prefix::String
    batch_size::Int
    
    # Batching
    result_buffer::Vector{MCTSResult}
    importance_buffer::Vector{FeatureImportance}
    
    # Write-ahead logging
    wal_enabled::Bool
    
    # Async writing
    write_queue::Channel{Union{MCTSResult, Vector{MCTSResult}}}
    importance_queue::Channel{Union{FeatureImportance, Vector{FeatureImportance}}}
    write_task::Union{Task, Nothing}
    
    # State
    total_results_written::Int
    last_flush_time::Float64
    flush_interval::Float64
    lock::SpinLock
    closed::Bool
end

"""
    ResultWriter(pool::SQLitePool; kwargs...)

Create a new result writer for MCTS outputs.

# Arguments
- `pool`: Database connection pool
- `table_prefix`: Prefix for result tables (default: "mcts")
- `batch_size`: Number of results to batch before writing (default: 1000)
- `flush_interval`: Maximum seconds between flushes (default: 5.0)
- `async_writing`: Enable asynchronous writing (default: true)
- `wal_enabled`: Enable write-ahead logging (default: true)
"""
function ResultWriter(
    pool::SQLitePool;
    table_prefix::String = "mcts",
    batch_size::Int = 1000,
    flush_interval::Float64 = 5.0,
    async_writing::Bool = true,
    wal_enabled::Bool = true
)
    writer = ResultWriter(
        pool,
        table_prefix,
        batch_size,
        Vector{MCTSResult}(),
        Vector{FeatureImportance}(),
        wal_enabled,
        Channel{Union{MCTSResult, Vector{MCTSResult}}}(100),
        Channel{Union{FeatureImportance, Vector{FeatureImportance}}}(100),
        nothing,
        0,
        time(),
        flush_interval,
        SpinLock(),
        false
    )
    
    # Create tables if needed
    create_result_tables(writer)
    
    # Start async writer if enabled
    if async_writing
        writer.write_task = start_async_writer(writer)
    end
    
    return writer
end

"""
Create result tables if they don't exist
"""
function create_result_tables(writer::ResultWriter)
    with_connection(writer.pool) do conn
        # Enable WAL mode if requested
        if writer.wal_enabled
            SQLite.execute(conn.db, "PRAGMA journal_mode = WAL")
            SQLite.execute(conn.db, "PRAGMA synchronous = NORMAL")
        end
        
        # Results table
        SQLite.execute(conn.db, """
            CREATE TABLE IF NOT EXISTS $(writer.table_prefix)_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER NOT NULL,
                selected_features TEXT NOT NULL,
                feature_scores TEXT NOT NULL,
                objective_value REAL NOT NULL,
                metadata TEXT,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices
        SQLite.execute(conn.db, """
            CREATE INDEX IF NOT EXISTS idx_$(writer.table_prefix)_results_iteration 
            ON $(writer.table_prefix)_results(iteration)
        """)
        
        SQLite.execute(conn.db, """
            CREATE INDEX IF NOT EXISTS idx_$(writer.table_prefix)_results_objective 
            ON $(writer.table_prefix)_results(objective_value DESC)
        """)
        
        # Feature importance table
        SQLite.execute(conn.db, """
            CREATE TABLE IF NOT EXISTS $(writer.table_prefix)_feature_importance (
                feature_index INTEGER PRIMARY KEY,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                selection_count INTEGER NOT NULL,
                average_contribution REAL NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Checkpoints table (created separately by checkpoint system)
        SQLite.execute(conn.db, """
            CREATE TABLE IF NOT EXISTS $(writer.table_prefix)_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER NOT NULL,
                checkpoint_data BLOB NOT NULL,
                compressed INTEGER DEFAULT 1,
                size_bytes INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        SQLite.execute(conn.db, """
            CREATE INDEX IF NOT EXISTS idx_$(writer.table_prefix)_checkpoints_iteration 
            ON $(writer.table_prefix)_checkpoints(iteration DESC)
        """)
    end
end

"""
    write_results!(writer::ResultWriter, result::MCTSResult)

Write a single MCTS result (may be batched).
"""
function write_results!(writer::ResultWriter, result::MCTSResult)
    if writer.closed
        throw(ErrorException("ResultWriter is closed"))
    end
    
    if !isnothing(writer.write_task)
        # Async mode - send to queue
        put!(writer.write_queue, result)
    else
        # Sync mode - add to buffer
        lock(writer.lock) do
            push!(writer.result_buffer, result)
            
            # Check if we should flush
            should_flush = length(writer.result_buffer) >= writer.batch_size ||
                          time() - writer.last_flush_time > writer.flush_interval
            
            if should_flush
                flush_results(writer)
            end
        end
    end
end

"""
    batch_write_results!(writer::ResultWriter, results::Vector{MCTSResult})

Write multiple results at once.
"""
function batch_write_results!(writer::ResultWriter, results::Vector{MCTSResult})
    if writer.closed
        throw(ErrorException("ResultWriter is closed"))
    end
    
    if !isnothing(writer.write_task)
        # Async mode
        put!(writer.write_queue, results)
    else
        # Sync mode
        lock(writer.lock) do
            append!(writer.result_buffer, results)
            
            if length(writer.result_buffer) >= writer.batch_size
                flush_results(writer)
            end
        end
    end
end

"""
Flush buffered results to database
"""
function flush_results(writer::ResultWriter)
    if isempty(writer.result_buffer)
        return
    end
    
    results_to_write = writer.result_buffer
    writer.result_buffer = Vector{MCTSResult}()
    writer.last_flush_time = time()
    
    with_connection(writer.pool) do conn
        # Begin transaction
        SQLite.transaction(conn.db) do
            stmt = SQLite.Stmt(conn.db, """
                INSERT INTO $(writer.table_prefix)_results 
                (iteration, selected_features, feature_scores, objective_value, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """)
            
            for result in results_to_write
                SQLite.execute(stmt, [
                    result.iteration,
                    JSON.json(result.selected_features),
                    JSON.json(result.feature_scores),
                    result.objective_value,
                    JSON.json(result.metadata),
                    string(result.timestamp)
                ])
            end
            
            SQLite.close(stmt)
        end
    end
    
    writer.total_results_written += length(results_to_write)
end

"""
    write_feature_importance!(writer::ResultWriter, importance::FeatureImportance)

Write or update feature importance.
"""
function write_feature_importance!(writer::ResultWriter, importance::FeatureImportance)
    if writer.closed
        throw(ErrorException("ResultWriter is closed"))
    end
    
    if !isnothing(writer.write_task)
        # Async mode
        put!(writer.importance_queue, importance)
    else
        # Sync mode
        lock(writer.lock) do
            push!(writer.importance_buffer, importance)
            
            # Feature importance updates less frequently
            if length(writer.importance_buffer) >= 100
                flush_importance(writer)
            end
        end
    end
end

"""
Flush feature importance updates
"""
function flush_importance(writer::ResultWriter)
    if isempty(writer.importance_buffer)
        return
    end
    
    importance_to_write = writer.importance_buffer
    writer.importance_buffer = Vector{FeatureImportance}()
    
    with_connection(writer.pool) do conn
        SQLite.transaction(conn.db) do
            stmt = SQLite.Stmt(conn.db, """
                INSERT OR REPLACE INTO $(writer.table_prefix)_feature_importance
                (feature_index, feature_name, importance_score, selection_count, average_contribution)
                VALUES (?, ?, ?, ?, ?)
            """)
            
            for imp in importance_to_write
                SQLite.execute(stmt, [
                    imp.feature_index,
                    imp.feature_name,
                    imp.importance_score,
                    imp.selection_count,
                    imp.average_contribution
                ])
            end
            
            SQLite.close(stmt)
        end
    end
end

"""
Start asynchronous writer task
"""
function start_async_writer(writer::ResultWriter)
    @async begin
        try
            while !writer.closed
                # Check for results
                result = nothing
                try
                    result = take!(writer.write_queue)
                catch e
                    if e isa InvalidStateException
                        break  # Channel closed
                    else
                        rethrow(e)
                    end
                end
                
                if !isnothing(result)
                    lock(writer.lock) do
                        if result isa MCTSResult
                            push!(writer.result_buffer, result)
                        else
                            append!(writer.result_buffer, result)
                        end
                        
                        if length(writer.result_buffer) >= writer.batch_size
                            flush_results(writer)
                        end
                    end
                end
                
                # Check for importance updates
                try
                    while isready(writer.importance_queue)
                        imp = take!(writer.importance_queue)
                        lock(writer.lock) do
                            if imp isa FeatureImportance
                                push!(writer.importance_buffer, imp)
                            else
                                append!(writer.importance_buffer, imp)
                            end
                        end
                    end
                    
                    lock(writer.lock) do
                        if length(writer.importance_buffer) >= 100
                            flush_importance(writer)
                        end
                    end
                catch
                    # Ignore channel errors
                end
                
                # Periodic flush
                lock(writer.lock) do
                    if time() - writer.last_flush_time > writer.flush_interval
                        flush_results(writer)
                        flush_importance(writer)
                    end
                end
                
                sleep(0.1)  # Small delay to prevent busy waiting
            end
        catch e
            @error "Async writer error" exception=e
        finally
            # Final flush
            lock(writer.lock) do
                flush_results(writer)
                flush_importance(writer)
            end
        end
    end
end

"""
Close the result writer and flush remaining data
"""
function Base.close(writer::ResultWriter)
    writer.closed = true
    
    # Close channels
    close(writer.write_queue)
    close(writer.importance_queue)
    
    # Wait for async writer to finish
    if !isnothing(writer.write_task)
        try
            wait(writer.write_task)
        catch
            # Ignore errors during shutdown
        end
    end
    
    # Final flush
    lock(writer.lock) do
        flush_results(writer)
        flush_importance(writer)
    end
end

"""
Get statistics about written results
"""
function get_write_statistics(writer::ResultWriter)
    lock(writer.lock) do
        return Dict(
            "total_results_written" => writer.total_results_written,
            "results_buffered" => length(writer.result_buffer),
            "importance_buffered" => length(writer.importance_buffer),
            "closed" => writer.closed
        )
    end
end

"""
Query best results from database
"""
function get_best_results(writer::ResultWriter, n::Int = 10)
    with_connection(writer.pool) do conn
        query = """
            SELECT * FROM $(writer.table_prefix)_results
            ORDER BY objective_value DESC
            LIMIT ?
        """
        
        return DBInterface.execute(conn.db, query, [n]) |> DataFrame
    end
end

"""
Get feature importance ranking
"""
function get_feature_importance(writer::ResultWriter)
    with_connection(writer.pool) do conn
        query = """
            SELECT * FROM $(writer.table_prefix)_feature_importance
            ORDER BY importance_score DESC
        """
        
        return DBInterface.execute(conn.db, query) |> DataFrame
    end
end

end # module