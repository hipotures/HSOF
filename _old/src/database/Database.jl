module Database

# Export all database functionality

# Connection pooling
include("connection_pool.jl")
using .ConnectionPool
export SQLitePool, acquire_connection, release_connection, close_pool, pool_stats, with_connection

# Metadata parsing
include("metadata_parser.jl")
using .MetadataParser
export DatasetMetadata, parse_metadata, MetadataCache, get_cached_metadata, clear_cache,
       validate_metadata, get_feature_columns, create_metadata_table, insert_metadata

# Data loading
include("data_loader.jl") 
using .DataLoader
export ChunkIterator, DataChunk, LoaderConfig, create_chunk_iterator,
       estimate_memory_usage, adaptive_chunk_size, stream_features

# Column validation
include("column_validator.jl")
using .ColumnValidator
export ColumnInfo, ValidationResult, ValidationConfig, MissingValueStrategy,
       validate_columns, infer_column_types, check_type_consistency,
       get_column_statistics, handle_missing_values

# Progress tracking
include("progress_tracker.jl")
using .DBProgressTracker
export ProgressTracker, ProgressInfo, update_progress!, finish_progress!,
       estimate_eta, get_throughput, cancel_loading, format_progress,
       console_progress_callback, file_progress_callback, combined_progress_callback

# Result writing
include("result_writer.jl")
using .ResultWriter
export ResultWriter, MCTSResult, FeatureImportance, write_results!,
       create_result_tables, batch_write_results!, write_feature_importance!,
       get_write_statistics, get_best_results, get_feature_importance

# Checkpoint management
include("checkpoint_manager.jl")
using .CheckpointManager
export CheckpointManager, Checkpoint, save_checkpoint!, load_checkpoint,
       list_checkpoints, delete_old_checkpoints!, get_checkpoint_stats,
       save_incremental_checkpoint!, apply_retention_policy!,
       export_checkpoint, import_checkpoint!

# High-level convenience functions

"""
    create_database_connection(db_path::String; kwargs...) -> SQLitePool

Create a database connection pool with sensible defaults for HSOF.
"""
function create_database_connection(db_path::String; 
                                  min_size::Int = 5,
                                  max_size::Int = 20,
                                  kwargs...)
    return SQLitePool(db_path; min_size=min_size, max_size=max_size, kwargs...)
end

"""
    load_dataset_lazy(pool::SQLitePool, table_name::String; kwargs...) -> ChunkIterator

Create a lazy iterator for loading large datasets with progress tracking.
"""
function load_dataset_lazy(pool::SQLitePool, table_name::String;
                         show_progress::Bool = true,
                         chunk_size::Int = 10_000,
                         kwargs...)
    return create_chunk_iterator(pool, table_name; 
                               show_progress=show_progress,
                               chunk_size=chunk_size,
                               kwargs...)
end

"""
    validate_dataset(pool::SQLitePool, table_name::String) -> ValidationResult

Perform comprehensive validation of a dataset.
"""
function validate_dataset(pool::SQLitePool, table_name::String)
    metadata = parse_metadata(pool, table_name)
    return validate_columns(pool, metadata)
end

end # module