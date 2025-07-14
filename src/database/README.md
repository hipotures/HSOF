# HSOF Database Integration Layer

This module provides comprehensive database functionality for the HSOF feature selection system, including connection pooling, lazy data loading, result persistence, and checkpoint management.

## Components

### Connection Pool (`connection_pool.jl`)
- Thread-safe SQLite connection pooling
- Read-only connections for data access
- Automatic health checking and recycling
- Configurable pool size and timeouts

### Metadata Parser (`metadata_parser.jl`)
- JSON metadata extraction from `dataset_metadata` table
- Caching with TTL for performance
- Column validation and type checking
- Support for excluded columns, ID columns, and target specification

### Data Loader (`data_loader.jl`)
- Lazy chunk-based data loading for large datasets
- Prefetch buffering for smooth streaming
- Adaptive chunk sizing based on memory limits
- Progress tracking integration
- Support for WHERE clauses and custom ordering

### Column Validator (`column_validator.jl`)
- Comprehensive column validation and type inference
- Missing value detection and handling strategies
- Statistical collection (min/max, mean, median, mode)
- Type consistency checking across chunks

### Progress Tracker (`progress_tracker.jl`)
- Real-time progress monitoring with ETA
- Throughput calculation (rows/sec, MB/sec)
- Memory usage tracking
- Flexible callback system
- Cancellation support

### Result Writer (`result_writer.jl`)
- Batch writing for MCTS results
- Asynchronous writing with queue buffering
- Feature importance tracking
- Transaction management with WAL mode
- Query functions for best results

### Checkpoint Manager (`checkpoint_manager.jl`)
- Zlib compression for space efficiency
- Incremental checkpointing support
- Retention policies (count and time-based)
- Export/import functionality
- SHA256 integrity verification

## Usage Examples

### Basic Data Loading
```julia
using HSOF.Database

# Create connection pool
pool = create_database_connection("features.db")

# Load dataset lazily with progress
iterator = load_dataset_lazy(pool, "feature_table", 
                           chunk_size=10_000,
                           show_progress=true)

# Process chunks
for chunk in iterator
    features = chunk.data[:, 3:end-1]  # Skip ID columns and target
    # Process features...
end

close_pool(pool)
```

### Result Persistence
```julia
# Create result writer
writer = ResultWriter(pool, batch_size=1000)

# Write MCTS results
result = MCTSResult(
    iteration,
    selected_features,
    feature_scores,
    objective_value,
    metadata,
    now()
)
write_results!(writer, result)

# Query best results
best = get_best_results(writer, 10)

close(writer)
```

### Checkpoint Management
```julia
# Create checkpoint manager
manager = CheckpointManager(pool, compression_level=6)

# Save checkpoint
checkpoint = Checkpoint(
    iteration,
    tree_state,
    feature_selection,
    scores,
    metadata,
    now()
)
save_checkpoint!(manager, checkpoint)

# Load latest checkpoint
latest = load_checkpoint(manager)
```

## Configuration

### Connection Pool Options
- `min_size`: Minimum connections (default: 5)
- `max_size`: Maximum connections (default: 20)
- `timeout_seconds`: Acquisition timeout (default: 30.0)
- `max_uses`: Uses before recycling (default: 100)
- `max_lifetime_hours`: Connection lifetime (default: 1.0)

### Data Loader Options
- `chunk_size`: Rows per chunk (default: 10,000)
- `prefetch_chunks`: Chunks to prefetch (default: 2)
- `memory_limit_gb`: Memory limit (default: 8.0)
- `adaptive_sizing`: Auto-adjust chunk size (default: true)

### Checkpoint Options
- `compression_level`: 0-9 (default: 6)
- `retention_count`: Checkpoints to keep (default: 10)
- `retention_days`: Days to retain (default: 7.0)
- `auto_checkpoint_interval`: Iterations between checkpoints (default: 10,000)
- `auto_checkpoint_time`: Minutes between checkpoints (default: 5.0)

## Performance Considerations

1. **Connection Pool Sizing**: Set pool size based on expected concurrency
2. **Chunk Size**: Balance between memory usage and I/O efficiency
3. **Compression Level**: Higher levels save space but increase CPU usage
4. **Batch Size**: Larger batches improve write performance but increase memory
5. **WAL Mode**: Enabled by default for better concurrent access

## Testing

Run all database tests:
```bash
julia test/database/runtests.jl
```

Run benchmarks:
```bash
julia benchmarks/database_benchmarks.jl
```

## Dependencies

- SQLite.jl: Database interface
- DataFrames.jl: Data manipulation
- JSON.jl: Metadata parsing
- CodecZlib.jl: Compression
- Serialization: Checkpoint storage