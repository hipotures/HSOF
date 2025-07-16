"""
SQLite Result Writing Module for Stage 2 GPU-MCTS Ensemble

This module implements a comprehensive result persistence layer that:
- Creates result schema with tables for final features, ensemble metrics, and tree statistics
- Provides atomic write operations ensuring database consistency
- Supports result versioning for multiple runs on same dataset
- Stores metadata capturing all ensemble configuration parameters
- Validates feature IDs match original dataset
"""

using SQLite
using DataFrames
using JSON3
using Dates
using UUIDs
using Logging
using Statistics

# Result Schema Definitions
const SCHEMA_VERSION = "1.0.0"

const CREATE_TABLES_SQL = [
    # Results versioning and metadata
    """CREATE TABLE IF NOT EXISTS result_runs (
        run_id TEXT PRIMARY KEY,
        dataset_name TEXT NOT NULL,
        run_timestamp DATETIME NOT NULL,
        schema_version TEXT NOT NULL,
        completion_status TEXT NOT NULL,
        total_features_input INTEGER NOT NULL,
        total_features_output INTEGER NOT NULL,
        execution_time_seconds REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    
    # Ensemble configuration parameters
    """CREATE TABLE IF NOT EXISTS ensemble_config (
        run_id TEXT NOT NULL,
        config_key TEXT NOT NULL,
        config_value TEXT NOT NULL,
        config_type TEXT NOT NULL,
        PRIMARY KEY (run_id, config_key),
        FOREIGN KEY (run_id) REFERENCES result_runs(run_id)
    )""",
    
    # Final selected features
    """CREATE TABLE IF NOT EXISTS selected_features (
        run_id TEXT NOT NULL,
        feature_id INTEGER NOT NULL,
        feature_name TEXT NOT NULL,
        original_feature_id INTEGER NOT NULL,
        selection_rank INTEGER NOT NULL,
        selection_score REAL NOT NULL,
        selection_confidence REAL NOT NULL,
        selection_frequency REAL NOT NULL,
        feature_importance REAL,
        PRIMARY KEY (run_id, feature_id),
        FOREIGN KEY (run_id) REFERENCES result_runs(run_id)
    )""",
    
    # Ensemble-level metrics and statistics
    """CREATE TABLE IF NOT EXISTS ensemble_metrics (
        run_id TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        metric_type TEXT NOT NULL,
        measurement_timestamp DATETIME NOT NULL,
        PRIMARY KEY (run_id, metric_name),
        FOREIGN KEY (run_id) REFERENCES result_runs(run_id)
    )""",
    
    # Individual tree statistics
    """CREATE TABLE IF NOT EXISTS tree_statistics (
        run_id TEXT NOT NULL,
        tree_id INTEGER NOT NULL,
        gpu_id INTEGER NOT NULL,
        total_iterations INTEGER NOT NULL,
        max_depth INTEGER NOT NULL,
        final_score REAL NOT NULL,
        convergence_iteration INTEGER,
        exploration_ratio REAL NOT NULL,
        unique_features_explored INTEGER NOT NULL,
        execution_time_seconds REAL NOT NULL,
        PRIMARY KEY (run_id, tree_id),
        FOREIGN KEY (run_id) REFERENCES result_runs(run_id)
    )""",
    
    # Feature selection history during consensus building
    """CREATE TABLE IF NOT EXISTS consensus_history (
        run_id TEXT NOT NULL,
        iteration INTEGER NOT NULL,
        feature_id INTEGER NOT NULL,
        vote_count INTEGER NOT NULL,
        weighted_score REAL NOT NULL,
        selection_probability REAL NOT NULL,
        timestamp DATETIME NOT NULL,
        PRIMARY KEY (run_id, iteration, feature_id),
        FOREIGN KEY (run_id) REFERENCES result_runs(run_id)
    )""",
    
    # Performance monitoring data
    """CREATE TABLE IF NOT EXISTS performance_metrics (
        run_id TEXT NOT NULL,
        metric_timestamp DATETIME NOT NULL,
        gpu_id INTEGER NOT NULL,
        gpu_utilization REAL NOT NULL,
        gpu_memory_used REAL NOT NULL,
        gpu_memory_total REAL NOT NULL,
        gpu_temperature REAL NOT NULL,
        throughput_trees_per_second REAL NOT NULL,
        PRIMARY KEY (run_id, metric_timestamp, gpu_id),
        FOREIGN KEY (run_id) REFERENCES result_runs(run_id)
    )""",
    
    # Feature validation and integrity checks
    """CREATE TABLE IF NOT EXISTS feature_validation (
        run_id TEXT NOT NULL,
        original_feature_count INTEGER NOT NULL,
        validated_feature_count INTEGER NOT NULL,
        missing_features TEXT,
        duplicate_features TEXT,
        invalid_features TEXT,
        validation_status TEXT NOT NULL,
        validation_timestamp DATETIME NOT NULL,
        PRIMARY KEY (run_id),
        FOREIGN KEY (run_id) REFERENCES result_runs(run_id)
    )""",
    
    # Indexes for performance
    """CREATE INDEX IF NOT EXISTS idx_result_runs_dataset ON result_runs(dataset_name)""",
    """CREATE INDEX IF NOT EXISTS idx_result_runs_timestamp ON result_runs(run_timestamp)""",
    """CREATE INDEX IF NOT EXISTS idx_selected_features_rank ON selected_features(selection_rank)""",
    """CREATE INDEX IF NOT EXISTS idx_consensus_history_iteration ON consensus_history(iteration)""",
    """CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(metric_timestamp)"""
]

# Result Status
@enum ResultStatus begin
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5
end

# Configuration types
@enum ConfigType begin
    INTEGER = 1
    FLOAT = 2
    STRING = 3
    BOOLEAN = 4
    ARRAY = 5
end

# Selected feature information
struct SelectedFeature
    feature_id::Int
    feature_name::String
    original_feature_id::Int
    selection_rank::Int
    selection_score::Float64
    selection_confidence::Float64
    selection_frequency::Float64
    feature_importance::Union{Float64, Nothing}
end

# Ensemble metrics
struct EnsembleMetric
    metric_name::String
    metric_value::Float64
    metric_type::String
    measurement_timestamp::DateTime
end

# Tree statistics
struct TreeStatistic
    tree_id::Int
    gpu_id::Int
    total_iterations::Int
    max_depth::Int
    final_score::Float64
    convergence_iteration::Union{Int, Nothing}
    exploration_ratio::Float64
    unique_features_explored::Int
    execution_time_seconds::Float64
end

# Consensus history entry
struct ConsensusEntry
    iteration::Int
    feature_id::Int
    vote_count::Int
    weighted_score::Float64
    selection_probability::Float64
    timestamp::DateTime
end

# Performance metrics
struct PerformanceMetric
    metric_timestamp::DateTime
    gpu_id::Int
    gpu_utilization::Float64
    gpu_memory_used::Float64
    gpu_memory_total::Float64
    gpu_temperature::Float64
    throughput_trees_per_second::Float64
end

# Feature validation result
struct FeatureValidation
    original_feature_count::Int
    validated_feature_count::Int
    missing_features::Vector{Int}
    duplicate_features::Vector{Int}
    invalid_features::Vector{Int}
    validation_status::String
    validation_timestamp::DateTime
end

# Main SQLite result writer
mutable struct SQLiteResultWriter
    db_path::String
    db_connection::SQLite.DB
    current_run_id::Union{String, Nothing}
    dataset_name::String
    run_timestamp::DateTime
    feature_cache::Dict{Int, String}
    transaction_active::Bool
    write_lock::ReentrantLock
    
    function SQLiteResultWriter(db_path::String, dataset_name::String)
        connection = SQLite.DB(db_path)
        
        # Create tables if they don't exist - execute each statement separately
        for statement in CREATE_TABLES_SQL
            try
                SQLite.execute(connection, statement)
            catch e
                @warn "Failed to execute SQL statement: $statement"
                @warn "Error: $e"
            end
        end
        
        new(
            db_path,
            connection,
            nothing,
            dataset_name,
            now(),
            Dict{Int, String}(),
            false,
            ReentrantLock()
        )
    end
end

# Initialize new result run
function initialize_run!(writer::SQLiteResultWriter, config::Dict{String, Any})
    lock(writer.write_lock) do
        # Generate unique run ID
        run_id = string(uuid4())
        writer.current_run_id = run_id
        writer.run_timestamp = now()
        
        # Insert run metadata
        SQLite.execute(writer.db_connection, """
            INSERT INTO result_runs (
                run_id, dataset_name, run_timestamp, schema_version, 
                completion_status, total_features_input, total_features_output
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id,
            writer.dataset_name,
            writer.run_timestamp,
            SCHEMA_VERSION,
            "RUNNING",
            get(config, "total_features_input", 0),
            0  # Will be updated when features are written
        ])
        
        # Store configuration parameters
        store_configuration!(writer, config)
        
        @info "Initialized result run $run_id for dataset $(writer.dataset_name)"
        return run_id
    end
end

# Store ensemble configuration
function store_configuration!(writer::SQLiteResultWriter, config::Dict{String, Any})
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    run_id = writer.current_run_id
    
    # Insert configuration parameters
    for (key, value) in config
        config_type, config_value = serialize_config_value(value)
        
        SQLite.execute(writer.db_connection, """
            INSERT OR REPLACE INTO ensemble_config (
                run_id, config_key, config_value, config_type
            ) VALUES (?, ?, ?, ?)
        """, [run_id, key, config_value, config_type])
    end
    
    @debug "Stored $(length(config)) configuration parameters"
end

# Serialize configuration value
function serialize_config_value(value::Any)::Tuple{String, String}
    if isa(value, Integer)
        return ("INTEGER", string(value))
    elseif isa(value, AbstractFloat)
        return ("FLOAT", string(value))
    elseif isa(value, Bool)
        return ("BOOLEAN", string(value))
    elseif isa(value, AbstractArray)
        return ("ARRAY", JSON3.write(value))
    else
        return ("STRING", string(value))
    end
end

# Write selected features (atomic operation)
function write_selected_features!(writer::SQLiteResultWriter, features::Vector{SelectedFeature})
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    lock(writer.write_lock) do
        run_id = writer.current_run_id
        
        # Begin transaction for atomic write
        SQLite.execute(writer.db_connection, "BEGIN TRANSACTION")
        writer.transaction_active = true
        
        try
            # Clear existing features for this run
            SQLite.execute(writer.db_connection, 
                         "DELETE FROM selected_features WHERE run_id = ?", 
                         [run_id])
            
            # Insert new features
            for feature in features
                SQLite.execute(writer.db_connection, """
                    INSERT INTO selected_features (
                        run_id, feature_id, feature_name, original_feature_id,
                        selection_rank, selection_score, selection_confidence,
                        selection_frequency, feature_importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    feature.feature_id,
                    feature.feature_name,
                    feature.original_feature_id,
                    feature.selection_rank,
                    feature.selection_score,
                    feature.selection_confidence,
                    feature.selection_frequency,
                    feature.feature_importance
                ])
            end
            
            # Update run with feature count
            SQLite.execute(writer.db_connection, """
                UPDATE result_runs 
                SET total_features_output = ? 
                WHERE run_id = ?
            """, [length(features), run_id])
            
            # Commit transaction
            SQLite.execute(writer.db_connection, "COMMIT")
            writer.transaction_active = false
            
            @info "Wrote $(length(features)) selected features to database"
            
        catch e
            # Rollback on error
            SQLite.execute(writer.db_connection, "ROLLBACK")
            writer.transaction_active = false
            @error "Failed to write selected features: $e"
            rethrow(e)
        end
    end
end

# Write ensemble metrics
function write_ensemble_metrics!(writer::SQLiteResultWriter, metrics::Vector{EnsembleMetric})
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    lock(writer.write_lock) do
        run_id = writer.current_run_id
        
        for metric in metrics
            SQLite.execute(writer.db_connection, """
                INSERT OR REPLACE INTO ensemble_metrics (
                    run_id, metric_name, metric_value, metric_type, measurement_timestamp
                ) VALUES (?, ?, ?, ?, ?)
            """, [
                run_id,
                metric.metric_name,
                metric.metric_value,
                metric.metric_type,
                metric.measurement_timestamp
            ])
        end
        
        @debug "Wrote $(length(metrics)) ensemble metrics"
    end
end

# Write tree statistics
function write_tree_statistics!(writer::SQLiteResultWriter, stats::Vector{TreeStatistic})
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    lock(writer.write_lock) do
        run_id = writer.current_run_id
        
        SQLite.execute(writer.db_connection, "BEGIN TRANSACTION")
        writer.transaction_active = true
        
        try
            # Clear existing tree statistics for this run
            SQLite.execute(writer.db_connection, 
                         "DELETE FROM tree_statistics WHERE run_id = ?", 
                         [run_id])
            
            # Insert new statistics
            for stat in stats
                SQLite.execute(writer.db_connection, """
                    INSERT INTO tree_statistics (
                        run_id, tree_id, gpu_id, total_iterations, max_depth,
                        final_score, convergence_iteration, exploration_ratio,
                        unique_features_explored, execution_time_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id,
                    stat.tree_id,
                    stat.gpu_id,
                    stat.total_iterations,
                    stat.max_depth,
                    stat.final_score,
                    stat.convergence_iteration,
                    stat.exploration_ratio,
                    stat.unique_features_explored,
                    stat.execution_time_seconds
                ])
            end
            
            SQLite.execute(writer.db_connection, "COMMIT")
            writer.transaction_active = false
            
            @info "Wrote $(length(stats)) tree statistics to database"
            
        catch e
            SQLite.execute(writer.db_connection, "ROLLBACK")
            writer.transaction_active = false
            @error "Failed to write tree statistics: $e"
            rethrow(e)
        end
    end
end

# Write consensus history
function write_consensus_history!(writer::SQLiteResultWriter, history::Vector{ConsensusEntry})
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    lock(writer.write_lock) do
        run_id = writer.current_run_id
        
        for entry in history
            SQLite.execute(writer.db_connection, """
                INSERT OR REPLACE INTO consensus_history (
                    run_id, iteration, feature_id, vote_count, weighted_score,
                    selection_probability, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                entry.iteration,
                entry.feature_id,
                entry.vote_count,
                entry.weighted_score,
                entry.selection_probability,
                entry.timestamp
            ])
        end
        
        @debug "Wrote $(length(history)) consensus history entries"
    end
end

# Write performance metrics
function write_performance_metrics!(writer::SQLiteResultWriter, metrics::Vector{PerformanceMetric})
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    lock(writer.write_lock) do
        run_id = writer.current_run_id
        
        for metric in metrics
            SQLite.execute(writer.db_connection, """
                INSERT INTO performance_metrics (
                    run_id, metric_timestamp, gpu_id, gpu_utilization,
                    gpu_memory_used, gpu_memory_total, gpu_temperature,
                    throughput_trees_per_second
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                metric.metric_timestamp,
                metric.gpu_id,
                metric.gpu_utilization,
                metric.gpu_memory_used,
                metric.gpu_memory_total,
                metric.gpu_temperature,
                metric.throughput_trees_per_second
            ])
        end
        
        @debug "Wrote $(length(metrics)) performance metrics"
    end
end

# Validate feature integrity
function validate_features!(writer::SQLiteResultWriter, original_features::Vector{Int})::FeatureValidation
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    lock(writer.write_lock) do
        run_id = writer.current_run_id
        
        # Get selected features from database
        selected_query = SQLite.execute(writer.db_connection, """
            SELECT original_feature_id FROM selected_features 
            WHERE run_id = ? ORDER BY selection_rank
        """, [run_id])
        
        selected_features = [row[1] for row in selected_query]
        
        # Find missing, duplicate, and invalid features
        missing_features = setdiff(original_features, selected_features)
        duplicate_features = Int[]
        invalid_features = Int[]
        
        # Check for duplicates
        seen = Set{Int}()
        for feature_id in selected_features
            if feature_id in seen
                push!(duplicate_features, feature_id)
            else
                push!(seen, feature_id)
            end
        end
        
        # Check for invalid features (not in original set)
        for feature_id in selected_features
            if feature_id ∉ original_features
                push!(invalid_features, feature_id)
            end
        end
        
        # Determine validation status
        validation_status = if isempty(missing_features) && isempty(duplicate_features) && isempty(invalid_features)
            "VALID"
        elseif isempty(invalid_features)
            "WARNING"
        else
            "INVALID"
        end
        
        validation = FeatureValidation(
            length(original_features),
            length(selected_features),
            missing_features,
            duplicate_features,
            invalid_features,
            validation_status,
            now()
        )
        
        # Store validation result
        SQLite.execute(writer.db_connection, """
            INSERT OR REPLACE INTO feature_validation (
                run_id, original_feature_count, validated_feature_count,
                missing_features, duplicate_features, invalid_features,
                validation_status, validation_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id,
            validation.original_feature_count,
            validation.validated_feature_count,
            JSON3.write(validation.missing_features),
            JSON3.write(validation.duplicate_features),
            JSON3.write(validation.invalid_features),
            validation.validation_status,
            validation.validation_timestamp
        ])
        
        @info "Feature validation completed: $(validation.validation_status)"
        if validation.validation_status == "WARNING"
            @warn "Found $(length(validation.missing_features)) missing features"
        elseif validation.validation_status == "INVALID"
            @error "Found $(length(validation.invalid_features)) invalid features"
        end
        
        return validation
    end
end

# Complete the current run
function complete_run!(writer::SQLiteResultWriter, status::ResultStatus, execution_time::Float64)
    if writer.current_run_id === nothing
        throw(ArgumentError("No active run. Call initialize_run! first."))
    end
    
    lock(writer.write_lock) do
        run_id = writer.current_run_id
        
        status_string = string(status)
        
        SQLite.execute(writer.db_connection, """
            UPDATE result_runs 
            SET completion_status = ?, execution_time_seconds = ?
            WHERE run_id = ?
        """, [status_string, execution_time, run_id])
        
        @info "Completed run $run_id with status: $status_string"
        
        # Clear current run
        writer.current_run_id = nothing
    end
end

# Query functions for retrieving results
function get_run_history(writer::SQLiteResultWriter, dataset_name::Union{String, Nothing} = nothing)::DataFrame
    query = if dataset_name === nothing
        "SELECT * FROM result_runs ORDER BY run_timestamp DESC"
    else
        "SELECT * FROM result_runs WHERE dataset_name = ? ORDER BY run_timestamp DESC"
    end
    
    params = dataset_name === nothing ? [] : [dataset_name]
    
    return DataFrame(SQLite.execute(writer.db_connection, query, params))
end

function get_selected_features(writer::SQLiteResultWriter, run_id::String)::DataFrame
    return DataFrame(SQLite.execute(writer.db_connection, """
        SELECT * FROM selected_features 
        WHERE run_id = ? 
        ORDER BY selection_rank
    """, [run_id]))
end

function get_ensemble_metrics(writer::SQLiteResultWriter, run_id::String)::DataFrame
    return DataFrame(SQLite.execute(writer.db_connection, """
        SELECT * FROM ensemble_metrics 
        WHERE run_id = ? 
        ORDER BY metric_name
    """, [run_id]))
end

function get_tree_statistics(writer::SQLiteResultWriter, run_id::String)::DataFrame
    return DataFrame(SQLite.execute(writer.db_connection, """
        SELECT * FROM tree_statistics 
        WHERE run_id = ? 
        ORDER BY tree_id
    """, [run_id]))
end

function get_consensus_history(writer::SQLiteResultWriter, run_id::String)::DataFrame
    return DataFrame(SQLite.execute(writer.db_connection, """
        SELECT * FROM consensus_history 
        WHERE run_id = ? 
        ORDER BY iteration, feature_id
    """, [run_id]))
end

function get_performance_metrics(writer::SQLiteResultWriter, run_id::String)::DataFrame
    return DataFrame(SQLite.execute(writer.db_connection, """
        SELECT * FROM performance_metrics 
        WHERE run_id = ? 
        ORDER BY metric_timestamp
    """, [run_id]))
end

# Database maintenance functions
function vacuum_database!(writer::SQLiteResultWriter)
    lock(writer.write_lock) do
        SQLite.execute(writer.db_connection, "VACUUM")
        @info "Database vacuumed successfully"
    end
end

function cleanup_old_runs!(writer::SQLiteResultWriter, days_to_keep::Int = 30)
    lock(writer.write_lock) do
        cutoff_date = now() - Day(days_to_keep)
        
        # Get runs to delete
        old_runs = SQLite.execute(writer.db_connection, """
            SELECT run_id FROM result_runs 
            WHERE run_timestamp < ?
        """, [cutoff_date])
        
        old_run_ids = [row[1] for row in old_runs]
        
        if isempty(old_run_ids)
            @info "No old runs to cleanup"
            return
        end
        
        # Delete old runs and related data
        for run_id in old_run_ids
            SQLite.execute(writer.db_connection, "DELETE FROM result_runs WHERE run_id = ?", [run_id])
            SQLite.execute(writer.db_connection, "DELETE FROM ensemble_config WHERE run_id = ?", [run_id])
            SQLite.execute(writer.db_connection, "DELETE FROM selected_features WHERE run_id = ?", [run_id])
            SQLite.execute(writer.db_connection, "DELETE FROM ensemble_metrics WHERE run_id = ?", [run_id])
            SQLite.execute(writer.db_connection, "DELETE FROM tree_statistics WHERE run_id = ?", [run_id])
            SQLite.execute(writer.db_connection, "DELETE FROM consensus_history WHERE run_id = ?", [run_id])
            SQLite.execute(writer.db_connection, "DELETE FROM performance_metrics WHERE run_id = ?", [run_id])
            SQLite.execute(writer.db_connection, "DELETE FROM feature_validation WHERE run_id = ?", [run_id])
        end
        
        @info "Cleaned up $(length(old_run_ids)) old runs"
    end
end

# Close database connection
function close_database!(writer::SQLiteResultWriter)
    lock(writer.write_lock) do
        if writer.transaction_active
            SQLite.execute(writer.db_connection, "ROLLBACK")
            writer.transaction_active = false
        end
        
        SQLite.close(writer.db_connection)
        @info "Database connection closed"
    end
end

# Helper functions for creating data structures
function create_selected_feature(feature_id::Int, feature_name::String, original_feature_id::Int,
                                selection_rank::Int, selection_score::Float64, 
                                selection_confidence::Float64, selection_frequency::Float64,
                                feature_importance::Union{Float64, Nothing} = nothing)::SelectedFeature
    return SelectedFeature(
        feature_id,
        feature_name,
        original_feature_id,
        selection_rank,
        selection_score,
        selection_confidence,
        selection_frequency,
        feature_importance
    )
end

function create_ensemble_metric(metric_name::String, metric_value::Float64, 
                               metric_type::String = "GENERAL")::EnsembleMetric
    return EnsembleMetric(
        metric_name,
        metric_value,
        metric_type,
        now()
    )
end

function create_tree_statistic(tree_id::Int, gpu_id::Int, total_iterations::Int,
                              max_depth::Int, final_score::Float64,
                              convergence_iteration::Union{Int, Nothing},
                              exploration_ratio::Float64, unique_features_explored::Int,
                              execution_time_seconds::Float64)::TreeStatistic
    return TreeStatistic(
        tree_id,
        gpu_id,
        total_iterations,
        max_depth,
        final_score,
        convergence_iteration,
        exploration_ratio,
        unique_features_explored,
        execution_time_seconds
    )
end

function create_consensus_entry(iteration::Int, feature_id::Int, vote_count::Int,
                               weighted_score::Float64, selection_probability::Float64)::ConsensusEntry
    return ConsensusEntry(
        iteration,
        feature_id,
        vote_count,
        weighted_score,
        selection_probability,
        now()
    )
end

function create_performance_metric(gpu_id::Int, gpu_utilization::Float64,
                                  gpu_memory_used::Float64, gpu_memory_total::Float64,
                                  gpu_temperature::Float64, throughput_trees_per_second::Float64)::PerformanceMetric
    return PerformanceMetric(
        now(),
        gpu_id,
        gpu_utilization,
        gpu_memory_used,
        gpu_memory_total,
        gpu_temperature,
        throughput_trees_per_second
    )
end

# Export main functions
export SQLiteResultWriter, initialize_run!, store_configuration!, write_selected_features!,
       write_ensemble_metrics!, write_tree_statistics!, write_consensus_history!,
       write_performance_metrics!, validate_features!, complete_run!, get_run_history,
       get_selected_features, get_ensemble_metrics, get_tree_statistics,
       get_consensus_history, get_performance_metrics, vacuum_database!,
       cleanup_old_runs!, close_database!, create_selected_feature, create_ensemble_metric,
       create_tree_statistic, create_consensus_entry, create_performance_metric,
       SelectedFeature, EnsembleMetric, TreeStatistic, ConsensusEntry, PerformanceMetric,
       FeatureValidation, ResultStatus, COMPLETED, FAILED, RUNNING, PENDING, CANCELLED