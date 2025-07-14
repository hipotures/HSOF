module ColumnValidator

using SQLite
using DataFrames
using Statistics
using Dates

# Include dependencies
include("connection_pool.jl")
include("metadata_parser.jl")
using .ConnectionPool
using .MetadataParser

export ColumnInfo, ValidationResult, ValidationConfig, MissingValueStrategy,
       validate_columns, infer_column_types, check_type_consistency,
       get_column_statistics, handle_missing_values

"""
Missing value handling strategies
"""
@enum MissingValueStrategy begin
    SKIP = 1
    FILL_MEAN = 2
    FILL_MEDIAN = 3
    FILL_MODE = 4
    FILL_ZERO = 5
    FILL_FORWARD = 6
    FILL_BACKWARD = 7
    RAISE_ERROR = 8
end

"""
Column type information
"""
struct ColumnInfo
    name::String
    sql_type::String
    julia_type::Type
    inferred_type::Union{Type, Nothing}
    is_numeric::Bool
    is_categorical::Bool
    is_datetime::Bool
    has_missing::Bool
    missing_count::Int
    missing_ratio::Float64
    unique_count::Int
    min_value::Any
    max_value::Any
    mean_value::Union{Float64, Nothing}
    median_value::Union{Float64, Nothing}
    mode_value::Any
end

"""
Validation result for a dataset
"""
struct ValidationResult
    valid::Bool
    column_info::Dict{String, ColumnInfo}
    errors::Vector{String}
    warnings::Vector{String}
    type_consistency::Dict{String, Bool}
    statistics_summary::Dict{String, Any}
end

"""
Configuration for validation
"""
Base.@kwdef struct ValidationConfig
    check_types::Bool = true
    check_missing::Bool = true
    missing_threshold::Float64 = 0.5  # Warn if >50% missing
    collect_statistics::Bool = true
    sample_size::Int = 10_000  # Sample size for statistics
    missing_strategy::MissingValueStrategy = SKIP
    categorical_threshold::Int = 100  # Max unique values for categorical
end

"""
    validate_columns(pool::SQLitePool, metadata::DatasetMetadata; config::ValidationConfig) -> ValidationResult

Validate columns existence, types, and consistency for a dataset.
"""
function validate_columns(
    pool::SQLitePool, 
    metadata::DatasetMetadata;
    config::ValidationConfig = ValidationConfig()
)
    errors = String[]
    warnings = String[]
    column_info = Dict{String, ColumnInfo}()
    type_consistency = Dict{String, Bool}()
    
    with_connection(pool) do conn
        # Get table schema
        table_info = DBInterface.execute(conn.db, 
            "PRAGMA table_info($(metadata.table_name))") |> DataFrame
        
        if isempty(table_info)
            push!(errors, "Table $(metadata.table_name) does not exist")
            return ValidationResult(false, column_info, errors, warnings, 
                                  type_consistency, Dict())
        end
        
        # Create column name to type mapping
        column_types = Dict(zip(table_info.name, table_info.type))
        
        # Validate metadata columns exist
        validate_metadata_columns(metadata, column_types, errors)
        
        # Get feature columns
        feature_columns = MetadataParser.get_feature_columns(metadata, conn.db)
        
        # Validate each feature column
        for col_name in feature_columns
            if !haskey(column_types, col_name)
                push!(errors, "Feature column '$col_name' does not exist")
                continue
            end
            
            # Analyze column
            info = analyze_column(conn.db, metadata.table_name, col_name, 
                                column_types[col_name], config)
            column_info[col_name] = info
            
            # Check for issues
            if info.missing_ratio > config.missing_threshold
                push!(warnings, "Column '$col_name' has $(round(info.missing_ratio * 100, digits=1))% missing values")
            end
            
            if config.check_types && !isnothing(info.inferred_type)
                # Check type consistency
                is_consistent = check_column_type_consistency(
                    conn.db, metadata.table_name, col_name, info, config
                )
                type_consistency[col_name] = is_consistent
                
                if !is_consistent
                    push!(warnings, "Column '$col_name' has inconsistent types across chunks")
                end
            end
        end
        
        # Add target column if exists
        if !isnothing(metadata.target_column)
            if haskey(column_types, metadata.target_column)
                info = analyze_column(conn.db, metadata.table_name, 
                                    metadata.target_column, 
                                    column_types[metadata.target_column], config)
                column_info[metadata.target_column] = info
                
                # Check if target is binary or multiclass
                if info.is_categorical && info.unique_count > 10
                    push!(warnings, "Target column has $(info.unique_count) unique values - consider if this is intended")
                end
            else
                push!(errors, "Target column '$(metadata.target_column)' does not exist")
            end
        end
    end
    
    # Collect statistics summary
    stats_summary = if config.collect_statistics
        collect_statistics_summary(column_info)
    else
        Dict{String, Any}()
    end
    
    valid = isempty(errors)
    
    return ValidationResult(valid, column_info, errors, warnings, 
                          type_consistency, stats_summary)
end

"""
Validate that metadata-specified columns exist
"""
function validate_metadata_columns(metadata::DatasetMetadata, 
                                 column_types::Dict, 
                                 errors::Vector{String})
    # Check excluded columns
    for col in metadata.excluded_columns
        if !haskey(column_types, col)
            push!(errors, "Excluded column '$col' does not exist")
        end
    end
    
    # Check ID columns
    for col in metadata.id_columns
        if !haskey(column_types, col)
            push!(errors, "ID column '$col' does not exist")
        end
    end
    
    # Check target column
    if !isnothing(metadata.target_column) && !haskey(column_types, metadata.target_column)
        push!(errors, "Target column '$(metadata.target_column)' does not exist")
    end
end

"""
Analyze a single column
"""
function analyze_column(db::SQLite.DB, table_name::String, col_name::String, 
                       sql_type::String, config::ValidationConfig)
    # Get sample data
    sample_query = """
        SELECT "$col_name" 
        FROM $table_name 
        WHERE "$col_name" IS NOT NULL
        LIMIT $(config.sample_size)
    """
    
    sample_data = DBInterface.execute(db, sample_query) |> DataFrame
    
    # Get missing count
    missing_query = """
        SELECT 
            COUNT(*) as total,
            COUNT("$col_name") as non_null
        FROM $table_name
    """
    
    missing_result = DBInterface.execute(db, missing_query) |> DataFrame
    total_count = missing_result.total[1]
    non_null_count = missing_result.non_null[1]
    missing_count = total_count - non_null_count
    missing_ratio = missing_count / total_count
    
    # Infer type from sample
    julia_type = sql_to_julia_type(sql_type)
    inferred_type = nothing
    
    if !isempty(sample_data) && size(sample_data, 1) > 0
        inferred_type = infer_column_type(sample_data[!, 1])
    end
    
    # Determine column characteristics
    is_numeric = julia_type <: Number || (!isnothing(inferred_type) && inferred_type <: Number)
    is_datetime = julia_type <: Union{Date, DateTime} || 
                  (!isnothing(inferred_type) && inferred_type <: Union{Date, DateTime})
    
    # Get unique count
    unique_query = """
        SELECT COUNT(DISTINCT "$col_name") as unique_count
        FROM $table_name
        WHERE "$col_name" IS NOT NULL
    """
    
    unique_result = DBInterface.execute(db, unique_query) |> DataFrame
    unique_count = unique_result.unique_count[1]
    
    is_categorical = !is_numeric && !is_datetime && unique_count <= config.categorical_threshold
    
    # Get statistics
    min_value, max_value, mean_value, median_value, mode_value = if config.collect_statistics
        get_column_stats(db, table_name, col_name, is_numeric, is_categorical)
    else
        nothing, nothing, nothing, nothing, nothing
    end
    
    return ColumnInfo(
        col_name,
        sql_type,
        julia_type,
        inferred_type,
        is_numeric,
        is_categorical,
        is_datetime,
        missing_count > 0,
        missing_count,
        missing_ratio,
        unique_count,
        min_value,
        max_value,
        mean_value,
        median_value,
        mode_value
    )
end

"""
Map SQL types to Julia types
"""
function sql_to_julia_type(sql_type::String)
    type_upper = uppercase(sql_type)
    
    if occursin("INT", type_upper)
        return Int64
    elseif occursin("REAL", type_upper) || occursin("FLOAT", type_upper) || 
           occursin("DOUBLE", type_upper) || occursin("NUMERIC", type_upper)
        return Float64
    elseif occursin("TEXT", type_upper) || occursin("VARCHAR", type_upper) || 
           occursin("CHAR", type_upper)
        return String
    elseif occursin("BLOB", type_upper)
        return Vector{UInt8}
    elseif occursin("DATE", type_upper) || occursin("TIME", type_upper)
        return String  # SQLite stores dates as strings
    else
        return Any
    end
end

"""
Infer Julia type from data sample
"""
function infer_column_type(data::AbstractVector)
    if isempty(data)
        return Nothing
    end
    
    # Remove missing values
    non_missing = filter(!ismissing, data)
    
    if isempty(non_missing)
        return Missing
    end
    
    # Try to parse as numbers
    numeric_count = 0
    int_count = 0
    
    for val in non_missing[1:min(100, length(non_missing))]
        val_str = string(val)
        
        # Try integer
        try
            parse(Int, val_str)
            int_count += 1
            numeric_count += 1
        catch
            # Try float
            try
                parse(Float64, val_str)
                numeric_count += 1
            catch
                # Not numeric
            end
        end
    end
    
    sample_size = min(100, length(non_missing))
    
    if int_count == sample_size
        return Int64
    elseif numeric_count == sample_size
        return Float64
    elseif all(v -> v in ["0", "1", "true", "false", "TRUE", "FALSE"], 
              string.(non_missing[1:sample_size]))
        return Bool
    else
        # Check for dates
        date_count = 0
        for val in non_missing[1:sample_size]
            try
                Date(string(val))
                date_count += 1
            catch
                try
                    DateTime(string(val))
                    date_count += 1
                catch
                    # Not a date
                end
            end
        end
        
        if date_count > sample_size * 0.8
            return DateTime
        else
            return String
        end
    end
end

"""
Get column statistics
"""
function get_column_stats(db::SQLite.DB, table_name::String, col_name::String,
                         is_numeric::Bool, is_categorical::Bool)
    if is_numeric
        stats_query = """
            SELECT 
                MIN(CAST("$col_name" AS REAL)) as min_val,
                MAX(CAST("$col_name" AS REAL)) as max_val,
                AVG(CAST("$col_name" AS REAL)) as mean_val
            FROM $table_name
            WHERE "$col_name" IS NOT NULL
        """
        
        stats = DBInterface.execute(db, stats_query) |> DataFrame
        
        min_val = stats.min_val[1]
        max_val = stats.max_val[1]
        mean_val = stats.mean_val[1]
        
        # Get median (approximate for large datasets)
        median_val = get_approximate_median(db, table_name, col_name)
        
        # Mode for numeric is less meaningful, set to nothing
        mode_val = nothing
        
        return min_val, max_val, mean_val, median_val, mode_val
    elseif is_categorical
        # Get mode (most frequent value)
        mode_query = """
            SELECT "$col_name" as value, COUNT(*) as count
            FROM $table_name
            WHERE "$col_name" IS NOT NULL
            GROUP BY "$col_name"
            ORDER BY count DESC
            LIMIT 1
        """
        
        mode_result = DBInterface.execute(db, mode_query) |> DataFrame
        mode_val = isempty(mode_result) ? nothing : mode_result.value[1]
        
        # Min/max for categorical (lexicographic)
        minmax_query = """
            SELECT 
                MIN("$col_name") as min_val,
                MAX("$col_name") as max_val
            FROM $table_name
            WHERE "$col_name" IS NOT NULL
        """
        
        minmax = DBInterface.execute(db, minmax_query) |> DataFrame
        min_val = minmax.min_val[1]
        max_val = minmax.max_val[1]
        
        return min_val, max_val, nothing, nothing, mode_val
    else
        return nothing, nothing, nothing, nothing, nothing
    end
end

"""
Get approximate median using percentile_cont (if available) or sampling
"""
function get_approximate_median(db::SQLite.DB, table_name::String, col_name::String)
    # Try percentile_cont (SQLite 3.35.0+)
    try
        median_query = """
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY CAST("$col_name" AS REAL)) as median_val
            FROM $table_name
            WHERE "$col_name" IS NOT NULL
        """
        result = DBInterface.execute(db, median_query) |> DataFrame
        return result.median_val[1]
    catch
        # Fallback: sample-based median
        sample_query = """
            SELECT CAST("$col_name" AS REAL) as value
            FROM $table_name
            WHERE "$col_name" IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 10000
        """
        sample = DBInterface.execute(db, sample_query) |> DataFrame
        return isempty(sample) ? nothing : median(sample.value)
    end
end

"""
Check type consistency across chunks
"""
function check_column_type_consistency(db::SQLite.DB, table_name::String, 
                                     col_name::String, info::ColumnInfo,
                                     config::ValidationConfig)
    # Sample from different parts of the table
    chunk_size = 1000
    n_chunks = 5
    
    # Get total rows
    count_result = DBInterface.execute(db, "SELECT COUNT(*) as count FROM $table_name") |> DataFrame
    total_rows = count_result.count[1]
    
    if total_rows < chunk_size * n_chunks
        return true  # Too small to check chunks
    end
    
    chunk_types = Type[]
    
    for i in 1:n_chunks
        offset = div(total_rows * (i - 1), n_chunks)
        
        chunk_query = """
            SELECT "$col_name"
            FROM $table_name
            WHERE "$col_name" IS NOT NULL
            LIMIT $chunk_size
            OFFSET $offset
        """
        
        chunk_data = DBInterface.execute(db, chunk_query) |> DataFrame
        
        if !isempty(chunk_data)
            chunk_type = infer_column_type(chunk_data[!, 1])
            push!(chunk_types, chunk_type)
        end
    end
    
    # Check if all chunks have same inferred type
    return length(unique(chunk_types)) <= 1
end

"""
Collect summary statistics across all columns
"""
function collect_statistics_summary(column_info::Dict{String, ColumnInfo})
    numeric_cols = filter(p -> p.second.is_numeric, column_info)
    categorical_cols = filter(p -> p.second.is_categorical, column_info)
    
    total_missing = sum(info.missing_count for (_, info) in column_info)
    
    return Dict{String, Any}(
        "total_columns" => length(column_info),
        "numeric_columns" => length(numeric_cols),
        "categorical_columns" => length(categorical_cols),
        "columns_with_missing" => count(info.has_missing for (_, info) in column_info),
        "total_missing_values" => total_missing,
        "average_missing_ratio" => mean(info.missing_ratio for (_, info) in column_info),
        "high_cardinality_columns" => count(info.unique_count > 1000 for (_, info) in column_info)
    )
end

"""
    handle_missing_values(data::DataFrame, column_info::Dict{String, ColumnInfo}, 
                         strategy::MissingValueStrategy) -> DataFrame

Handle missing values according to specified strategy.
"""
function handle_missing_values(data::DataFrame, column_info::Dict{String, ColumnInfo}, 
                             strategy::MissingValueStrategy)
    result = copy(data)
    
    for (col_name, info) in column_info
        if !info.has_missing || !(col_name in names(result))
            continue
        end
        
        col_data = result[!, col_name]
        
        if strategy == SKIP
            # Do nothing, keep missing values
            continue
        elseif strategy == RAISE_ERROR
            if any(ismissing, col_data)
                throw(ErrorException("Column '$col_name' contains missing values"))
            end
        elseif strategy == FILL_ZERO
            result[!, col_name] = coalesce.(col_data, 0)
        elseif strategy == FILL_MEAN && info.is_numeric && !isnothing(info.mean_value)
            result[!, col_name] = coalesce.(col_data, info.mean_value)
        elseif strategy == FILL_MEDIAN && info.is_numeric && !isnothing(info.median_value)
            result[!, col_name] = coalesce.(col_data, info.median_value)
        elseif strategy == FILL_MODE && !isnothing(info.mode_value)
            result[!, col_name] = coalesce.(col_data, info.mode_value)
        elseif strategy == FILL_FORWARD
            # Forward fill
            last_valid = nothing
            for i in 1:nrow(result)
                if ismissing(col_data[i])
                    if !isnothing(last_valid)
                        result[i, col_name] = last_valid
                    end
                else
                    last_valid = col_data[i]
                end
            end
        elseif strategy == FILL_BACKWARD
            # Backward fill
            next_valid = nothing
            for i in nrow(result):-1:1
                if ismissing(col_data[i])
                    if !isnothing(next_valid)
                        result[i, col_name] = next_valid
                    end
                else
                    next_valid = col_data[i]
                end
            end
        end
    end
    
    return result
end

end # module