using SQLite, DataFrames

function load_dataset(config::HSoFConfig)
    println("Loading dataset: $(config.dataset_name)")
    
    conn = SQLite.DB(config.database_path)
    query = "SELECT * FROM $(config.table_name)"
    data = DataFrame(SQLite.DBInterface.execute(conn, query))
    SQLite.close(conn)
    
    # Remove ID columns
    select!(data, Not(config.id_columns))
    
    # Extract target
    y = data[!, config.target_column]
    select!(data, Not([config.target_column]))
    
    # Convert to numeric data only
    numeric_data = DataFrame()
    feature_names = String[]
    
    for col in names(data)
        col_data = data[!, col]
        if eltype(col_data) <: Union{Number, Missing}
            numeric_data[!, col] = col_data
            push!(feature_names, col)
        else
            # Try to convert string columns to numeric
            try
                numeric_col = parse.(Float64, replace.(string.(col_data), "missing" => "NaN"))
                numeric_data[!, col] = numeric_col
                push!(feature_names, col)
            catch
                println("Warning: Skipping non-numeric column: $col")
            end
        end
    end
    
    # Replace missing values with 0
    for col in names(numeric_data)
        numeric_data[!, col] = coalesce.(numeric_data[!, col], 0.0)
    end
    
    X = Matrix{Float64}(numeric_data)
    
    println("Data loaded: $(size(X, 1)) samples × $(size(X, 2)) features")
    return X, y, feature_names
end