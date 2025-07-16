"""
GPU-optimized data loading for HSOF pipeline.
Handles SQLite database loading and direct GPU memory transfer.
"""

using SQLite, DataFrames, CUDA, YAML

"""
Load configuration from YAML file.
"""
function load_config(yaml_path::String)
    config = YAML.load_file(yaml_path)
    
    # Handle different table structures
    if haskey(config, "feature_tables") && haskey(config["feature_tables"], "train_features")
        table = config["feature_tables"]["train_features"]
    elseif haskey(config, "tables") && haskey(config["tables"], "train")
        table = config["tables"]["train"]
    else
        error("No train table found in configuration")
    end
    
    return (
        name = config["name"],
        db_path = config["database"]["path"],
        table = table,
        target = config["target_column"],
        id_cols = config["id_columns"],
        problem_type = config["problem_type"]
    )
end

"""
Load dataset from SQLite database with GPU memory validation.
Returns GPU-ready Float32 matrices.
"""
function load_dataset(config)
    println("Loading dataset: $(config.name)")
    
    # SQLite connection
    conn = SQLite.DB(config.db_path)
    query = "SELECT * FROM $(config.table)"
    data = DataFrame(SQLite.DBInterface.execute(conn, query))
    SQLite.close(conn)
    
    # Remove ID columns
    id_cols_symbols = [Symbol(col) for col in config.id_cols]
    select!(data, Not(id_cols_symbols))
    
    # First extract target variable
    target_symbol = Symbol(config.target)
    y_raw = data[!, target_symbol]
    
    # Convert target to Float32
    if eltype(y_raw) <: Number
        y = Vector{Float32}(y_raw)
    elseif eltype(y_raw) <: AbstractString
        # For string targets (classification), convert to numeric
        unique_values = unique(y_raw)
        value_to_idx = Dict(val => Float32(idx - 1) for (idx, val) in enumerate(unique_values))
        y = [value_to_idx[val] for val in y_raw]
        println("Target variable converted from categorical: $(length(unique_values)) classes")
    else
        # For other types, try to convert
        y = Vector{Float32}(y_raw)
    end
    
    # Remove target from features
    select!(data, Not([target_symbol]))
    
    # Convert to Float32 matrix for GPU optimization
    # First, select only numeric columns
    numeric_cols = Symbol[]
    for col in names(data)
        col_symbol = Symbol(col)
        col_type = eltype(data[!, col_symbol])
        # Handle union types with Missing
        if col_type <: Union{Missing, <:Number}
            # Convert missing values to 0
            data[!, col_symbol] = coalesce.(data[!, col_symbol], 0.0)
            push!(numeric_cols, col_symbol)
        elseif col_type <: Number
            push!(numeric_cols, col_symbol)
        else
            println("Skipping non-numeric column: $col ($(col_type))")
        end
    end
    
    # Select only numeric columns
    select!(data, numeric_cols)
    
    # Now convert to Float32 matrix
    X = Matrix{Float32}(data)
    feature_names = String[string(col) for col in names(data)]
    
    println("Dataset loaded: $(size(X, 1)) samples × $(size(X, 2)) features")
    
    # GPU memory validation
    validate_gpu_memory(X, y)
    
    return X, y, feature_names
end

"""
Validate that dataset fits in GPU memory with safety margin.
"""
function validate_gpu_memory(X::Matrix{Float32}, y::Vector{Float32})
    # Calculate required memory (conservative estimate)
    x_bytes = sizeof(X) * 2  # Original + processed
    y_bytes = sizeof(y) * 2  # Original + processed  
    workspace_bytes = sizeof(Float32) * size(X, 2) * 1000  # Working memory
    total_required = x_bytes + y_bytes + workspace_bytes
    
    # Add 30% safety margin
    total_required = round(Int, total_required * 1.3)
    
    available = CUDA.available_memory()
    
    if total_required > available
        error("Dataset too large for GPU memory!\n" *
              "Required: $(round(total_required/1024^3, digits=2)) GB\n" *
              "Available: $(round(available/1024^3, digits=2)) GB\n" *
              "Consider reducing dataset size or using larger GPU.")
    end
    
    println("GPU memory validation passed:")
    println("  Required: $(round(total_required/1024^3, digits=2)) GB")
    println("  Available: $(round(available/1024^3, digits=2)) GB")
    println("  Safety margin: $(round((available - total_required)/1024^3, digits=2)) GB")
end

"""
Transfer data to GPU with optimized memory layout.
"""
function transfer_to_gpu(X::Matrix{Float32}, y::Vector{Float32})
    println("Transferring data to GPU...")
    
    # Transfer with memory optimization
    X_gpu = CuArray(X)
    y_gpu = CuArray(y)
    
    # Verify transfer
    @assert size(X_gpu) == size(X) "GPU transfer failed for features"
    @assert size(y_gpu) == size(y) "GPU transfer failed for targets"
    
    println("GPU transfer completed successfully")
    return X_gpu, y_gpu
end

"""
Validate GPU functionality and display GPU information.
"""
function validate_gpu_setup()
    if !CUDA.functional()
        error("CUDA not functional! GPU required for HSOF implementation.")
    end
    
    device = CUDA.device()
    println("\n=== GPU Setup Validation ===")
    println("GPU Device: $(CUDA.name(device))")
    println("CUDA Runtime: $(CUDA.runtime_version())")
    println("Total VRAM: $(round(CUDA.total_memory()/1024^3, digits=1)) GB")
    println("Available VRAM: $(round(CUDA.available_memory()/1024^3, digits=1)) GB")
    println("Compute Capability: $(CUDA.capability(device))")
    
    # Test basic GPU operation
    test_array = CUDA.ones(Float32, 1000, 1000)
    result = sum(test_array)
    @assert result ≈ 1000000.0f0 "GPU computation test failed"
    
    println("✅ GPU validation passed - ready for HSOF pipeline")
    return true
end