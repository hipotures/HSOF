"""
Test dataset generation and validation for ensemble integration testing.
Provides standard datasets for consistent testing across different components.
"""

using Random
using Statistics
using JSON3
using CSV
using DataFrames

"""
Generate synthetic dataset with controlled properties for testing
"""
function generate_synthetic_dataset(
    num_samples::Int,
    num_features::Int;
    relevant_features::Int = 50,
    noise_level::Float64 = 0.1,
    correlation_strength::Float64 = 0.3,
    random_seed::Int = 42
)
    Random.seed!(random_seed)
    
    # Generate relevant features with signal
    X_relevant = randn(num_samples, relevant_features)
    
    # Add correlations between relevant features
    for i in 2:relevant_features
        correlation = correlation_strength * randn()
        X_relevant[:, i] .+= correlation .* X_relevant[:, 1]
    end
    
    # Generate noise features
    X_noise = randn(num_samples, num_features - relevant_features) * noise_level
    
    # Combine features
    X = hcat(X_relevant, X_noise)
    
    # Generate target based on relevant features with non-linear relationships
    linear_component = sum(X_relevant[:, 1:min(10, relevant_features)], dims=2)
    nonlinear_component = sum(X_relevant[:, 1:min(5, relevant_features)].^2, dims=2)
    
    target_scores = vec(linear_component + 0.3 * nonlinear_component)
    
    # Create binary classification target
    threshold = median(target_scores)
    y = target_scores .> threshold
    
    # Feature metadata
    feature_names = ["feature_$i" for i in 1:num_features]
    relevant_indices = collect(1:relevant_features)
    
    metadata = Dict(
        "num_samples" => num_samples,
        "num_features" => num_features,
        "relevant_features" => relevant_features,
        "relevant_indices" => relevant_indices,
        "noise_level" => noise_level,
        "correlation_strength" => correlation_strength,
        "class_balance" => sum(y) / length(y),
        "random_seed" => random_seed
    )
    
    return X, y, feature_names, metadata
end

"""
Generate Titanic-like dataset for testing
"""
function generate_titanic_dataset(num_samples::Int = 1000; random_seed::Int = 42)
    Random.seed!(random_seed)
    
    # Generate features similar to Titanic dataset
    age = max.(18, 80 .* rand(num_samples))
    fare = exp.(randn(num_samples) * 0.5 + 3.0)  # Log-normal distribution
    pclass = rand(1:3, num_samples)
    sex = rand([0, 1], num_samples)  # 0 = male, 1 = female
    sibsp = rand(0:5, num_samples)
    parch = rand(0:4, num_samples)
    
    # Add some noise features
    noise_features = randn(num_samples, 10)
    
    # Create survival probability (similar to Titanic patterns)
    survival_prob = 0.1 +  # Base survival
                   0.4 * sex +  # Women more likely to survive
                   0.2 * (pclass .== 1) +  # First class more likely
                   0.1 * (pclass .== 2) +  # Second class somewhat likely
                   0.15 * (age .< 30) +  # Young people more likely
                   0.1 * (sibsp .== 0) +  # Alone more likely
                   0.05 * (fare .> 50)  # High fare more likely
    
    # Add noise to survival probability
    survival_prob .+= 0.1 * randn(num_samples)
    survival_prob = clamp.(survival_prob, 0.0, 1.0)
    
    # Generate binary outcomes
    survived = rand(num_samples) .< survival_prob
    
    # Combine all features
    X = hcat(age, fare, pclass, sex, sibsp, parch, noise_features)
    
    feature_names = ["age", "fare", "pclass", "sex", "sibsp", "parch", 
                    "noise_1", "noise_2", "noise_3", "noise_4", "noise_5",
                    "noise_6", "noise_7", "noise_8", "noise_9", "noise_10"]
    
    relevant_indices = [1, 2, 3, 4, 5, 6]  # First 6 features are relevant
    
    metadata = Dict(
        "dataset_type" => "titanic_like",
        "num_samples" => num_samples,
        "num_features" => size(X, 2),
        "relevant_features" => length(relevant_indices),
        "relevant_indices" => relevant_indices,
        "class_balance" => sum(survived) / length(survived),
        "random_seed" => random_seed
    )
    
    return X, survived, feature_names, metadata
end

"""
Generate MNIST-like feature dataset for testing
"""
function generate_mnist_features(num_samples::Int = 1000; random_seed::Int = 42)
    Random.seed!(random_seed)
    
    # Simulate 28x28 pixel features (784 features)
    num_features = 784
    
    # Generate 10-class classification problem
    true_labels = rand(0:9, num_samples)
    
    # Create feature patterns for each digit
    X = zeros(num_samples, num_features)
    
    for i in 1:num_samples
        label = true_labels[i]
        
        # Create simple patterns for each digit
        base_pattern = zeros(28, 28)
        
        # Add digit-specific patterns
        if label == 0  # Circle-like pattern
            for r in 8:20, c in 8:20
                if 6 < sqrt((r-14)^2 + (c-14)^2) < 10
                    base_pattern[r, c] = 1.0
                end
            end
        elseif label == 1  # Vertical line
            base_pattern[5:23, 12:16] .= 1.0
        elseif label == 2  # Horizontal lines
            base_pattern[8:10, 5:23] .= 1.0
            base_pattern[14:16, 5:23] .= 1.0
            base_pattern[20:22, 5:23] .= 1.0
        else  # Random patterns for other digits
            for _ in 1:50
                r, c = rand(1:28), rand(1:28)
                base_pattern[r, c] = 1.0
            end
        end
        
        # Add noise
        noise = randn(28, 28) * 0.2
        pattern = base_pattern + noise
        
        X[i, :] = vec(pattern)
    end
    
    # Create binary classification (digit 0 vs others)
    y = true_labels .== 0
    
    # Generate feature names
    feature_names = ["pixel_$(i)_$(j)" for i in 1:28 for j in 1:28]
    
    # Relevant features are mostly in the center region
    relevant_indices = []
    for i in 8:20, j in 8:20
        pixel_idx = (i-1) * 28 + j
        push!(relevant_indices, pixel_idx)
    end
    
    metadata = Dict(
        "dataset_type" => "mnist_like",
        "num_samples" => num_samples,
        "num_features" => num_features,
        "relevant_features" => length(relevant_indices),
        "relevant_indices" => relevant_indices,
        "class_balance" => sum(y) / length(y),
        "random_seed" => random_seed,
        "image_size" => (28, 28)
    )
    
    return X, y, feature_names, metadata
end

"""
Generate large-scale synthetic dataset for stress testing
"""
function generate_large_synthetic_dataset(
    num_samples::Int = 5000,
    num_features::Int = 5000;
    relevant_features::Int = 100,
    random_seed::Int = 42
)
    Random.seed!(random_seed)
    
    @info "Generating large synthetic dataset" num_samples num_features relevant_features
    
    # Generate in chunks to avoid memory issues
    chunk_size = 1000
    
    X_chunks = []
    y_chunks = []
    
    for start_idx in 1:chunk_size:num_samples
        end_idx = min(start_idx + chunk_size - 1, num_samples)
        chunk_samples = end_idx - start_idx + 1
        
        # Generate relevant features with structured patterns
        X_relevant = randn(chunk_samples, relevant_features)
        
        # Add group structure to relevant features
        groups = div(relevant_features, 10)
        for g in 1:groups
            group_start = (g-1) * 10 + 1
            group_end = min(g * 10, relevant_features)
            
            # Make features within group correlated
            base_feature = X_relevant[:, group_start]
            for i in (group_start+1):group_end
                X_relevant[:, i] .+= 0.5 * base_feature + 0.3 * randn(chunk_samples)
            end
        end
        
        # Generate noise features
        X_noise = randn(chunk_samples, num_features - relevant_features) * 0.1
        
        # Combine features
        X_chunk = hcat(X_relevant, X_noise)
        
        # Generate complex target
        linear_part = sum(X_relevant[:, 1:min(20, relevant_features)], dims=2)
        interaction_part = sum(X_relevant[:, 1:min(10, relevant_features)] .* 
                             X_relevant[:, 2:min(11, relevant_features)], dims=2)
        
        target_scores = vec(linear_part + 0.5 * interaction_part)
        y_chunk = target_scores .> median(target_scores)
        
        push!(X_chunks, X_chunk)
        push!(y_chunks, y_chunk)
    end
    
    # Combine chunks
    X = vcat(X_chunks...)
    y = vcat(y_chunks...)
    
    feature_names = ["feature_$i" for i in 1:num_features]
    relevant_indices = collect(1:relevant_features)
    
    metadata = Dict(
        "dataset_type" => "large_synthetic",
        "num_samples" => num_samples,
        "num_features" => num_features,
        "relevant_features" => relevant_features,
        "relevant_indices" => relevant_indices,
        "class_balance" => sum(y) / length(y),
        "random_seed" => random_seed,
        "generation_method" => "chunked"
    )
    
    @info "Large synthetic dataset generated" size(X) class_balance=metadata["class_balance"]
    
    return X, y, feature_names, metadata
end

"""
Save dataset to files for reuse
"""
function save_test_dataset(X, y, feature_names, metadata, base_path::String)
    mkpath(dirname(base_path))
    
    # Save features as CSV
    df = DataFrame(X, feature_names)
    CSV.write("$(base_path)_features.csv", df)
    
    # Save targets
    target_df = DataFrame(target = y)
    CSV.write("$(base_path)_targets.csv", target_df)
    
    # Save metadata
    open("$(base_path)_metadata.json", "w") do io
        JSON3.pretty(io, metadata)
    end
    
    @info "Dataset saved to $base_path"
end

"""
Load previously saved dataset
"""
function load_test_dataset(base_path::String)
    # Load features
    df = CSV.read("$(base_path)_features.csv", DataFrame)
    X = Matrix(df)
    feature_names = names(df)
    
    # Load targets
    target_df = CSV.read("$(base_path)_targets.csv", DataFrame)
    y = target_df.target
    
    # Load metadata
    metadata = JSON3.read("$(base_path)_metadata.json", Dict)
    
    return X, y, feature_names, metadata
end

"""
Generate all standard test datasets
"""
function generate_all_test_datasets(output_dir::String = "test/data")
    mkpath(output_dir)
    
    datasets = []
    
    # Generate synthetic dataset
    @info "Generating synthetic dataset..."
    X, y, names, meta = generate_synthetic_dataset(1000, 500, relevant_features=50)
    save_test_dataset(X, y, names, meta, joinpath(output_dir, "synthetic"))
    push!(datasets, ("synthetic", meta))
    
    # Generate Titanic-like dataset
    @info "Generating Titanic-like dataset..."
    X, y, names, meta = generate_titanic_dataset(1000)
    save_test_dataset(X, y, names, meta, joinpath(output_dir, "titanic"))
    push!(datasets, ("titanic", meta))
    
    # Generate MNIST-like dataset
    @info "Generating MNIST-like dataset..."
    X, y, names, meta = generate_mnist_features(1000)
    save_test_dataset(X, y, names, meta, joinpath(output_dir, "mnist"))
    push!(datasets, ("mnist", meta))
    
    # Generate large dataset (smaller for testing)
    @info "Generating large synthetic dataset..."
    X, y, names, meta = generate_large_synthetic_dataset(2000, 2000, relevant_features=100)
    save_test_dataset(X, y, names, meta, joinpath(output_dir, "large"))
    push!(datasets, ("large", meta))
    
    # Save dataset summary
    summary = Dict(
        "generated_at" => string(now()),
        "datasets" => datasets
    )
    
    open(joinpath(output_dir, "dataset_summary.json"), "w") do io
        JSON3.pretty(io, summary)
    end
    
    @info "All test datasets generated in $output_dir"
    return datasets
end

"""
Validate dataset properties for testing
"""
function validate_dataset_properties(X, y, metadata)
    @info "Validating dataset properties..."
    
    # Check dimensions
    @assert size(X, 1) == length(y) "Feature matrix and target vector size mismatch"
    @assert size(X, 2) == metadata["num_features"] "Feature count mismatch"
    
    # Check class balance
    actual_balance = sum(y) / length(y)
    @assert 0.1 < actual_balance < 0.9 "Class imbalance too extreme: $actual_balance"
    
    # Check for relevant features
    if haskey(metadata, "relevant_indices")
        relevant_indices = metadata["relevant_indices"]
        @assert all(1 .<= relevant_indices .<= size(X, 2)) "Relevant indices out of bounds"
    end
    
    # Check for NaN/Inf values
    @assert !any(isnan.(X)) "NaN values found in features"
    @assert !any(isinf.(X)) "Infinite values found in features"
    
    @info "Dataset validation passed" size(X) class_balance=actual_balance
    return true
end

export generate_synthetic_dataset, generate_titanic_dataset, generate_mnist_features
export generate_large_synthetic_dataset, save_test_dataset, load_test_dataset
export generate_all_test_datasets, validate_dataset_properties