using DataFrames

"""
    clean_data(X::Matrix{Float64})

Clean data by handling missing values and infinite values.
"""
function clean_data(X::Matrix{Float64})
    # Replace NaN and Inf values with zeros
    X_clean = copy(X)
    for i in eachindex(X_clean)
        if !isfinite(X_clean[i])
            X_clean[i] = 0.0
        end
    end
    return X_clean
end

"""
    normalize_features(X::Matrix{Float64})

Normalize features to have zero mean and unit variance.
"""
function normalize_features(X::Matrix{Float64})
    X_norm = copy(X)
    n_features = size(X, 2)
    
    for i in 1:n_features
        col = X[:, i]
        if var(col) > 1e-10  # Avoid division by zero
            X_norm[:, i] = (col .- mean(col)) ./ std(col)
        end
    end
    
    return X_norm
end

"""
    save_feature_ranking(features::Vector{String}, scores::Vector{Float64}, filename::String)

Save feature ranking to CSV file.
"""
function save_feature_ranking(features::Vector{String}, scores::Vector{Float64}, filename::String)
    df = DataFrame(
        feature = features,
        score = scores,
        rank = 1:length(features)
    )
    
    CSV.write(filename, df)
    println("Feature ranking saved to: $filename")
end