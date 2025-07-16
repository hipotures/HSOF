using Statistics, StatsBase

function stage1_filter(X::Matrix{Float64}, y::Vector, feature_names::Vector{String}, target_features::Int)
    println("\n=== Stage 1: Fast Filtering ===")
    println("Input: $(size(X, 2)) features → Reducing to: $target_features features")
    
    n_features = size(X, 2)
    scores = zeros(n_features)
    
    # Compute correlation with target
    for i in 1:n_features
        if !all(isnan.(X[:, i])) && var(X[:, i]) > 1e-10
            scores[i] = abs(cor(X[:, i], y))
        end
    end
    
    # Select top features
    selected_indices = sortperm(scores, rev=true)[1:min(target_features, n_features)]
    selected_features = feature_names[selected_indices]
    
    println("Selected $(length(selected_features)) features")
    println("Top 5 features: $(selected_features[1:min(5, end)])")
    
    return X[:, selected_indices], selected_features, selected_indices
end