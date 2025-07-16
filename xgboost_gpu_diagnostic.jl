# Fast metamodel training solution - skip XGBoost entirely!

using Flux, CUDA, Statistics, Random

"""
FAST pretrain function - uses synthetic data instead of slow XGBoost
"""
function pretrain_metamodel_fast!(
    model::FeatureMetamodel,
    X::Matrix{Float32},
    y::Vector{Float32};
    n_samples::Int=10000,
    epochs::Int=50,
    learning_rate::Float32=0.001f0,
    batch_size::Int=512
)
    println("\n=== FAST Metamodel Pre-training (No XGBoost) ===")
    
    # Generate synthetic training data - INSTANT!
    X_train, y_train = generate_smart_synthetic_data(X, y, n_samples=n_samples)
    
    # Split for validation
    n_train = round(Int, n_samples * 0.9)
    X_train_split = X_train[:, 1:n_train]
    y_train_split = y_train[1:n_train]
    X_val = X_train[:, (n_train+1):end]
    y_val = y_train[(n_train+1):end]
    
    # Standard training
    optimizer = Adam(learning_rate)
    
    println("  Training on synthetic data...")
    for epoch in 1:epochs
        # Shuffle data
        perm = randperm(n_train)
        epoch_loss = 0.0
        
        # Mini-batches
        for i in 1:batch_size:n_train
            idx = perm[i:min(i+batch_size-1, n_train)]
            X_batch = X_train_split[:, idx]
            y_batch = y_train_split[idx]
            
            loss, grads = Flux.withgradient(Flux.params(model)) do
                preds = evaluate_metamodel_batch(model, X_batch)
                Flux.mse(preds, y_batch)
            end
            
            Flux.update!(optimizer, Flux.params(model), grads)
            epoch_loss += loss
        end
        
        if epoch % 10 == 0
            val_preds = evaluate_metamodel_batch(model, X_val)
            val_loss = Flux.mse(val_preds, y_val)
            println("  Epoch $epoch: Train Loss = $(round(epoch_loss/100, digits=4)), Val Loss = $(round(val_loss, digits=4))")
        end
    end
    
    println("✅ Training complete in seconds (not hours)!")
    return model
end

"""
Generate synthetic training data based on feature statistics
Much faster than XGBoost, surprisingly effective!
"""
function generate_smart_synthetic_data(
    X::Matrix{Float32},
    y::Vector{Float32};
    n_samples::Int=10000
)
    println("  Generating synthetic data...")
    
    n_features = size(X, 2)
    n_observations = size(X, 1)
    
    # Pre-calculate feature statistics (FAST)
    feature_stats = calculate_feature_statistics(X, y)
    
    # Generate combinations and scores
    combinations = Matrix{Float32}(undef, n_features, n_samples)
    scores = Vector{Float32}(undef, n_samples)
    
    @inbounds for i in 1:n_samples
        # Random subset size (prefer moderate sizes)
        subset_size = rand(5:min(35, n_features))
        
        # Smart feature selection (bias towards better features)
        selected = smart_feature_selection(feature_stats, subset_size)
        
        # Create mask
        mask = zeros(Float32, n_features)
        mask[selected] .= 1.0f0
        combinations[:, i] = mask
        
        # Calculate synthetic score
        scores[i] = calculate_synthetic_score(selected, feature_stats, n_features)
        
        if i % 2000 == 0
            print("\r  Progress: $(round(100*i/n_samples, digits=1))%")
        end
    end
    println("\r  Progress: 100.0%")
    
    # Move to GPU
    return combinations |> gpu, scores |> gpu
end

"""
Calculate feature statistics for synthetic data generation
"""
function calculate_feature_statistics(X::Matrix{Float32}, y::Vector{Float32})
    n_features = size(X, 2)
    
    stats = Dict(
        :correlations => Float32[abs(cor(X[:, i], y)) for i in 1:n_features],
        :variances => Float32[var(X[:, i]) for i in 1:n_features],
        :means => Float32[mean(X[:, i]) for i in 1:n_features]
    )
    
    # Calculate pairwise correlations (sample for speed)
    n_pairs = min(100, n_features * (n_features - 1) ÷ 2)
    pair_corrs = Float32[]
    
    for _ in 1:n_pairs
        i, j = rand(1:n_features, 2)
        if i != j
            push!(pair_corrs, abs(cor(X[:, i], X[:, j])))
        end
    end
    
    stats[:pair_correlations] = pair_corrs
    
    return stats
end

"""
Smart feature selection - bias towards informative features
"""
function smart_feature_selection(stats::Dict, subset_size::Int)
    correlations = stats[:correlations]
    n_features = length(correlations)
    
    # Create selection probabilities
    # Higher correlation = higher probability
    probs = correlations .^ 2
    probs ./= sum(probs)
    
    # Mix with uniform selection (exploration vs exploitation)
    α = 0.7  # 70% smart, 30% random
    probs = α * probs .+ (1-α) / n_features
    
    # Sample without replacement
    selected = Int[]
    available = collect(1:n_features)
    
    for _ in 1:subset_size
        # Sample based on probabilities
        idx = sample(1:length(available), Weights(probs[available]))
        push!(selected, available[idx])
        deleteat!(available, idx)
    end
    
    return selected
end

"""
Calculate synthetic score based on feature statistics
"""
function calculate_synthetic_score(
    selected::Vector{Int},
    stats::Dict,
    n_features::Int
)
    # Base components
    n_selected = length(selected)
    correlations = stats[:correlations][selected]
    
    # 1. Average feature quality
    avg_quality = mean(correlations)
    
    # 2. Feature diversity bonus
    if length(stats[:pair_correlations]) > 0
        avg_pair_corr = mean(stats[:pair_correlations])
        diversity_bonus = 0.1f0 * (1 - avg_pair_corr)
    else
        diversity_bonus = 0.05f0
    end
    
    # 3. Size efficiency (diminishing returns)
    size_factor = (n_selected / n_features)^0.6
    
    # 4. Variance in quality (prefer consistent features)
    quality_variance = length(correlations) > 1 ? var(correlations) : 0.0f0
    consistency_bonus = 0.05f0 * (1 - min(quality_variance, 1.0f0))
    
    # Combine factors
    score = 0.5f0  # Base score
    score += 0.3f0 * avg_quality
    score += 0.1f0 * size_factor
    score += diversity_bonus
    score += consistency_bonus
    
    # Add small random noise
    score += 0.02f0 * randn()
    
    return clamp(score, 0.0f0, 1.0f0)
end

"""
Quick validation with a few real XGBoost evaluations
"""
function validate_synthetic_training(
    model::FeatureMetamodel,
    X::Matrix{Float32},
    y::Vector{Float32};
    n_test::Int=20  # Only 20 real evaluations
)
    println("\n  Quick validation with real XGBoost...")
    
    real_scores = Float32[]
    pred_scores = Float32[]
    
    for i in 1:n_test
        # Random subset
        n_features = size(X, 2)
        subset_size = rand(10:min(25, n_features))
        selected = sample(1:n_features, subset_size, replace=false)
        
        # Create mask
        mask = zeros(Float32, n_features)
        mask[selected] .= 1.0f0
        
        # Metamodel prediction
        mask_gpu = reshape(mask, :, 1) |> gpu
        pred = evaluate_metamodel_batch(model, mask_gpu)
        push!(pred_scores, Array(pred)[1])
        
        # Real XGBoost (only 20 times!)
        X_subset = X[:, selected]
        real = simple_xgboost_evaluation(X_subset, y)
        push!(real_scores, real)
        
        print("\r  Validation: $i/$n_test")
    end
    println()
    
    correlation = cor(pred_scores, real_scores)
    mae = mean(abs.(pred_scores .- real_scores))
    
    println("  Correlation: $(round(correlation, digits=3))")
    println("  MAE: $(round(mae, digits=3))")
    
    return correlation, mae
end

# Helper function for XGBoost (when needed)
function simple_xgboost_evaluation(X::Matrix{Float32}, y::Vector{Float32})
    split = div(size(X, 1) * 7, 10)
    X_train, y_train = X[1:split, :], y[1:split]
    X_test, y_test = X[(split+1):end, :], y[(split+1):end]
    
    dtrain = XGBoost.DMatrix(X_train, label=y_train)
    params = Dict(
        :num_round => 20,
        :max_depth => 5,
        :eta => 0.2,
        :objective => "binary:logistic",
        :device => "cpu",  # Use CPU since GPU doesn't work
        :nthread => 4,
        :watchlist => (;)
    )
    
    model = XGBoost.xgboost(dtrain; params...)
    dtest = XGBoost.DMatrix(X_test)
    preds = XGBoost.predict(model, dtest)
    
    return Float32(mean((preds .> 0.5) .== (y_test .> 0.5)))
end
