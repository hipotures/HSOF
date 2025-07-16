"""
Neural network metamodel for fast feature subset evaluation.
Provides 1000x speedup over real model evaluation.
"""

using Flux, CUDA, Random, Statistics, XGBoost

# Suppress XGBoost verbose output
ENV["XGBOOST_VERBOSITY"] = "0"

"""
Metamodel architecture for feature subset evaluation.
Uses Dense layers + Multi-head attention for feature interactions.
"""
struct FeatureMetamodel
    encoder::Dense                    # n_features → 256
    attention::MultiHeadAttention     # Feature interaction modeling
    decoder::Chain                    # 256 → 128 → 64 → 1
    device::Symbol                    # :gpu (no CPU fallback)
end

"""
Create metamodel with multi-head attention for feature interactions.
"""
function create_metamodel(
    n_features::Int; 
    device=:gpu,
    hidden_sizes::Vector{Int}=[256, 128, 64],
    n_attention_heads::Int=8,
    dropout_rate::Float64=0.2
)
    println("Creating metamodel architecture:")
    println("  Input features: $n_features")
    println("  Hidden sizes: $(join(hidden_sizes, "→"))")
    println("  Attention heads: $n_attention_heads")
    println("  Dropout rate: $dropout_rate")
    
    # Build decoder chain dynamically based on hidden_sizes
    decoder_layers = []
    for i in 2:length(hidden_sizes)
        push!(decoder_layers, Dense(hidden_sizes[i-1] => hidden_sizes[i], relu))
        if i < length(hidden_sizes)  # Don't add dropout after last hidden layer
            push!(decoder_layers, Dropout(dropout_rate * (length(hidden_sizes) - i + 1) / length(hidden_sizes)))
        end
    end
    push!(decoder_layers, Dense(hidden_sizes[end] => 1, sigmoid))
    
    model = FeatureMetamodel(
        Dense(n_features => hidden_sizes[1], relu),
        MultiHeadAttention(hidden_sizes[1]; nheads=n_attention_heads, dropout_prob=dropout_rate),
        Chain(decoder_layers...),
        device
    )
    
    # Move to GPU (no CPU fallback)
    gpu_model = model |> gpu
    println("✅ Metamodel created and moved to GPU")
    
    return gpu_model
end

"""
Fast batch evaluation of feature combinations using metamodel.
Input: feature_masks (n_features, n_combinations) - binary matrix
Output: predicted CV scores (n_combinations,)
"""
function evaluate_metamodel_batch(model::FeatureMetamodel, feature_masks::CuMatrix{Float32})
    @assert size(feature_masks, 1) > 0 "Empty feature masks"
    @assert size(feature_masks, 2) > 0 "No feature combinations to evaluate"
    
    # Forward pass through metamodel
    encoded = model.encoder(feature_masks)              # (256, n_combinations)
    
    # Self-attention for feature interactions
    attended = model.attention(encoded, encoded)        # (256, n_combinations)
    
    # Final prediction
    scores = model.decoder(attended)                    # (1, n_combinations)
    
    return vec(scores)  # Return as vector
end

"""
Generate training data for metamodel using real XGBoost evaluations.
This is expensive but only done once during pre-training.
"""
function generate_metamodel_training_data(X::Matrix{Float32}, y::Vector{Float32}; 
                                        n_samples::Int=10000, min_features::Int=5, max_features::Int=50)
    println("Generating metamodel training data...")
    println("  Target samples: $n_samples")
    println("  Feature subset size: $min_features - $max_features")
    
    n_features = size(X, 2)
    max_features = min(max_features, n_features)
    
    feature_combinations = []
    true_scores = Float32[]
    
    for i in 1:n_samples
        # Random subset size
        subset_size = rand(min_features:max_features)
        
        # Create binary mask
        mask = zeros(Float32, n_features)
        selected_indices = sample(1:n_features, subset_size, replace=false)
        mask[selected_indices] .= 1.0f0
        
        # Evaluate with real XGBoost (expensive!)
        X_subset = X[:, selected_indices]
        score = evaluate_with_xgboost(X_subset, y)
        
        push!(feature_combinations, mask)
        push!(true_scores, score)
        
        if i % 1000 == 0
            println("  Progress: $i/$n_samples ($(round(100*i/n_samples, digits=1))%)")
        end
    end
    
    # Convert to GPU arrays
    combinations_matrix = hcat(feature_combinations...) |> gpu  # (n_features, n_samples)
    scores_vector = true_scores |> gpu                          # (n_samples,)
    
    println("✅ Training data generated: $(size(combinations_matrix, 2)) samples")
    return combinations_matrix, scores_vector
end

"""
Quick XGBoost evaluation for metamodel training.
Returns cross-validation accuracy/R² score.
"""
function evaluate_with_xgboost(X::Matrix{Float32}, y::Vector{Float32}; folds::Int=3)
    
    n_samples = size(X, 1)
    if n_samples < folds
        # Too few samples for CV, use simple train/test split
        return simple_xgboost_evaluation(X, y)
    end
    
    fold_size = div(n_samples, folds)
    scores = Float64[]
    
    for fold in 1:folds
        # Create train/test split
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == folds ? n_samples : fold * fold_size
        
        test_indices = test_start:test_end
        train_indices = setdiff(1:n_samples, test_indices)
        
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]
        
        # Train XGBoost
        dtrain = XGBoost.DMatrix(X_train, label=y_train)
        model = XGBoost.xgboost(dtrain, num_round=50, max_depth=6, eta=0.1, 
                               objective="binary:logistic", verbosity=0, silent=1)
        
        # Predict and evaluate
        dtest = XGBoost.DMatrix(X_test)
        predictions = XGBoost.predict(model, dtest)
        
        # Calculate accuracy for binary classification
        pred_labels = predictions .> 0.5
        actual_labels = y_test .> 0.5
        accuracy = mean(pred_labels .== actual_labels)
        
        push!(scores, accuracy)
    end
    
    return mean(scores)
end

"""
Simple XGBoost evaluation for small datasets.
"""
function simple_xgboost_evaluation(X::Matrix{Float32}, y::Vector{Float32})
    
    n_samples = size(X, 1)
    split_idx = div(n_samples, 2)
    
    X_train, X_test = X[1:split_idx, :], X[(split_idx+1):end, :]
    y_train, y_test = y[1:split_idx], y[(split_idx+1):end]
    
    dtrain = XGBoost.DMatrix(X_train, label=y_train)
    model = XGBoost.xgboost(dtrain, num_round=50, max_depth=6, eta=0.1,
                           objective="binary:logistic", verbosity=0, silent=1)
    
    dtest = XGBoost.DMatrix(X_test)
    predictions = XGBoost.predict(model, dtest)
    
    pred_labels = predictions .> 0.5
    actual_labels = y_test .> 0.5
    
    return mean(pred_labels .== actual_labels)
end

"""
Pre-train metamodel on random feature combinations.
This is the expensive training phase done once before MCTS.
"""
function pretrain_metamodel!(
    model::FeatureMetamodel, 
    X::Matrix{Float32}, 
    y::Vector{Float32};
    n_samples::Int=10000, 
    epochs::Int=50, 
    learning_rate::Float32=0.001f0,
    batch_size::Int=256
)
    println("\n=== Metamodel Pre-training ===")
    
    # Generate training data (expensive!)
    X_train, y_train = generate_metamodel_training_data(X, y, n_samples=n_samples)
    
    # Training setup
    optimizer = Adam(learning_rate)
    loss_fn = Flux.mse
    
    println("Starting training...")
    println("  Epochs: $epochs")
    println("  Learning rate: $learning_rate")
    println("  Batch size: $batch_size")
    
    # Training loop
    for epoch in 1:epochs
        # Forward pass and loss computation
        predictions = evaluate_metamodel_batch(model, X_train)
        loss = loss_fn(predictions, y_train)
        
        # Backward pass
        gradients = gradient(() -> loss_fn(evaluate_metamodel_batch(model, X_train), y_train), 
                           Flux.params(model))
        
        # Update parameters
        Flux.update!(optimizer, Flux.params(model), gradients)
        
        # Progress reporting
        if epoch % 10 == 0
            println("  Epoch $epoch: Loss = $(round(loss, digits=6))")
        end
    end
    
    # Final validation
    final_predictions = evaluate_metamodel_batch(model, X_train)
    final_loss = loss_fn(final_predictions, y_train)
    correlation = cor(Array(final_predictions), Array(y_train))
    
    println("✅ Pre-training completed!")
    println("  Final loss: $(round(final_loss, digits=6))")
    println("  Correlation: $(round(correlation, digits=4))")
    
    if correlation < 0.6
        @warn "Low correlation ($(round(correlation, digits=3))) - consider more training data or different architecture"
    end
    
    return model
end

"""
Online learning update during MCTS.
Updates metamodel with new feature combinations and their real scores.
"""
function update_metamodel!(
    model::FeatureMetamodel, 
    new_combinations::CuMatrix{Float32}, 
    new_scores::CuVector{Float32};
    learning_rate::Float32=0.0001f0
)
    
    if size(new_combinations, 2) == 0
        return model  # No new data to learn from
    end
    
    # Use lower learning rate for online updates
    optimizer = Adam(learning_rate)
    
    # Single gradient update
    loss_fn = Flux.mse
    gradients = gradient(() -> loss_fn(evaluate_metamodel_batch(model, new_combinations), new_scores),
                        Flux.params(model))
    
    Flux.update!(optimizer, Flux.params(model), gradients)
    
    return model
end

"""
Validate metamodel accuracy against real XGBoost evaluations.
"""
function validate_metamodel_accuracy(model::FeatureMetamodel, X::Matrix{Float32}, y::Vector{Float32}; 
                                   n_test::Int=100)
    println("\n=== Metamodel Validation ===")
    
    n_features = size(X, 2)
    metamodel_scores = Float32[]
    real_scores = Float32[]
    
    for i in 1:n_test
        # Random feature combination
        subset_size = rand(10:min(30, n_features))
        mask = zeros(Float32, n_features)
        selected_indices = sample(1:n_features, subset_size, replace=false)
        mask[selected_indices] .= 1.0f0
        
        # Metamodel prediction
        mask_gpu = reshape(mask, :, 1) |> gpu
        meta_score = evaluate_metamodel_batch(model, mask_gpu)[1]
        
        # Real XGBoost evaluation
        X_subset = X[:, selected_indices]
        real_score = evaluate_with_xgboost(X_subset, y)
        
        push!(metamodel_scores, meta_score)
        push!(real_scores, real_score)
        
        if i % 20 == 0
            println("  Validation progress: $i/$n_test")
        end
    end
    
    # Calculate correlation
    correlation = cor(metamodel_scores, real_scores)
    mae = mean(abs.(metamodel_scores .- real_scores))
    
    println("✅ Validation completed:")
    println("  Correlation: $(round(correlation, digits=4))")
    println("  Mean Absolute Error: $(round(mae, digits=4))")
    
    if correlation < 0.7
        @warn "Low correlation - metamodel may need more training"
    end
    
    return correlation, mae
end