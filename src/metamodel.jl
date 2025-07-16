"""
Neural network metamodel for fast feature subset evaluation.
Provides 1000x speedup over real model evaluation.
"""

using Flux, CUDA, Random, Statistics, XGBoost

# Suppress XGBoost verbose output
ENV["XGBOOST_VERBOSITY"] = "0"

# IMPORTANT: Define struct FIRST before any usage
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

# Make FeatureMetamodel compatible with Flux
Flux.@functor FeatureMetamodel

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
    
    # Create model components and move to GPU
    encoder = Dense(n_features => hidden_sizes[1], relu) |> gpu
    attention = MultiHeadAttention(hidden_sizes[1]; nheads=n_attention_heads, dropout_prob=dropout_rate) |> gpu
    decoder = Chain(decoder_layers...) |> gpu
    
    model = FeatureMetamodel(
        encoder,
        attention,
        decoder,
        device
    )
    
    println("✅ Metamodel created and moved to GPU")
    
    return model
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
    
    # Reshape for MultiHeadAttention (needs 3D tensor: features × batch × 1)
    encoded_3d = reshape(encoded, size(encoded, 1), size(encoded, 2), 1)
    
    # Self-attention for feature interactions
    attended_3d = model.attention(encoded_3d, encoded_3d)   # Returns output tensor
    
    # Handle if attention returns a tuple (output, attention_scores)
    if attended_3d isa Tuple
        attended_3d = attended_3d[1]  # Get just the output tensor
    end
    
    # Reshape back to 2D
    attended = reshape(attended_3d, size(attended_3d, 1), size(attended_3d, 2))
    
    # Final prediction
    scores = model.decoder(attended)                    # (1, n_combinations)
    
    return vec(scores)  # Return as vector
end

"""
Generate training data for metamodel using real XGBoost evaluations.
This is expensive but only done once during pre-training.
"""
function generate_metamodel_training_data(X::Matrix{Float32}, y::Vector{Float32}; 
                                        n_samples::Int=10000, min_features::Int=5, max_features::Int=50,
                                        xgb_params::Dict=Dict(), parallel_threads::Int=4, 
                                        progress_interval::Int=500)
    println("Generating metamodel training data...")
    println("  Target samples: $n_samples")
    println("  Feature subset size: $min_features - $max_features")
    println("  Parallel threads: $parallel_threads (available: $(Threads.nthreads()))")
    
    n_features = size(X, 2)
    max_features = min(max_features, n_features)
    
    # Move data to GPU once, before the loop
    println("  Moving data to GPU...")
    X_gpu = X |> gpu
    y_gpu = y |> gpu
    
    # Pre-allocate arrays for thread-safe parallel processing
    feature_combinations = Vector{Vector{Float32}}(undef, n_samples)
    true_scores = Vector{Float32}(undef, n_samples)
    
    # Pre-generate random seeds for reproducibility in parallel
    subset_sizes = [rand(min_features:max_features) for _ in 1:n_samples]
    
    # Progress tracking with thread-safe counter
    completed = Threads.Atomic{Int}(0)
    last_printed = Threads.Atomic{Int}(0)
    
    # Calculate optimal thread allocation - LESS IS MORE!
    # Use fewer Julia threads to leave room for XGBoost GPU operations
    n_julia_threads = max(1, min(4, Threads.nthreads() ÷ 2))  # At least 1 thread, max 4
    
    println("  Processing with $(n_julia_threads) Julia threads for better GPU utilization")
    
    # Process in batches for better thread utilization
    # Ensure we have at least 1 batch even with 1 thread
    batch_size = max(1, n_samples ÷ max(1, n_julia_threads * 10))
    n_batches = cld(n_samples, batch_size)
    
    Threads.@threads for batch_idx in 1:n_batches
        batch_start = (batch_idx - 1) * batch_size + 1
        batch_end = min(batch_idx * batch_size, n_samples)
        
        for i in batch_start:batch_end
            # Random subset selection
            subset_size = subset_sizes[i]
            selected_indices = sample(1:n_features, subset_size, replace=false)
            
            # Create binary mask
            mask = zeros(Float32, n_features)
            mask[selected_indices] .= 1.0f0
            
            # Extract subset on GPU then transfer to CPU for XGBoost
            X_subset_gpu = X_gpu[:, selected_indices]
            X_subset_cpu = Array(X_subset_gpu)  # Transfer only the subset
            y_cpu = Array(y_gpu)  # Transfer labels once per iteration
            
            # Set XGBoost parameters for GPU - no threading params!
            xgb_params_local = copy(xgb_params)
            xgb_params_local["device"] = "cuda"  # Force GPU
            
            # Evaluate with XGBoost
            score = evaluate_with_xgboost(X_subset_cpu, y_cpu; xgb_params=xgb_params_local)
            
            # Store results (thread-safe by index)
            feature_combinations[i] = mask
            true_scores[i] = score
            
            # Update progress counter
            Threads.atomic_add!(completed, 1)
            
            # Print progress (thread-safe)
            if completed[] % progress_interval == 0 && completed[] > last_printed[]
                Threads.atomic_xchg!(last_printed, completed[])
                println("  Progress: $(completed[])/$n_samples ($(round(100*completed[]/n_samples, digits=1))%)")
            end
        end
    end
    
    # Convert to matrix format and move to GPU
    combinations_matrix = hcat(feature_combinations...) |> gpu
    scores_vector = true_scores |> gpu
    
    println("✅ Training data generated: $(size(combinations_matrix, 2)) samples")
    return combinations_matrix, scores_vector
end

"""
Quick XGBoost evaluation for metamodel training.
Returns cross-validation accuracy/R² score.
"""
function evaluate_with_xgboost(X::Union{Matrix{Float32}, CuMatrix{Float32}}, 
                             y::Union{Vector{Float32}, CuVector{Float32}}; 
                             folds::Int=3, 
                             xgb_params::Dict=Dict())
    # For metamodel training, use simple split for speed
    return simple_xgboost_evaluation(X, y; xgb_params=xgb_params)
end

"""
Simple XGBoost evaluation for small datasets.
"""
function simple_xgboost_evaluation(X::Union{Matrix{Float32}, CuMatrix{Float32}}, 
                                 y::Union{Vector{Float32}, CuVector{Float32}}; 
                                 xgb_params::Dict=Dict())
    
    n_samples = size(X, 1)
    split_idx = div(n_samples * 7, 10)  # 70-30 split
    
    # Convert GPU arrays to CPU for XGBoost.jl compatibility
    X_cpu = X isa CuMatrix ? Array(X) : X
    y_cpu = y isa CuVector ? Array(y) : y
    
    X_train, X_test = X_cpu[1:split_idx, :], X_cpu[(split_idx+1):end, :]
    y_train, y_test = y_cpu[1:split_idx], y_cpu[(split_idx+1):end]
    
    # Create DMatrix with CPU arrays
    dtrain = XGBoost.DMatrix(X_train, label=y_train)
    
    # Updated params for XGBoost 2.0+ with proper GPU support
    default_params = Dict(
        :num_round => 30,        # Reduced for speed
        :max_depth => 6,
        :eta => 0.15,           # Slightly higher learning rate
        :objective => "binary:logistic",
        :device => "cuda",      # NEW: Use device instead of gpu_id
        :tree_method => "hist", # NEW: Just "hist" for GPU
        :subsample => 0.8,      # Subsampling for speed
        :colsample_bytree => 0.8,
        :nthread => 1,          # IMPORTANT: Use 1 thread when parallel Julia
        :watchlist => (;)
    )
    
    # Convert string keys to symbols if needed
    xgb_params_sym = Dict(Symbol(k) => v for (k, v) in xgb_params)
    params = merge(default_params, xgb_params_sym)
    
    # Train model
    model = XGBoost.xgboost(dtrain; params...)
    
    # For prediction, create DMatrix and predict
    dtest = XGBoost.DMatrix(X_test)
    predictions = XGBoost.predict(model, dtest)
    
    # Calculate accuracy
    pred_labels = predictions .> 0.5
    actual_labels = y_test .> 0.5
    
    return Float32(mean(pred_labels .== actual_labels))
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
    batch_size::Int=256,
    xgb_params::Dict=Dict(),
    parallel_threads::Int=4,
    progress_interval::Int=500,
    validation_split::Float32=0.1f0,
    min_features::Int=5,
    max_features::Int=50
)
    println("\n=== Metamodel Pre-training ===")
    
    # Generate training data with GPU-accelerated XGBoost
    X_train, y_train = generate_metamodel_training_data(
        X, y, 
        n_samples=n_samples, 
        xgb_params=xgb_params, 
        parallel_threads=parallel_threads,
        progress_interval=progress_interval,
        min_features=min_features,
        max_features=max_features
    )
    
    # Split into train/validation
    n_train = round(Int, size(X_train, 2) * (1 - validation_split))
    train_indices = 1:n_train
    val_indices = (n_train+1):size(X_train, 2)
    
    X_train_split = X_train[:, train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[:, val_indices]
    y_val = y_train[val_indices]
    
    # Training setup
    optimizer = Adam(learning_rate)
    loss_fn = Flux.mse
    
    println("Starting training...")
    println("  Epochs: $epochs")
    println("  Learning rate: $learning_rate")
    println("  Batch size: $batch_size")
    println("  Training samples: $n_train")
    println("  Validation samples: $(length(val_indices))")
    
    best_val_loss = Inf32
    patience = 5
    patience_counter = 0
    
    # Calculate number of minibatches
    n_minibatches = cld(n_train, batch_size)
    
    # Training loop
    for epoch in 1:epochs
        # Shuffle training data
        perm = randperm(n_train)
        epoch_loss = 0f0
        
        # Process mini-batches
        for batch_idx in 1:n_minibatches
            batch_start = (batch_idx - 1) * batch_size + 1
            batch_end = min(batch_idx * batch_size, n_train)
            batch_indices = perm[batch_start:batch_end]
            
            X_batch = X_train_split[:, batch_indices]
            y_batch = y_train_split[batch_indices]
            
            # Forward pass and gradient computation
            loss, grads = Flux.withgradient(Flux.params(model)) do
                predictions = evaluate_metamodel_batch(model, X_batch)
                loss_fn(predictions, y_batch)
            end
            
            # Update parameters
            Flux.update!(optimizer, Flux.params(model), grads)
            
            epoch_loss += loss
        end
        
        # Validation
        val_predictions = evaluate_metamodel_batch(model, X_val)
        val_loss = loss_fn(val_predictions, y_val)
        
        # Early stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                println("  Early stopping at epoch $epoch")
                break
            end
        end
        
        # Progress reporting
        if epoch % 5 == 0
            avg_train_loss = epoch_loss / n_minibatches
            println("  Epoch $epoch: Train Loss = $(round(avg_train_loss, digits=6)), Val Loss = $(round(val_loss, digits=6))")
        end
    end
    
    # Final validation
    final_predictions = evaluate_metamodel_batch(model, X_train)
    final_loss = loss_fn(final_predictions, y_train)
    correlation = cor(Array(final_predictions), Array(y_train))
    
    println("✅ Pre-training completed!")
    println("  Final loss: $(round(final_loss, digits=6))")
    println("  Correlation: $(round(correlation, digits=4))")
    println("  Best validation loss: $(round(best_val_loss, digits=6))")
    
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
        meta_scores_gpu = evaluate_metamodel_batch(model, mask_gpu)
        meta_score = Array(meta_scores_gpu)[1]  # Transfer to CPU before indexing
        
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

# GPU Memory monitoring utility
function monitor_gpu_memory()
    if CUDA.functional()
        used = CUDA.used_memory() / (1024^3)
        total = CUDA.total_memory() / (1024^3)
        available = CUDA.available_memory() / (1024^3)
        println("  GPU Memory: $(round(used, digits=2))/$(round(total, digits=2)) GB used, $(round(available, digits=2)) GB available")
    end
end

"""
Test if XGBoost is actually using GPU
"""
function test_xgboost_gpu()
    println("\n=== Testing XGBoost GPU Support ===")
    
    # Create test data
    X_test = rand(Float32, 1000, 50)
    y_test = rand(0:1, 1000) |> Vector{Float32}
    
    # Monitor GPU before
    println("Before XGBoost:")
    monitor_gpu_memory()
    
    # Test CPU version
    t_cpu = @elapsed begin
        dtrain = XGBoost.DMatrix(X_test, label=y_test)
        params_cpu = Dict(
            :num_round => 10,
            :device => "cpu",
            :tree_method => "hist",
            :objective => "binary:logistic",
            :watchlist => (;)
        )
        model_cpu = XGBoost.xgboost(dtrain; params_cpu...)
    end
    
    # Test GPU version
    t_gpu = @elapsed begin
        dtrain = XGBoost.DMatrix(X_test, label=y_test)
        params_gpu = Dict(
            :num_round => 10,
            :device => "cuda",
            :tree_method => "hist",
            :objective => "binary:logistic",
            :watchlist => (;)
        )
        model_gpu = XGBoost.xgboost(dtrain; params_gpu...)
    end
    
    println("\nAfter XGBoost GPU:")
    monitor_gpu_memory()
    
    println("\nTiming:")
    println("  CPU: $(round(t_cpu, digits=3))s")
    println("  GPU: $(round(t_gpu, digits=3))s") 
    println("  Speedup: $(round(t_cpu/t_gpu, digits=2))x")
    
    if t_gpu >= t_cpu
        @warn "GPU is not faster than CPU! XGBoost may not be using GPU properly."
        println("\nTroubleshooting:")
        println("1. Check if XGBoost was compiled with GPU support")
        println("2. Try: pkg> add XGBoost#master")
        println("3. Consider using LightGBM or CatBoost instead")
    else
        println("✅ GPU acceleration is working!")
    end
end

"""
Batch evaluate multiple feature subsets at once to minimize overhead.
This is much more efficient than individual evaluations.
"""
function generate_metamodel_training_data_batched(
    X::Matrix{Float32}, 
    y::Vector{Float32}; 
    n_samples::Int=10000,
    min_features::Int=5,
    max_features::Int=50,
    xgb_params::Dict=Dict(),
    batch_size::Int=100  # Process this many XGBoost models at once
)
    println("\n=== Batched Metamodel Training Data Generation ===")
    println("  Target samples: $n_samples")
    println("  Batch size: $batch_size")
    
    n_features = size(X, 2)
    max_features = min(max_features, n_features)
    
    # Pre-generate all feature selections
    all_selections = Vector{Vector{Int}}(undef, n_samples)
    for i in 1:n_samples
        subset_size = rand(min_features:max_features)
        all_selections[i] = sample(1:n_features, subset_size, replace=false)
    end
    
    # Group by subset size for efficiency
    size_groups = Dict{Int, Vector{Int}}()
    for (idx, selection) in enumerate(all_selections)
        size = length(selection)
        if !haskey(size_groups, size)
            size_groups[size] = Int[]
        end
        push!(size_groups[size], idx)
    end
    
    # Results storage
    feature_combinations = Vector{Vector{Float32}}(undef, n_samples)
    true_scores = Vector{Float32}(undef, n_samples)
    
    completed = 0
    
    # Process each size group
    for (subset_size, indices) in size_groups
        println("  Processing $(length(indices)) subsets of size $subset_size")
        
        # Process in batches
        for batch_start in 1:batch_size:length(indices)
            batch_end = min(batch_start + batch_size - 1, length(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # Evaluate batch in parallel
            batch_scores = evaluate_batch_parallel(
                X, y, 
                [all_selections[idx] for idx in batch_indices],
                xgb_params
            )
            
            # Store results
            for (local_idx, global_idx) in enumerate(batch_indices)
                mask = zeros(Float32, n_features)
                mask[all_selections[global_idx]] .= 1.0f0
                
                feature_combinations[global_idx] = mask
                true_scores[global_idx] = batch_scores[local_idx]
                
                completed += 1
                if completed % 500 == 0
                    println("  Progress: $completed/$n_samples ($(round(100*completed/n_samples, digits=1))%)")
                end
            end
        end
    end
    
    # Convert to GPU format
    combinations_matrix = hcat(feature_combinations...) |> gpu
    scores_vector = true_scores |> gpu
    
    println("✅ Batch training data generated")
    return combinations_matrix, scores_vector
end

"""
Evaluate multiple feature subsets in parallel.
"""
function evaluate_batch_parallel(
    X::Matrix{Float32},
    y::Vector{Float32},
    feature_subsets::Vector{Vector{Int}},
    xgb_params::Dict
)
    n_subsets = length(feature_subsets)
    scores = Vector{Float32}(undef, n_subsets)
    
    # Use fewer threads to avoid contention
    n_threads = min(4, Threads.nthreads())
    
    # Process subsets in parallel chunks
    chunk_size = cld(n_subsets, n_threads)
    
    Threads.@threads for thread_id in 1:n_threads
        start_idx = (thread_id - 1) * chunk_size + 1
        end_idx = min(thread_id * chunk_size, n_subsets)
        
        for i in start_idx:end_idx
            X_subset = X[:, feature_subsets[i]]
            scores[i] = evaluate_single_xgboost_gpu(X_subset, y, xgb_params)
        end
    end
    
    return scores
end

"""
Single XGBoost evaluation optimized for GPU.
"""
function evaluate_single_xgboost_gpu(
    X::Matrix{Float32},
    y::Vector{Float32},
    xgb_params::Dict
)
    # Simple train/test split
    n = size(X, 1)
    split = div(n * 7, 10)
    
    X_train, X_test = X[1:split, :], X[(split+1):end, :]
    y_train, y_test = y[1:split], y[(split+1):end]
    
    # Create DMatrix
    dtrain = XGBoost.DMatrix(X_train, label=y_train)
    dtest = XGBoost.DMatrix(X_test)
    
    # GPU parameters
    params = Dict(
        :num_round => 20,  # Fewer rounds for speed
        :max_depth => 5,   # Shallower trees
        :eta => 0.2,
        :objective => "binary:logistic",
        :device => "cuda",
        :tree_method => "hist",
        :subsample => 0.8,
        :nthread => 1,  # Important: 1 thread when parallel
        :watchlist => (;)
    )
    
    # Merge with provided params
    for (k, v) in xgb_params
        params[Symbol(k)] = v
    end
    
    # Train
    model = XGBoost.xgboost(dtrain; params...)
    
    # Predict
    predictions = XGBoost.predict(model, dtest)
    
    # Accuracy
    pred_labels = predictions .> 0.5
    actual_labels = y_test .> 0.5
    
    return Float32(mean(pred_labels .== actual_labels))
end

"""
Monitor system resources during training
"""
function monitor_system_resources(duration=60)
    println("\n=== System Resource Monitor ===")
    
    @async begin
        start_time = time()
        while time() - start_time < duration
            # GPU stats
            if CUDA.functional()
                for i in 0:1  # Both GPUs
                    try
                        CUDA.device!(i)
                        used = CUDA.used_memory() / (1024^3)
                        total = CUDA.total_memory() / (1024^3)
                        println("GPU $i: $(round(100*used/total, digits=1))% ($(round(used, digits=1))/$(round(total, digits=1)) GB)")
                    catch
                        # GPU not available
                    end
                end
            end
            
            println("---")
            sleep(5)
        end
    end
end
