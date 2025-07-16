#!/usr/bin/env julia

"""
Test full HSOF pipeline with detailed metrics from all stages
"""

push!(LOAD_PATH, "src/")
using CUDA
using DataFrames
using Statistics
using Random

include("src/hsof.jl")

function test_full_pipeline_with_metrics()
    println("="^80)
    println("TESTING FULL HSOF PIPELINE WITH METRICS")
    println("="^80)
    
    # Generate synthetic dataset with known patterns
    Random.seed!(42)
    n_samples = 1000
    n_features = 50
    
    # Create features with varying importance
    X = randn(Float32, n_samples, n_features)
    
    # First 5 features are strongly correlated with target
    # Next 10 features are weakly correlated
    # Remaining 35 features are noise
    
    # Create target based on first 15 features
    y = Float32.(
        0.8 * sum(X[:, 1:5], dims=2)[:] +     # Strong features
        0.2 * sum(X[:, 6:15], dims=2)[:] +    # Weak features
        0.1 * randn(n_samples)                 # Noise
    )
    
    # Convert to binary classification
    y = Float32.(y .> median(y))
    
    feature_names = ["feature_$i" for i in 1:n_features]
    
    # Calculate true correlations for reference
    true_correlations = [abs(cor(X[:, i], y)) for i in 1:n_features]
    
    println("Synthetic dataset created:")
    println("  Samples: $n_samples")
    println("  Features: $n_features")
    println("  Target distribution: $(round(mean(y), digits=3)) positive")
    println("\nTrue feature importance:")
    println("  Strong features (1-5): mean correlation = $(round(mean(true_correlations[1:5]), digits=3))")
    println("  Weak features (6-15): mean correlation = $(round(mean(true_correlations[6:15]), digits=3))")
    println("  Noise features (16-50): mean correlation = $(round(mean(true_correlations[16:50]), digits=3))")
    
    # Load configuration
    config_path = "config/hsof.yaml"
    hsof_config = load_hsof_config(config_path)
    
    println("\n" * "="^80)
    println("STAGE 1: GPU CORRELATION FILTERING")
    println("="^80)
    
    # Run Stage 1
    X1, features1, indices1 = gpu_stage1_filter(
        X, y, feature_names,
        correlation_threshold=hsof_config["stage1"]["correlation_threshold"],
        min_features_to_keep=hsof_config["stage1"]["min_features_to_keep"],
        variance_threshold=hsof_config["stage1"]["variance_threshold"]
    )
    
    # Analyze Stage 1 results
    println("\nStage 1 Analysis:")
    strong_retained = sum(i <= 5 for i in indices1)
    weak_retained = sum(6 <= i <= 15 for i in indices1)
    noise_retained = sum(i > 15 for i in indices1)
    
    println("  Strong features retained: $strong_retained / 5 ($(round(100*strong_retained/5, digits=1))%)")
    println("  Weak features retained: $weak_retained / 10 ($(round(100*weak_retained/10, digits=1))%)")
    println("  Noise features retained: $noise_retained / 35 ($(round(100*noise_retained/35, digits=1))%)")
    
    println("\n" * "="^80)
    println("STAGE 2: MCTS WITH METAMODEL")
    println("="^80)
    
    # Create metamodel
    metamodel = create_metamodel(
        length(features1),
        hidden_sizes=hsof_config["stage2"]["hidden_sizes"],
        n_attention_heads=hsof_config["stage2"]["n_attention_heads"],
        dropout_rate=hsof_config["stage2"]["dropout_rate"]
    )
    
    # Pre-train metamodel
    println("\nPre-training metamodel...")
    pretrain_metamodel!(
        metamodel, X1, y,
        n_samples=hsof_config["stage2"]["pretraining_samples"],
        epochs=hsof_config["stage2"]["pretraining_epochs"],
        batch_size=hsof_config["stage2"]["batch_size"],
        learning_rate=Float32(hsof_config["stage2"]["learning_rate"])
    )
    
    # Validate metamodel
    correlation, mae = validate_metamodel_accuracy(metamodel, X1, y, n_test=50)
    println("Metamodel validation: correlation = $(round(correlation, digits=3)), MAE = $(round(mae, digits=3))")
    
    # Run MCTS
    X2, features2, indices2 = gpu_stage2_mcts_metamodel(
        X1, y, features1, metamodel,
        total_iterations=hsof_config["stage2"]["total_iterations"],
        n_trees=hsof_config["stage2"]["n_trees"],
        exploration_constant=hsof_config["stage2"]["exploration_constant"],
        min_features=hsof_config["stage2"]["min_features"],
        max_features=hsof_config["stage2"]["max_features"]
    )
    
    # Analyze Stage 2 results (indices2 are relative to X1)
    original_indices2 = indices1[indices2]
    strong_retained2 = sum(i <= 5 for i in original_indices2)
    weak_retained2 = sum(6 <= i <= 15 for i in original_indices2)
    noise_retained2 = sum(i > 15 for i in original_indices2)
    
    println("\nStage 2 Analysis:")
    println("  Strong features retained: $strong_retained2 / $strong_retained")
    println("  Weak features retained: $weak_retained2 / $weak_retained")
    println("  Noise features retained: $noise_retained2 / $noise_retained")
    
    println("\n" * "="^80)
    println("STAGE 3: REAL MODEL EVALUATION")
    println("="^80)
    
    # Run Stage 3
    final_features, final_score, best_model = stage3_precise_evaluation(
        X2, y, features2, "binary_classification",
        n_candidates=hsof_config["stage3"]["n_candidate_subsets"],
        target_range=(hsof_config["stage3"]["min_features_final"], hsof_config["stage3"]["max_features_final"]),
        cv_folds=hsof_config["stage3"]["cv_folds"],
        xgboost_params=hsof_config["stage3"]["xgboost"],
        rf_params=hsof_config["stage3"]["random_forest"]
    )
    
    # Final analysis
    println("\n" * "="^80)
    println("FINAL PIPELINE RESULTS")
    println("="^80)
    
    # Map back to original indices
    final_indices = []
    for feat in final_features
        idx = findfirst(==(feat), feature_names)
        if idx !== nothing
            push!(final_indices, idx)
        end
    end
    
    strong_final = sum(i <= 5 for i in final_indices)
    weak_final = sum(6 <= i <= 15 for i in final_indices)
    noise_final = sum(i > 15 for i in final_indices)
    
    println("\nFeature Selection Summary:")
    println("  Original features: $n_features")
    println("  After Stage 1: $(length(features1)) ($(round(100*length(features1)/n_features, digits=1))% retained)")
    println("  After Stage 2: $(length(features2)) ($(round(100*length(features2)/n_features, digits=1))% retained)")
    println("  After Stage 3: $(length(final_features)) ($(round(100*length(final_features)/n_features, digits=1))% retained)")
    
    println("\nFinal Feature Quality:")
    println("  Strong features: $strong_final / 5 ($(round(100*strong_final/5, digits=1))%)")
    println("  Weak features: $weak_final / 10 ($(round(100*weak_final/10, digits=1))%)")
    println("  Noise features: $noise_final / 35 ($(round(100*noise_final/35, digits=1))%)")
    
    println("\nBest Model Performance:")
    println("  Model: $best_model")
    println("  CV Score: $(round(final_score, digits=4))")
    
    # Calculate efficiency metrics
    total_reduction = 100 * (1 - length(final_features) / n_features)
    precision = (strong_final + weak_final) / length(final_features)
    
    println("\nEfficiency Metrics:")
    println("  Total reduction: $(round(total_reduction, digits=1))%")
    println("  Feature precision: $(round(100*precision, digits=1))% (relevant features)")
    
    println("\n✅ Full pipeline test completed successfully!")
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    test_full_pipeline_with_metrics()
end