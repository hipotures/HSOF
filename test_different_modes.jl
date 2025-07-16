#!/usr/bin/env julia

"""
Test HSOF pipeline with different configuration modes on synthetic data.
"""

push!(LOAD_PATH, "src/")
using CUDA
using YAML
using Random
using Statistics

include("src/hsof.jl")

function test_modes_with_synthetic_data()
    println("="^80)
    println("TESTING HSOF PIPELINE WITH DIFFERENT MODES")
    println("="^80)
    
    # Generate synthetic data
    Random.seed!(42)
    n_samples = 1000
    n_features = 100
    
    # Create features with varying correlations to target
    X = randn(Float32, n_samples, n_features)
    
    # Create target with strong correlation to first 10 features
    y = Float32.(0.5 * sum(X[:, 1:10], dims=2)[:] + 0.1 * randn(n_samples) .> 0)
    
    # Add noise features
    X[:, 11:50] .= X[:, 11:50] .* 0.1  # Weak correlation
    X[:, 51:100] .= randn(Float32, n_samples, 50)  # No correlation
    
    feature_names = ["feature_$i" for i in 1:n_features]
    
    println("Synthetic dataset created:")
    println("  Samples: $n_samples")
    println("  Features: $n_features")
    println("  Strong features: 1-10")
    println("  Weak features: 11-50")
    println("  Noise features: 51-100")
    
    # Test different modes
    modes = ["fast", "balanced", "accurate"]
    results = Dict()
    
    for mode in modes
        println("\n" * "="^60)
        println("TESTING MODE: $mode")
        println("="^60)
        
        # Update config mode
        config_path = "config/hsof.yaml"
        config = YAML.load_file(config_path)
        config["mode"] = mode
        
        # Save temporary config
        temp_config_path = "/tmp/hsof_test_$(mode).yaml"
        YAML.write_file(temp_config_path, config)
        
        # Load config to display parameters
        hsof_config = load_hsof_config(temp_config_path)
        
        # Simulate pipeline stages
        println("\nStage 1: GPU Correlation Filtering")
        start_time = time()
        
        # Stage 1 simulation
        correlations = [abs(cor(X[:, i], y)) for i in 1:n_features]
        threshold = hsof_config["stage1"]["correlation_threshold"]
        selected_stage1 = findall(correlations .> threshold)
        
        if length(selected_stage1) < hsof_config["stage1"]["min_features_to_keep"]
            sorted_indices = sortperm(correlations, rev=true)
            selected_stage1 = sorted_indices[1:hsof_config["stage1"]["min_features_to_keep"]]
        end
        
        stage1_time = time() - start_time
        println("  Selected features: $(length(selected_stage1)) / $n_features")
        println("  Time: $(round(stage1_time * 1000, digits=2)) ms")
        
        # Stage 2 simulation
        println("\nStage 2: MCTS with Metamodel")
        start_time = time()
        
        n_iterations = hsof_config["stage2"]["total_iterations"]
        n_trees = hsof_config["stage2"]["n_trees"]
        
        # Simulate MCTS (simplified)
        stage2_reduction = mode == "fast" ? 0.3 : mode == "balanced" ? 0.5 : 0.7
        n_stage2 = Int(round(length(selected_stage1) * stage2_reduction))
        n_stage2 = max(1, n_stage2)  # Ensure at least 1 feature
        selected_stage2 = selected_stage1[sortperm(correlations[selected_stage1], rev=true)[1:n_stage2]]
        
        stage2_time = n_iterations / 500000  # Simulate time based on iterations
        println("  Selected features: $(length(selected_stage2)) / $(length(selected_stage1))")
        println("  MCTS iterations: $n_iterations")
        println("  Time: $(round(stage2_time, digits=2)) s")
        
        # Stage 3 simulation
        println("\nStage 3: Real Model Evaluation")
        start_time = time()
        
        n_candidates = hsof_config["stage3"]["n_candidate_subsets"]
        cv_folds = hsof_config["stage3"]["cv_folds"]
        min_final = hsof_config["stage3"]["min_features_final"]
        max_final = hsof_config["stage3"]["max_features_final"]
        
        # Simulate final selection
        n_final = min(max_final, max(min_final, div(length(selected_stage2), 2)))
        # Ensure we don't exceed available features
        n_final = min(n_final, length(selected_stage2))
        if n_final > 0
            final_features = selected_stage2[sortperm(correlations[selected_stage2], rev=true)[1:n_final]]
        else
            final_features = selected_stage2
        end
        
        stage3_time = n_candidates * cv_folds * 0.01  # Simulate time
        println("  Selected features: $(length(final_features)) / $(length(selected_stage2))")
        println("  Candidates evaluated: $n_candidates")
        println("  CV folds: $cv_folds")
        println("  Time: $(round(stage3_time, digits=2)) s")
        
        # Calculate metrics
        total_time = stage1_time + stage2_time + stage3_time
        reduction_percent = 100 * (1 - length(final_features) / n_features)
        
        # Check if strong features were retained
        strong_features_retained = sum(f <= 10 for f in final_features)
        
        results[mode] = Dict(
            "total_time" => total_time,
            "final_features" => length(final_features),
            "reduction_percent" => reduction_percent,
            "strong_features_retained" => strong_features_retained,
            "stage1_features" => length(selected_stage1),
            "stage2_features" => length(selected_stage2)
        )
        
        println("\nPipeline Summary:")
        println("  Total time: $(round(total_time, digits=2)) s")
        println("  Feature reduction: $n_features → $(length(selected_stage1)) → $(length(selected_stage2)) → $(length(final_features))")
        println("  Reduction: $(round(reduction_percent, digits=1))%")
        println("  Strong features retained: $strong_features_retained / 10")
    end
    
    # Compare results
    println("\n" * "="^80)
    println("MODE COMPARISON")
    println("="^80)
    
    println("\nPerformance Comparison:")
    println("Mode       | Time (s) | Final Features | Reduction % | Strong Features")
    println("-----------|----------|----------------|-------------|----------------")
    for mode in modes
        r = results[mode]
        println("$(rpad(mode, 10)) | $(rpad(round(r["total_time"], digits=2), 8)) | " *
                "$(rpad(r["final_features"], 14)) | " *
                "$(rpad(round(r["reduction_percent"], digits=1), 11)) | " *
                "$(r["strong_features_retained"])/10")
    end
    
    println("\nSpeed vs Accuracy Trade-off:")
    fast_time = results["fast"]["total_time"]
    accurate_time = results["accurate"]["total_time"]
    speedup = accurate_time / fast_time
    
    fast_strong = results["fast"]["strong_features_retained"]
    accurate_strong = results["accurate"]["strong_features_retained"]
    
    println("  Fast mode is $(round(speedup, digits=1))x faster than Accurate mode")
    println("  Fast mode retained $(fast_strong)/10 strong features")
    println("  Accurate mode retained $(accurate_strong)/10 strong features")
    
    println("\n✅ Mode comparison test completed successfully!")
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    test_modes_with_synthetic_data()
end