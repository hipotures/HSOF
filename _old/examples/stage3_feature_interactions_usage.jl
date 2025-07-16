# Example usage of Feature Interactions analysis for Stage 3

using Random
using Statistics
using DataFrames
using CSV
using LinearAlgebra
using SparseArrays

# Include the modules
include("../src/stage3_evaluation/feature_interactions.jl")
include("../src/stage3_evaluation/mlj_infrastructure.jl")
include("../src/stage3_evaluation/cross_validation.jl")

using .FeatureInteractions
using .MLJInfrastructure
using .CrossValidation

# Set random seed
Random.seed!(123)

"""
Example 1: Basic H-statistic calculation for interaction detection
"""
function h_statistic_example()
    println("=== H-statistic Feature Interaction Example ===\n")
    
    # Generate synthetic data with known interactions
    n_samples = 500
    n_features = 6
    X = randn(n_samples, n_features)
    
    # Create target with explicit interactions
    # Strong interaction between features 1 and 2
    # Weak interaction between features 3 and 4
    y = Int.((
        X[:, 1] + X[:, 2] +                    # Main effects
        3.0 * X[:, 1] .* X[:, 2] +            # Strong interaction
        X[:, 3] + X[:, 4] +                    # Main effects
        0.5 * X[:, 3] .* X[:, 4] +            # Weak interaction
        0.1 * randn(n_samples)                 # Noise
    ) .> 0)
    
    println("Data shape: $(size(X))")
    println("Class distribution: $(countmap(y))")
    println("\nTrue interactions:")
    println("- Strong: Features 1 × 2 (coefficient = 3.0)")
    println("- Weak: Features 3 × 4 (coefficient = 0.5)")
    
    # Train model
    println("\nTraining XGBoost model...")
    model = create_model(:xgboost, :classification, 
                        n_estimators=100, 
                        max_depth=4,
                        learning_rate=0.1)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Create interaction calculator
    calc = InteractionCalculator(
        method=:h_statistic,
        n_samples=200,
        random_state=42
    )
    
    # Calculate specific interactions
    println("\nCalculating H-statistics...")
    
    # Strong interaction
    result_12 = calculate_h_statistic(calc, fitted_model, machine, X, y, 1, 2, n_permutations=10)
    println("\nInteraction 1×2 (strong expected):")
    println("  H-statistic: $(round(result_12.interaction_strength, digits=4))")
    println("  95% CI: [$(round(result_12.confidence_interval[1], digits=4)), $(round(result_12.confidence_interval[2], digits=4))]")
    
    # Weak interaction
    result_34 = calculate_h_statistic(calc, fitted_model, machine, X, y, 3, 4, n_permutations=10)
    println("\nInteraction 3×4 (weak expected):")
    println("  H-statistic: $(round(result_34.interaction_strength, digits=4))")
    println("  95% CI: [$(round(result_34.confidence_interval[1], digits=4)), $(round(result_34.confidence_interval[2], digits=4))]")
    
    # No interaction (control)
    result_56 = calculate_h_statistic(calc, fitted_model, machine, X, y, 5, 6, n_permutations=10)
    println("\nInteraction 5×6 (none expected):")
    println("  H-statistic: $(round(result_56.interaction_strength, digits=4))")
    println("  95% CI: [$(round(result_56.confidence_interval[1], digits=4)), $(round(result_56.confidence_interval[2], digits=4))]")
    
    return result_12, result_34, result_56
end

"""
Example 2: Mutual information for feature interactions
"""
function mutual_information_example()
    println("\n\n=== Mutual Information Interaction Example ===\n")
    
    # Generate data with different types of relationships
    n_samples = 1000
    n_features = 8
    
    # Create correlated features
    X = randn(n_samples, n_features)
    
    # Features 1 and 2 are strongly correlated
    X[:, 2] = 0.8 * X[:, 1] + 0.2 * randn(n_samples)
    
    # Features 3 and 4 have non-linear relationship
    X[:, 4] = sin.(2 * X[:, 3]) + 0.3 * randn(n_samples)
    
    # Features 5 and 6 are categorical
    X[:, 5] = rand(1:3, n_samples)
    X[:, 6] = [x5 == 1 ? rand(1:2) : rand(2:4) for x5 in X[:, 5]]  # Dependent on feature 5
    
    println("Feature relationships:")
    println("- Features 1-2: Linear correlation (r ≈ 0.8)")
    println("- Features 3-4: Non-linear (sine) relationship")
    println("- Features 5-6: Categorical dependency")
    println("- Features 7-8: Independent")
    
    # Create interaction calculator
    calc = InteractionCalculator(
        method=:mutual_info,
        categorical_features=[5, 6],
        random_state=42
    )
    
    # Calculate mutual information for different pairs
    println("\nCalculating mutual information...")
    
    # Linear correlation
    mi_12 = calculate_mutual_information(calc, X, 1, 2, n_bins=10)
    println("\nMI(1,2) - Linear correlation:")
    println("  Normalized MI: $(round(mi_12.interaction_strength, digits=4))")
    println("  Raw MI: $(round(mi_12.metadata[:mi_raw], digits=4))")
    
    # Non-linear relationship
    mi_34 = calculate_mutual_information(calc, X, 3, 4, n_bins=10)
    println("\nMI(3,4) - Non-linear relationship:")
    println("  Normalized MI: $(round(mi_34.interaction_strength, digits=4))")
    println("  Raw MI: $(round(mi_34.metadata[:mi_raw], digits=4))")
    
    # Categorical dependency
    mi_56 = calculate_mutual_information(calc, X, 5, 6, n_bins=10)
    println("\nMI(5,6) - Categorical dependency:")
    println("  Normalized MI: $(round(mi_56.interaction_strength, digits=4))")
    println("  Raw MI: $(round(mi_56.metadata[:mi_raw], digits=4))")
    
    # Independent features
    mi_78 = calculate_mutual_information(calc, X, 7, 8, n_bins=10)
    println("\nMI(7,8) - Independent features:")
    println("  Normalized MI: $(round(mi_78.interaction_strength, digits=4))")
    println("  Raw MI: $(round(mi_78.metadata[:mi_raw], digits=4))")
    
    return mi_12, mi_34, mi_56, mi_78
end

"""
Example 3: Partial dependence based interactions
"""
function partial_dependence_interaction_example()
    println("\n\n=== Partial Dependence Interaction Example ===\n")
    
    # Generate regression data with interactions
    n_samples = 600
    n_features = 5
    X = randn(n_samples, n_features)
    
    # Create target with varying interaction effects
    y = (
        # Main effects
        2 * X[:, 1] + X[:, 2] + 0.5 * X[:, 3] +
        # Interaction effects
        1.5 * X[:, 1] .* X[:, 2] +              # Multiplicative interaction
        sin.(X[:, 3]) .* X[:, 4] +              # Non-linear interaction
        0.2 * randn(n_samples)                   # Noise
    )
    
    println("Regression target with interactions:")
    println("- Main effects: 2*X₁ + X₂ + 0.5*X₃")
    println("- Multiplicative: 1.5*X₁*X₂")
    println("- Non-linear: sin(X₃)*X₄")
    
    # Train Random Forest model
    println("\nTraining Random Forest model...")
    model = create_model(:random_forest, :regression,
                        n_estimators=100,
                        max_depth=10)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Create calculator
    calc = InteractionCalculator(method=:partial_dependence)
    
    # Calculate PD interactions
    println("\nCalculating partial dependence interactions...")
    
    # Multiplicative interaction
    pd_12 = calculate_partial_dependence_interaction(calc, fitted_model, machine, X, 1, 2, grid_size=15)
    println("\nPD Interaction 1×2 (multiplicative):")
    println("  Interaction strength: $(round(pd_12.interaction_strength, digits=4))")
    println("  Grid size: $(pd_12.metadata[:grid_size])×$(pd_12.metadata[:grid_size])")
    
    # Non-linear interaction
    pd_34 = calculate_partial_dependence_interaction(calc, fitted_model, machine, X, 3, 4, grid_size=15)
    println("\nPD Interaction 3×4 (non-linear):")
    println("  Interaction strength: $(round(pd_34.interaction_strength, digits=4))")
    
    # No interaction
    pd_15 = calculate_partial_dependence_interaction(calc, fitted_model, machine, X, 1, 5, grid_size=15)
    println("\nPD Interaction 1×5 (none expected):")
    println("  Interaction strength: $(round(pd_15.interaction_strength, digits=4))")
    
    # Visualize interaction matrix for strongest interaction
    println("\nInteraction matrix sample (1×2, corners):")
    int_matrix = pd_12.metadata[:interaction_matrix]
    println("  Top-left: $(round(int_matrix[1,1], digits=3))")
    println("  Top-right: $(round(int_matrix[1,end], digits=3))")
    println("  Bottom-left: $(round(int_matrix[end,1], digits=3))")
    println("  Bottom-right: $(round(int_matrix[end,end], digits=3))")
    
    return pd_12, pd_34, pd_15
end

"""
Example 4: Performance degradation interaction detection
"""
function performance_degradation_example()
    println("\n\n=== Performance Degradation Interaction Example ===\n")
    
    # Generate classification data with synergistic features
    n_samples = 500
    n_features = 6
    X = randn(n_samples, n_features)
    
    # Create target where features 1 and 2 work synergistically
    # They are weak individually but strong together
    y = Int.((
        0.2 * X[:, 1] + 0.2 * X[:, 2] +        # Weak individual effects
        2.0 * (X[:, 1] .> 0) .* (X[:, 2] .> 0) +  # Strong synergy when both positive
        0.5 * X[:, 3] +                         # Moderate individual effect
        0.3 * randn(n_samples)
    ) .> 0.5)
    
    println("Data with synergistic features:")
    println("- Features 1,2: Weak individually, strong when both positive")
    println("- Feature 3: Moderate individual effect")
    println("- Features 4-6: Noise")
    
    # Train model
    println("\nTraining XGBoost model...")
    model = create_model(:xgboost, :classification,
                        n_estimators=100,
                        max_depth=4)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Create calculator
    calc = InteractionCalculator(
        method=:performance_degradation,
        random_state=42
    )
    
    # Calculate performance degradation
    println("\nCalculating performance degradation interactions...")
    
    # Synergistic features
    deg_12 = calculate_performance_degradation(calc, fitted_model, machine, X, y, 1, 2,
                                             metric=accuracy, n_shuffles=10)
    println("\nDegradation 1×2 (synergistic):")
    println("  Interaction strength: $(round(deg_12.interaction_strength, digits=4))")
    println("  95% CI: [$(round(deg_12.confidence_interval[1], digits=4)), $(round(deg_12.confidence_interval[2], digits=4))]")
    println("  Baseline accuracy: $(round(deg_12.metadata[:baseline_score], digits=4))")
    println("  Degradation when shuffling:")
    println("    - Feature 1 only: $(round(deg_12.metadata[:degradation_i], digits=4))")
    println("    - Feature 2 only: $(round(deg_12.metadata[:degradation_j], digits=4))")
    println("    - Both independently: $(round(deg_12.metadata[:degradation_both], digits=4))")
    println("    - Both jointly: $(round(deg_12.metadata[:degradation_joint], digits=4))")
    
    # Independent features
    deg_34 = calculate_performance_degradation(calc, fitted_model, machine, X, y, 3, 4,
                                             metric=accuracy, n_shuffles=10)
    println("\nDegradation 3×4 (independent):")
    println("  Interaction strength: $(round(deg_34.interaction_strength, digits=4))")
    println("  95% CI: [$(round(deg_34.confidence_interval[1], digits=4)), $(round(deg_34.confidence_interval[2], digits=4))]")
    
    return deg_12, deg_34
end

"""
Example 5: Complete interaction analysis with all methods
"""
function complete_interaction_analysis_example()
    println("\n\n=== Complete Interaction Analysis Example ===\n")
    
    # Generate complex dataset
    n_samples = 800
    n_features = 10
    X = randn(n_samples, n_features)
    
    # Add categorical features
    X[:, 7] = rand(1:3, n_samples)
    X[:, 8] = rand(1:4, n_samples)
    
    # Create target with multiple interaction types
    y = Int.((
        # Main effects
        X[:, 1] + 0.5 * X[:, 2] + 0.3 * X[:, 3] +
        # Various interactions
        2 * X[:, 1] .* X[:, 2] +                # Strong linear interaction
        sin.(X[:, 3]) .* X[:, 4] +              # Non-linear interaction
        (X[:, 5] .> 0) .* (X[:, 6] .> 0) +     # Threshold interaction
        0.5 * (X[:, 7] .== X[:, 8]) +          # Categorical interaction
        0.3 * randn(n_samples)
    ) .> 0.5)
    
    feature_names = ["Linear1", "Linear2", "NonLin1", "NonLin2", 
                    "Thresh1", "Thresh2", "Cat1", "Cat2", "Noise1", "Noise2"]
    
    println("Complex dataset with multiple interaction types:")
    println("- Linear interaction: Linear1 × Linear2")
    println("- Non-linear: sin(NonLin1) × NonLin2")
    println("- Threshold: (Thresh1>0) × (Thresh2>0)")
    println("- Categorical: Cat1 == Cat2")
    
    # Train model
    println("\nTraining model...")
    model = create_model(:xgboost, :classification,
                        n_estimators=150,
                        max_depth=5)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Calculate all interactions with combined method
    println("\nCalculating all pairwise interactions...")
    calc = InteractionCalculator(
        method=:all,
        n_samples=300,
        categorical_features=[7, 8],
        random_state=42
    )
    
    @time interaction_matrix = calculate_all_interactions(
        calc, fitted_model, machine, X, y,
        feature_names=feature_names,
        threshold=0.05,
        show_progress=true
    )
    
    # Get top interactions
    top_interactions = get_significant_interactions(interaction_matrix, top_k=10)
    
    println("\n--- Top 10 Feature Interactions ---")
    for (i, interaction) in enumerate(top_interactions)
        feat1, feat2 = interaction.feature_names
        strength = round(interaction.interaction_strength, digits=4)
        println("$i. $feat1 × $feat2: $strength")
    end
    
    # Export heatmap
    println("\nExporting interaction heatmap...")
    heatmap_df = export_interaction_heatmap(interaction_matrix, "interaction_heatmap.csv")
    println("Saved to: interaction_heatmap.csv")
    
    # Analyze by interaction type
    println("\n--- Interaction Analysis Summary ---")
    println("Total features: $(interaction_matrix.n_features)")
    println("Possible interactions: $(interaction_matrix.n_features * (interaction_matrix.n_features - 1) ÷ 2)")
    println("Significant interactions (>$(interaction_matrix.threshold)): $(length(top_interactions))")
    
    # Find specific expected interactions
    expected_pairs = [
        ("Linear1", "Linear2"),
        ("NonLin1", "NonLin2"),
        ("Thresh1", "Thresh2"),
        ("Cat1", "Cat2")
    ]
    
    println("\nExpected interactions found:")
    for (feat1, feat2) in expected_pairs
        found = false
        for interaction in top_interactions
            if (interaction.feature_names == (feat1, feat2) || 
                interaction.feature_names == (feat2, feat1))
                println("  $feat1 × $feat2: ✓ (strength = $(round(interaction.interaction_strength, digits=4)))")
                found = true
                break
            end
        end
        if !found
            println("  $feat1 × $feat2: ✗ (not in top 10)")
        end
    end
    
    return interaction_matrix, top_interactions
end

"""
Example 6: Method comparison and combination
"""
function method_comparison_example()
    println("\n\n=== Interaction Method Comparison Example ===\n")
    
    # Generate data with known interaction
    n_samples = 400
    X = randn(n_samples, 4)
    
    # Strong interaction between features 1 and 2
    y = Int.((
        X[:, 1] + X[:, 2] + 
        2.5 * X[:, 1] .* X[:, 2] + 
        0.2 * randn(n_samples)
    ) .> 0)
    
    # Train model
    model = create_model(:xgboost, :classification, n_estimators=80, max_depth=4)
    fitted_model, machine = fit_model!(model, X, y, verbosity=0)
    
    # Calculate interaction with different methods
    println("Calculating feature 1×2 interaction with different methods...")
    
    # H-statistic
    calc_h = InteractionCalculator(method=:h_statistic, n_samples=200, random_state=42)
    result_h = calculate_h_statistic(calc_h, fitted_model, machine, X, y, 1, 2, n_permutations=5)
    
    # Mutual information
    calc_mi = InteractionCalculator(method=:mutual_info, random_state=42)
    result_mi = calculate_mutual_information(calc_mi, X, 1, 2)
    
    # Partial dependence
    calc_pd = InteractionCalculator(method=:partial_dependence)
    result_pd = calculate_partial_dependence_interaction(calc_pd, fitted_model, machine, X, 1, 2, grid_size=10)
    
    # Performance degradation
    calc_deg = InteractionCalculator(method=:performance_degradation, random_state=42)
    result_deg = calculate_performance_degradation(calc_deg, fitted_model, machine, X, y, 1, 2, n_shuffles=5)
    
    # Display comparison
    println("\n--- Method Comparison for Features 1×2 ---")
    println("H-statistic:              $(round(result_h.interaction_strength, digits=4))")
    println("Mutual Information:       $(round(result_mi.interaction_strength, digits=4))")
    println("Partial Dependence:       $(round(result_pd.interaction_strength, digits=4))")
    println("Performance Degradation:  $(round(result_deg.interaction_strength, digits=4))")
    
    # Combine methods
    all_results = [result_h, result_mi, result_pd, result_deg]
    
    # Equal weights
    combined_equal = combine_interaction_methods(all_results)
    println("\nCombined (equal weights): $(round(combined_equal[1].interaction_strength, digits=4))")
    
    # Custom weights (emphasize model-based methods)
    weights = [0.4, 0.1, 0.3, 0.2]  # h_stat, mi, pd, perf_deg
    combined_custom = combine_interaction_methods(all_results, weights=weights)
    println("Combined (custom weights): $(round(combined_custom[1].interaction_strength, digits=4))")
    
    # Compare with no interaction (features 3×4)
    println("\n--- Control: Features 3×4 (no interaction) ---")
    result_h_34 = calculate_h_statistic(calc_h, fitted_model, machine, X, y, 3, 4, n_permutations=5)
    result_mi_34 = calculate_mutual_information(calc_mi, X, 3, 4)
    result_pd_34 = calculate_partial_dependence_interaction(calc_pd, fitted_model, machine, X, 3, 4, grid_size=10)
    result_deg_34 = calculate_performance_degradation(calc_deg, fitted_model, machine, X, y, 3, 4, n_shuffles=5)
    
    println("H-statistic:              $(round(result_h_34.interaction_strength, digits=4))")
    println("Mutual Information:       $(round(result_mi_34.interaction_strength, digits=4))")
    println("Partial Dependence:       $(round(result_pd_34.interaction_strength, digits=4))")
    println("Performance Degradation:  $(round(result_deg_34.interaction_strength, digits=4))")
    
    return combined_equal, combined_custom
end

# Run all examples
if abspath(PROGRAM_FILE) == @__FILE__
    println("Feature Interactions Analysis Examples")
    println("=" ^ 50)
    
    # Run examples
    h_results = h_statistic_example()
    mi_results = mutual_information_example()
    pd_results = partial_dependence_interaction_example()
    deg_results = performance_degradation_example()
    matrix, top_ints = complete_interaction_analysis_example()
    combined_results = method_comparison_example()
    
    println("\n" * "=" ^ 50)
    println("All feature interaction examples completed!")
    
    # Summary
    println("\nKey takeaways:")
    println("1. H-statistic measures variance explained by interactions")
    println("2. Mutual information captures both linear and non-linear dependencies")
    println("3. Partial dependence visualizes interaction effects on predictions")
    println("4. Performance degradation reveals synergistic feature pairs")
    println("5. Combining methods provides robust interaction detection")
    println("6. Sparse matrices efficiently store large interaction networks")
    
    # Clean up temporary files
    rm("interaction_heatmap.csv", force=true)
end