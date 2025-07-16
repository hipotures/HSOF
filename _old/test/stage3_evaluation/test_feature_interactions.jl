using Test
using Random
using Statistics
using LinearAlgebra
using SparseArrays
using DataFrames
using CSV

# Include the modules
include("../../src/stage3_evaluation/feature_interactions.jl")
include("../../src/stage3_evaluation/mlj_infrastructure.jl")

using .FeatureInteractions
using .MLJInfrastructure

# Set random seed for reproducibility
Random.seed!(42)

@testset "Feature Interactions Tests" begin
    
    @testset "InteractionResult Structure" begin
        result = InteractionResult(
            (1, 2),
            ("feat1", "feat2"),
            0.75,
            (0.7, 0.8),
            :h_statistic,
            Dict(:test => true)
        )
        
        @test result.feature_indices == (1, 2)
        @test result.feature_names == ("feat1", "feat2")
        @test result.interaction_strength == 0.75
        @test result.confidence_interval == (0.7, 0.8)
        @test result.method == :h_statistic
        @test result.metadata[:test] == true
    end
    
    @testset "SparseInteractionMatrix" begin
        n_features = 10
        matrix = SparseInteractionMatrix(n_features, threshold=0.05)
        
        @test matrix.n_features == 10
        @test size(matrix.interactions) == (10, 10)
        @test nnz(matrix.interactions) == 0  # Initially empty
        @test matrix.threshold == 0.05
        @test matrix.method == :combined
        
        # Test with feature names
        feature_names = ["f$i" for i in 1:10]
        matrix2 = SparseInteractionMatrix(n_features, feature_names=feature_names)
        @test matrix2.feature_names == feature_names
    end
    
    @testset "InteractionCalculator Creation" begin
        calc = InteractionCalculator(
            method=:h_statistic,
            n_samples=100,
            categorical_features=[3, 5],
            n_jobs=2,
            random_state=42
        )
        
        @test calc.method == :h_statistic
        @test calc.n_samples == 100
        @test calc.categorical_features == [3, 5]
        @test calc.n_jobs == 2
        @test calc.random_state == 42
        
        # Test defaults
        calc2 = InteractionCalculator()
        @test calc2.method == :all
        @test isnothing(calc2.n_samples)
        @test calc2.n_jobs == 1
    end
    
    @testset "Discretization" begin
        # Test continuous variable discretization
        x = randn(100)
        x_discrete = FeatureInteractions.discretize(x, 5)
        
        @test length(unique(x_discrete)) <= 5
        @test all(1 .<= x_discrete .<= 5)
        @test length(x_discrete) == 100
        
        # Test that all values are assigned
        @test !any(x_discrete .== 0)
    end
    
    @testset "Entropy Calculation" begin
        # Uniform distribution - maximum entropy
        x_uniform = repeat(1:4, 25)
        h_uniform = FeatureInteractions.entropy_discrete(x_uniform)
        @test h_uniform ≈ log(4) rtol=0.01
        
        # Single value - zero entropy
        x_single = ones(Int, 100)
        h_single = FeatureInteractions.entropy_discrete(x_single)
        @test h_single ≈ 0.0
        
        # Skewed distribution
        x_skewed = vcat(ones(Int, 90), fill(2, 10))
        h_skewed = FeatureInteractions.entropy_discrete(x_skewed)
        @test 0 < h_skewed < h_uniform
    end
    
    @testset "Mutual Information Discrete" begin
        # Perfect correlation
        x = repeat(1:2, 50)
        y = x  # Perfect correlation
        mi = FeatureInteractions.mutual_information_discrete(x, y)
        @test mi ≈ log(2) rtol=0.01  # Maximum MI
        
        # Independent variables
        x_ind = repeat(1:2, 50)
        y_ind = repeat([1, 1, 2, 2], 25)
        mi_ind = FeatureInteractions.mutual_information_discrete(x_ind, y_ind)
        @test mi_ind < 0.01  # Should be close to 0
    end
    
    @testset "Basic Mutual Information Calculation" begin
        calc = InteractionCalculator(method=:mutual_info, random_state=42)
        
        # Generate correlated features
        n_samples = 200
        X = randn(n_samples, 5)
        X[:, 2] = 0.8 * X[:, 1] + 0.2 * randn(n_samples)  # Correlated with feature 1
        X[:, 4] = randn(n_samples)  # Independent
        
        # Calculate MI between correlated features
        result_corr = calculate_mutual_information(calc, X, 1, 2)
        
        # Calculate MI between independent features
        result_ind = calculate_mutual_information(calc, X, 1, 4)
        
        @test result_corr.interaction_strength > result_ind.interaction_strength
        @test result_corr.method == :mutual_info
        @test haskey(result_corr.metadata, :n_bins)
        @test haskey(result_corr.metadata, :mi_raw)
    end
    
    @testset "H-statistic Calculation" begin
        # Generate data with interaction
        n_samples = 150
        X = randn(n_samples, 4)
        
        # Create target with interaction between features 1 and 2
        y = Int.((X[:, 1] + X[:, 2] + 2 * X[:, 1] .* X[:, 2] + 0.1 * randn(n_samples)) .> 0)
        
        # Train a model
        model = create_model(:xgboost, :classification, n_estimators=20, max_depth=3)
        fitted_model, machine = fit_model!(model, X, y, verbosity=0)
        
        # Calculate H-statistic
        calc = InteractionCalculator(method=:h_statistic, n_samples=100, random_state=42)
        
        # Strong interaction (features 1 and 2)
        result_strong = calculate_h_statistic(calc, fitted_model, machine, X, y, 1, 2, n_permutations=5)
        
        # Weak/no interaction (features 3 and 4)
        result_weak = calculate_h_statistic(calc, fitted_model, machine, X, y, 3, 4, n_permutations=5)
        
        @test result_strong.interaction_strength > 0
        @test result_strong.interaction_strength > result_weak.interaction_strength
        @test !isnothing(result_strong.confidence_interval)
        @test result_strong.confidence_interval[1] <= result_strong.interaction_strength
        @test result_strong.interaction_strength <= result_strong.confidence_interval[2]
    end
    
    @testset "Partial Dependence Interaction" begin
        # Generate data
        n_samples = 100
        X = randn(n_samples, 3)
        y = X[:, 1].^2 + X[:, 2] + 0.5 * X[:, 1] .* X[:, 2] + 0.1 * randn(n_samples)
        
        # Train model
        model = create_model(:random_forest, :regression, n_estimators=30)
        fitted_model, machine = fit_model!(model, X, y, verbosity=0)
        
        # Calculate PD interaction
        calc = InteractionCalculator(method=:partial_dependence)
        result = calculate_partial_dependence_interaction(calc, fitted_model, machine, X, 1, 2, grid_size=10)
        
        @test result.interaction_strength > 0
        @test result.method == :partial_dependence
        @test haskey(result.metadata, :pd_matrix)
        @test size(result.metadata[:pd_matrix]) == (10, 10)
    end
    
    @testset "Performance Degradation" begin
        # Generate data
        n_samples = 100
        X = randn(n_samples, 4)
        
        # Features 1 and 2 have synergistic effect
        y = Int.((X[:, 1] + X[:, 2] + sign.(X[:, 1]) .* sign.(X[:, 2]) + 0.1 * randn(n_samples)) .> 0)
        
        # Train model
        model = create_model(:xgboost, :classification, n_estimators=20)
        fitted_model, machine = fit_model!(model, X, y, verbosity=0)
        
        # Define accuracy function for the test
        accuracy_func = (y_true, y_pred) -> mean(y_true .== y_pred)
        
        # Calculate performance degradation
        calc = InteractionCalculator(method=:performance_degradation, random_state=42)
        result = calculate_performance_degradation(calc, fitted_model, machine, X, y, 1, 2,
                                                 metric=accuracy_func, n_shuffles=5)
        
        @test result.interaction_strength >= 0
        @test !isnothing(result.confidence_interval)
        @test haskey(result.metadata, :baseline_score)
        @test haskey(result.metadata, :degradation_both)
        @test haskey(result.metadata, :degradation_joint)
    end
    
    @testset "Categorical Feature Support" begin
        calc = InteractionCalculator(categorical_features=[2, 4])
        
        # Mixed data (continuous and categorical)
        n_samples = 100
        X = randn(n_samples, 4)
        X[:, 2] = rand(1:3, n_samples)  # Categorical
        X[:, 4] = rand(1:2, n_samples)  # Categorical
        
        # MI between continuous and categorical
        result_mixed = calculate_mutual_information(calc, X, 1, 2)
        @test result_mixed.interaction_strength >= 0
        @test result_mixed.interaction_strength <= 1
        
        # MI between two categorical
        result_cat = calculate_mutual_information(calc, X, 2, 4)
        @test result_cat.interaction_strength >= 0
        @test result_cat.interaction_strength <= 1
    end
    
    @testset "Sparse Matrix Operations" begin
        # Create sparse interaction matrix
        n_features = 5
        matrix = SparseInteractionMatrix(n_features, threshold=0.1)
        
        # Add some interactions
        matrix.interactions[1, 2] = 0.5
        matrix.interactions[1, 3] = 0.15
        matrix.interactions[2, 4] = 0.8
        matrix.interactions[3, 5] = 0.05  # Below threshold
        
        # Get significant interactions
        significant = get_significant_interactions(matrix, min_strength=0.1)
        
        @test length(significant) == 3  # Only 3 above threshold
        @test significant[1].interaction_strength == 0.8  # Sorted by strength
        @test significant[1].feature_indices == (2, 4)
        @test significant[2].interaction_strength == 0.5
        @test significant[3].interaction_strength == 0.15
        
        # Test top-k
        top_2 = get_significant_interactions(matrix, top_k=2)
        @test length(top_2) == 2
    end
    
    @testset "Method Combination" begin
        # Create mock results for different methods
        results = [
            InteractionResult((1, 2), nothing, 0.6, nothing, :h_statistic, Dict()),
            InteractionResult((1, 2), nothing, 0.4, nothing, :mutual_info, Dict()),
            InteractionResult((1, 2), nothing, 0.8, nothing, :partial_dependence, Dict()),
            InteractionResult((2, 3), nothing, 0.3, nothing, :h_statistic, Dict()),
            InteractionResult((2, 3), nothing, 0.5, nothing, :mutual_info, Dict())
        ]
        
        # Combine with equal weights
        combined = combine_interaction_methods(results)
        
        @test length(combined) == 2  # Two unique pairs
        
        # Find (1,2) combination
        pair_12 = first(filter(x -> x.feature_indices == (1, 2), combined))
        @test pair_12.interaction_strength ≈ mean([0.6, 0.4, 0.8])
        @test pair_12.method == :combined
        @test haskey(pair_12.metadata, :method_scores)
        
        # Test with custom weights
        weights = [0.5, 0.3, 0.2, 0.0]  # h_stat, mi, pd, perf_deg
        combined_weighted = combine_interaction_methods(results, weights=weights)
        pair_12_weighted = first(filter(x -> x.feature_indices == (1, 2), combined_weighted))
        expected = (0.5 * 0.6 + 0.3 * 0.4 + 0.2 * 0.8) / (0.5 + 0.3 + 0.2)
        @test pair_12_weighted.interaction_strength ≈ expected
    end
    
    @testset "Export Functionality" begin
        # Create test matrix
        n_features = 4
        feature_names = ["A", "B", "C", "D"]
        matrix = SparseInteractionMatrix(n_features, feature_names=feature_names)
        
        # Add interactions
        matrix.interactions[1, 2] = 0.5
        matrix.interactions[1, 3] = 0.3
        matrix.interactions[2, 4] = 0.7
        
        # Export to temporary file
        temp_file = tempname() * ".csv"
        df = export_interaction_heatmap(matrix, temp_file, symmetric=true)
        
        @test size(df) == (4, 5)  # 4 features + 1 feature name column
        @test df.Feature == feature_names
        @test df[1, :B] ≈ 0.5
        @test df[2, :A] ≈ 0.5  # Symmetric
        @test df[1, :C] ≈ 0.3
        @test df[2, :D] ≈ 0.7
        
        # Clean up
        rm(temp_file, force=true)
    end
    
    @testset "Parallel Calculation" begin
        if Threads.nthreads() > 1
            # Generate test data
            n_samples = 50
            n_features = 5
            X = randn(n_samples, n_features)
            y = Int.(X[:, 1] .> 0)
            
            # Train simple model
            model = create_model(:xgboost, :classification, n_estimators=10)
            fitted_model, machine = fit_model!(model, X, y, verbosity=0)
            
            # Calculate interactions in parallel
            calc = InteractionCalculator(method=:mutual_info, n_jobs=2)
            matrix = calculate_all_interactions_parallel(calc, fitted_model, machine, X, y,
                                                       threshold=0.0)
            
            # Check that we got results
            @test nnz(matrix.interactions) > 0
            @test matrix.n_features == n_features
            
            # Number of possible pairs
            n_pairs = n_features * (n_features - 1) ÷ 2
            @test nnz(matrix.interactions) <= n_pairs
        end
    end
    
    @testset "Edge Cases" begin
        # Empty results combination
        @test_throws ErrorException combine_interaction_methods(InteractionResult[])
        
        # Single feature (no interactions possible)
        X_single = randn(50, 1)
        calc = InteractionCalculator()
        matrix = SparseInteractionMatrix(1)
        interactions = get_significant_interactions(matrix)
        @test isempty(interactions)
        
        # All interactions below threshold
        matrix2 = SparseInteractionMatrix(3, threshold=0.5)
        matrix2.interactions[1, 2] = 0.1
        matrix2.interactions[2, 3] = 0.2
        interactions2 = get_significant_interactions(matrix2, min_strength=0.5)
        @test isempty(interactions2)
    end
    
    @testset "Full Integration Test" begin
        # Generate synthetic data with known interactions
        n_samples = 200
        n_features = 6
        X = randn(n_samples, n_features)
        
        # Create interactions:
        # - Strong: features 1 & 2
        # - Medium: features 3 & 4
        # - None: features 5 & 6
        y = Int.((
            X[:, 1] + X[:, 2] + 3 * X[:, 1] .* X[:, 2] +  # Strong
            0.5 * X[:, 3] + 0.5 * X[:, 4] + 0.5 * X[:, 3] .* X[:, 4] +  # Medium
            0.1 * X[:, 5] + 0.1 * X[:, 6] +  # Weak/none
            0.2 * randn(n_samples)
        ) .> 0)
        
        # Train model
        model = create_model(:xgboost, :classification, n_estimators=50, max_depth=4)
        fitted_model, machine = fit_model!(model, X, y, verbosity=0)
        
        # Calculate all interactions
        calc = InteractionCalculator(method=:all, n_samples=150, random_state=42)
        matrix = calculate_all_interactions(calc, fitted_model, machine, X, y,
                                          show_progress=false, threshold=0.05)
        
        # Get top interactions
        top_interactions = get_significant_interactions(matrix, top_k=3)
        
        @test length(top_interactions) >= 1
        @test top_interactions[1].feature_indices in [(1, 2), (2, 1)]  # Should detect strong interaction
        
        # Verify relative strengths
        all_interactions = get_significant_interactions(matrix)
        interaction_12 = first(filter(x -> x.feature_indices == (1, 2) || x.feature_indices == (2, 1), 
                                    all_interactions))
        
        # Find if we have interaction between 3 and 4
        interaction_34_list = filter(x -> x.feature_indices == (3, 4) || x.feature_indices == (4, 3), 
                                   all_interactions)
        
        if !isempty(interaction_34_list)
            interaction_34 = first(interaction_34_list)
            @test interaction_12.interaction_strength > interaction_34.interaction_strength
        end
    end
end

println("All feature interaction tests passed! ✓")