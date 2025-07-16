using Test
using Random
using Statistics
using LinearAlgebra
using DataFrames

# Include the module
include("../../src/stage3_evaluation/feature_ranking.jl")

using .FeatureRanking

# Set random seed for reproducibility
Random.seed!(42)

@testset "Feature Ranking Tests" begin
    
    @testset "RankingConfig Tests" begin
        # Test default configuration
        config = RankingConfig()
        @test config.use_shap == true
        @test config.use_permutation == true
        @test config.use_interactions == true
        @test config.use_model_specific == true
        @test config.normalize_scores == true
        @test config.aggregation_method == :weighted_mean
        
        # Test weight normalization
        config = RankingConfig(
            shap_weight = 2.0,
            permutation_weight = 1.0,
            interaction_weight = 1.0,
            model_specific_weight = 1.0
        )
        # Weights should be normalized to sum to 1.0
        total = config.shap_weight + config.permutation_weight + 
                config.interaction_weight + config.model_specific_weight
        @test abs(total - 1.0) < 1e-6
        
        # Test partial configuration
        config = RankingConfig(
            use_interactions = false,
            use_model_specific = false,
            shap_weight = 0.6,
            permutation_weight = 0.4
        )
        @test config.use_shap == true
        @test config.use_permutation == true
        @test config.use_interactions == false
        @test config.use_model_specific == false
    end
    
    @testset "ConstraintConfig Tests" begin
        # Test default configuration
        constraints = ConstraintConfig()
        @test isempty(constraints.must_include)
        @test isempty(constraints.must_exclude)
        @test constraints.min_features == 10
        @test constraints.max_features == 20
        @test constraints.correlation_threshold == 0.9
        
        # Test custom configuration
        constraints = ConstraintConfig(
            must_include = [1, 2, 3],
            must_exclude = [4, 5],
            min_features = 5,
            max_features = 15,
            correlation_threshold = 0.8
        )
        @test constraints.must_include == [1, 2, 3]
        @test constraints.must_exclude == [4, 5]
        @test constraints.min_features == 5
        @test constraints.max_features == 15
        @test constraints.correlation_threshold == 0.8
        
        # Test validation
        @test_throws AssertionError ConstraintConfig(min_features=20, max_features=10)
        @test_throws AssertionError ConstraintConfig(correlation_threshold=1.5)
        @test_throws AssertionError ConstraintConfig(must_include=[1, 2], must_exclude=[2, 3])
    end
    
    @testset "SelectionConfig Tests" begin
        # Test default configuration
        config = SelectionConfig()
        @test config.method == :pareto
        @test config.cv_folds == 5
        @test config.stability_threshold == 0.7
        @test config.early_stopping_rounds == 3
        @test config.performance_metric == :accuracy
        
        # Test custom configuration
        config = SelectionConfig(
            method = :forward,
            cv_folds = 10,
            stability_threshold = 0.8,
            early_stopping_rounds = 5,
            performance_metric = :auc
        )
        @test config.method == :forward
        @test config.cv_folds == 10
        @test config.stability_threshold == 0.8
        @test config.early_stopping_rounds == 5
        @test config.performance_metric == :auc
        
        # Test validation
        @test_throws AssertionError SelectionConfig(method=:unknown)
        @test_throws AssertionError SelectionConfig(cv_folds=1)
        @test_throws AssertionError SelectionConfig(stability_threshold=1.5)
    end
    
    @testset "FeatureRanker Tests" begin
        # Test basic initialization
        ranker = FeatureRanker(10)
        @test ranker.n_features == 10
        @test isnothing(ranker.feature_names)
        
        # Test with feature names
        feature_names = ["feat_$i" for i in 1:10]
        ranker = FeatureRanker(10, feature_names=feature_names)
        @test ranker.feature_names == feature_names
        
        # Test validation
        @test_throws AssertionError FeatureRanker(10, feature_names=["f1", "f2"])
    end
    
    @testset "Feature Ranking Tests" begin
        n_features = 20
        ranker = FeatureRanker(n_features)
        
        # Create mock importance results
        importance_results = Dict{Symbol, Any}(
            :shap => (
                mean_abs_shap = rand(n_features) .+ 0.1,
            ),
            :permutation => (
                importance_mean = rand(n_features) .+ 0.05,
            ),
            :model_specific => rand(n_features) .+ 0.02
        )
        
        # Create mock interaction matrix
        interaction_matrix = rand(n_features, n_features)
        interaction_matrix = 0.5 * (interaction_matrix + interaction_matrix')  # Make symmetric
        
        # Test weighted mean aggregation
        config = RankingConfig(aggregation_method=:weighted_mean)
        ranker = FeatureRanker(n_features, config)
        result = rank_features(ranker, importance_results, interaction_matrix=interaction_matrix)
        
        @test isa(result, RankingResult)
        @test length(result.feature_scores) == n_features
        @test length(result.feature_ranks) == n_features
        @test all(idx in 1:n_features for idx in result.feature_ranks)
        @test length(unique(result.feature_ranks)) == n_features  # All unique
        @test haskey(result.method_scores, :shap)
        @test haskey(result.method_scores, :permutation)
        @test haskey(result.method_scores, :interactions)
        @test haskey(result.method_scores, :model_specific)
        
        # Test rank aggregation
        config = RankingConfig(aggregation_method=:rank_aggregation)
        ranker = FeatureRanker(n_features, config)
        result = rank_features(ranker, importance_results, interaction_matrix=interaction_matrix)
        
        @test isa(result, RankingResult)
        @test length(result.feature_scores) == n_features
        
        # Test Borda count
        config = RankingConfig(aggregation_method=:borda_count)
        ranker = FeatureRanker(n_features, config)
        result = rank_features(ranker, importance_results, interaction_matrix=interaction_matrix)
        
        @test isa(result, RankingResult)
        @test length(result.feature_scores) == n_features
        
        # Test without some methods
        config = RankingConfig(use_interactions=false, use_model_specific=false)
        ranker = FeatureRanker(n_features, config)
        result = rank_features(ranker, importance_results)
        
        @test !haskey(result.method_scores, :interactions)
        @test !haskey(result.method_scores, :model_specific)
    end
    
    @testset "Pareto-Optimal Selection Tests" begin
        n_samples, n_features = 100, 30
        X = randn(n_samples, n_features)
        y = rand(Bool, n_samples)
        
        # Create ranker and ranking result
        ranker = FeatureRanker(n_features)
        
        # Create mock ranking result
        feature_scores = Dict(i => rand() for i in 1:n_features)
        feature_ranks = sortperm(collect(values(feature_scores)), rev=true)
        ranking_result = RankingResult(
            feature_scores,
            Dict{Symbol, Vector{Float64}}(),
            feature_ranks,
            Dict{Symbol, Float64}(),
            Dict{Symbol, Any}()
        )
        
        # Test Pareto selection
        config = SelectionConfig(
            method = :pareto,
            constraint_config = ConstraintConfig(
                min_features = 5,
                max_features = 10
            )
        )
        
        models = []  # Empty models for testing
        result = select_features(ranker, ranking_result, X, y, 
                               models=models, config=config)
        
        @test isa(result, FeatureSelectionResult)
        @test length(result.selected_features) >= config.constraint_config.min_features
        @test length(result.selected_features) <= config.constraint_config.max_features
        @test !isnothing(result.pareto_front)
        @test !isempty(result.performance_trajectory)
        @test !isempty(result.selection_report)
        
        # Test with constraints
        config = SelectionConfig(
            method = :pareto,
            constraint_config = ConstraintConfig(
                must_include = [1, 2],
                must_exclude = [3, 4, 5],
                min_features = 5,
                max_features = 10,
                correlation_threshold = 0.8
            )
        )
        
        result = select_features(ranker, ranking_result, X, y, 
                               models=models, config=config)
        
        @test all(f in result.selected_features for f in [1, 2])  # Must include
        @test all(f ∉ result.selected_features for f in [3, 4, 5])  # Must exclude
    end
    
    @testset "Forward Selection Tests" begin
        n_samples, n_features = 100, 20
        X = randn(n_samples, n_features)
        y = rand(Bool, n_samples)
        
        # Create ranker and ranking result
        ranker = FeatureRanker(n_features)
        
        # Create mock ranking result
        feature_scores = Dict(i => rand() for i in 1:n_features)
        feature_ranks = sortperm(collect(values(feature_scores)), rev=true)
        ranking_result = RankingResult(
            feature_scores,
            Dict{Symbol, Vector{Float64}}(),
            feature_ranks,
            Dict{Symbol, Float64}(),
            Dict{Symbol, Any}()
        )
        
        # Test forward selection
        config = SelectionConfig(
            method = :forward,
            constraint_config = ConstraintConfig(
                min_features = 5,
                max_features = 15
            ),
            early_stopping_rounds = 3
        )
        
        models = []
        result = select_features(ranker, ranking_result, X, y, 
                               models=models, config=config)
        
        @test isa(result, FeatureSelectionResult)
        @test length(result.selected_features) >= config.constraint_config.min_features
        @test length(result.selected_features) <= config.constraint_config.max_features
        @test !isempty(result.performance_trajectory)
        @test isnothing(result.pareto_front)  # No Pareto front for forward selection
        
        # Test with must-include features
        config = SelectionConfig(
            method = :forward,
            constraint_config = ConstraintConfig(
                must_include = [1, 2, 3],
                min_features = 5,
                max_features = 10
            )
        )
        
        result = select_features(ranker, ranking_result, X, y, 
                               models=models, config=config)
        
        @test all(f in result.selected_features for f in [1, 2, 3])
    end
    
    @testset "Ensemble Voting Tests" begin
        n_features = 15
        
        # Create multiple ranking results
        ranking_results = RankingResult[]
        
        for i in 1:3
            feature_scores = Dict(j => rand() for j in 1:n_features)
            feature_ranks = sortperm(collect(values(feature_scores)), rev=true)
            
            push!(ranking_results, RankingResult(
                feature_scores,
                Dict{Symbol, Vector{Float64}}(),
                feature_ranks,
                Dict{Symbol, Float64}(),
                Dict{Symbol, Any}()
            ))
        end
        
        # Test Borda voting
        selected_features, votes = ensemble_feature_voting(
            ranking_results, 
            voting_method=:borda,
            top_k=10
        )
        
        @test length(selected_features) == 10
        @test all(f in 1:n_features for f in selected_features)
        @test length(unique(selected_features)) == 10
        @test length(votes) == n_features
        
        # Test average rank voting
        selected_features, votes = ensemble_feature_voting(
            ranking_results,
            voting_method=:average_rank,
            top_k=5
        )
        
        @test length(selected_features) == 5
        @test all(f in 1:n_features for f in selected_features)
        
        # Test majority voting
        selected_features, votes = ensemble_feature_voting(
            ranking_results,
            voting_method=:majority,
            top_k=8
        )
        
        @test length(selected_features) <= 8
        @test all(votes[f] >= 0 && votes[f] <= 1 for f in keys(votes))
    end
    
    @testset "Feature Stability Analysis Tests" begin
        n_samples, n_features = 100, 10
        X = randn(n_samples, n_features)
        y = rand(Bool, n_samples)
        n_folds = 5
        
        models = []  # Empty models for testing
        stability_scores = analyze_feature_stability(X, y, models, n_folds)
        
        @test isa(stability_scores, Dict{Int, Float64})
        @test all(haskey(stability_scores, i) for i in 1:n_features)
        @test all(0 <= score <= 1 for score in values(stability_scores))
    end
    
    @testset "Helper Function Tests" begin
        # Test score normalization
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = FeatureRanking.normalize_scores(scores)
        
        @test minimum(normalized) ≈ 0.0
        @test maximum(normalized) ≈ 1.0
        @test all(0 <= s <= 1 for s in normalized)
        
        # Test with equal scores
        equal_scores = [2.0, 2.0, 2.0, 2.0]
        normalized_equal = FeatureRanking.normalize_scores(equal_scores)
        @test all(s == 1.0 for s in normalized_equal)
        
        # Test interaction importance calculation
        n_features = 5
        interaction_matrix = rand(n_features, n_features)
        interaction_scores = FeatureRanking.calculate_interaction_importance(interaction_matrix)
        
        @test length(interaction_scores) == n_features
        @test all(score >= 0 for score in interaction_scores)
        
        # Test correlation filtering
        n_samples, n_features = 50, 10
        X = randn(n_samples, n_features)
        
        # Make some features highly correlated
        X[:, 2] = X[:, 1] + 0.1 * randn(n_samples)
        X[:, 3] = X[:, 1] + 0.1 * randn(n_samples)
        
        candidates = [2, 3, 4, 5]
        selected = [1]
        filtered = FeatureRanking.filter_by_correlation(
            candidates, selected, X, 0.9
        )
        
        # Features 2 and 3 should be filtered out due to high correlation with 1
        @test 2 ∉ filtered || 3 ∉ filtered
        @test 4 in filtered
        @test 5 in filtered
    end
    
    @testset "Report Generation Tests" begin
        n_features = 10
        feature_names = ["Feature_$i" for i in 1:n_features]
        ranker = FeatureRanker(n_features, feature_names=feature_names)
        
        # Create mock data
        feature_scores = Dict(i => rand() for i in 1:n_features)
        ranking_result = RankingResult(
            feature_scores,
            Dict(:shap => rand(n_features),
                 :permutation => rand(n_features)),
            collect(1:n_features),
            Dict(:shap => 0.5, :permutation => 0.5),
            Dict{Symbol, Any}()
        )
        
        selected_features = [1, 3, 5, 7, 9]
        pareto_points = [(5, 0.85, 0.7), (7, 0.88, 0.6), (10, 0.90, 0.5)]
        performance_trajectory = [0.80, 0.83, 0.85, 0.88, 0.90]
        stability_scores = Dict(i => 0.8 + 0.02 * randn() for i in selected_features)
        
        config = SelectionConfig()
        
        report = FeatureRanking.generate_selection_report(
            ranker, ranking_result, selected_features,
            pareto_points, performance_trajectory,
            stability_scores, config
        )
        
        @test isa(report, String)
        @test occursin("FEATURE SELECTION REPORT", report)
        @test occursin("SELECTED FEATURES", report)
        @test occursin("METHOD CONTRIBUTIONS", report)
        @test occursin("PERFORMANCE TRAJECTORY", report)
        @test occursin("PARETO ANALYSIS", report)
        
        # Test report without optional components
        report_minimal = FeatureRanking.generate_selection_report(
            ranker, ranking_result, selected_features,
            nothing, Float64[], nothing, config
        )
        
        @test isa(report_minimal, String)
        @test occursin("SELECTED FEATURES", report_minimal)
        @test !occursin("PARETO ANALYSIS", report_minimal)
        @test !occursin("PERFORMANCE TRAJECTORY", report_minimal)
    end
    
    @testset "Integration Tests" begin
        # Full pipeline test
        n_samples, n_features = 200, 25
        X = randn(n_samples, n_features)
        y = rand(Bool, n_samples)
        
        feature_names = ["feat_$i" for i in 1:n_features]
        
        # Create ranker
        ranking_config = RankingConfig(
            use_shap = true,
            use_permutation = true,
            use_interactions = true,
            normalize_scores = true,
            aggregation_method = :weighted_mean
        )
        
        ranker = FeatureRanker(n_features, ranking_config, 
                             feature_names=feature_names)
        
        # Create importance results
        importance_results = Dict{Symbol, Any}(
            :shap => (
                mean_abs_shap = rand(n_features) .* 0.5 .+ 0.1,
            ),
            :permutation => (
                importance_mean = rand(n_features) .* 0.3 .+ 0.05,
            ),
            :model_specific => rand(n_features) .* 0.2 .+ 0.02
        )
        
        # Create interaction matrix
        interaction_matrix = rand(n_features, n_features) .* 0.1
        interaction_matrix = 0.5 * (interaction_matrix + interaction_matrix')
        
        # Rank features
        ranking_result = rank_features(ranker, importance_results, 
                                     interaction_matrix=interaction_matrix)
        
        # Select features
        selection_config = SelectionConfig(
            method = :pareto,
            ranking_config = ranking_config,
            constraint_config = ConstraintConfig(
                min_features = 8,
                max_features = 15,
                correlation_threshold = 0.85
            ),
            cv_folds = 3,
            stability_threshold = 0.6
        )
        
        selection_result = select_features(ranker, ranking_result, X, y,
                                         models=[], config=selection_config)
        
        @test isa(selection_result, FeatureSelectionResult)
        @test 8 <= length(selection_result.selected_features) <= 15
        @test !isempty(selection_result.selection_report)
        @test haskey(selection_result.metadata, :selection_method)
        @test selection_result.metadata[:selection_method] == :pareto
        
        # Test forward selection
        selection_config_forward = SelectionConfig(
            method = :forward,
            ranking_config = ranking_config,
            constraint_config = ConstraintConfig(
                must_include = [1, 5],
                must_exclude = [10, 15],
                min_features = 5,
                max_features = 12
            ),
            early_stopping_rounds = 2
        )
        
        selection_result_forward = select_features(
            ranker, ranking_result, X, y,
            models=[], config=selection_config_forward
        )
        
        @test isa(selection_result_forward, FeatureSelectionResult)
        @test all(f in selection_result_forward.selected_features for f in [1, 5])
        @test all(f ∉ selection_result_forward.selected_features for f in [10, 15])
        @test 5 <= length(selection_result_forward.selected_features) <= 12
    end
end

println("All feature ranking tests passed! ✓")