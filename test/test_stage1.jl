using Test
include("../src/stage1_filter.jl")

@testset "Stage 1 Tests" begin
    # Simple synthetic data
    X = randn(100, 20)
    y = X[:, 1] + 0.5 * X[:, 2] + randn(100) * 0.1  # y depends on features 1,2
    feature_names = ["f$i" for i in 1:20]
    
    X_filtered, selected_features, indices = stage1_filter(X, y, feature_names, 5)
    
    @test size(X_filtered, 2) == 5
    @test length(selected_features) == 5
    @test 1 in indices  # Feature 1 should be selected
    @test 2 in indices  # Feature 2 should be selected
end