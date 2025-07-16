using Test
include("../src/stage3_evaluation.jl")

@testset "Stage 3 Tests" begin
    X = randn(100, 8)
    y = rand([0, 1], 100)  # Binary classification
    feature_names = ["f$i" for i in 1:8]
    
    final_features, score, model = stage3_evaluation(X, y, feature_names, 3, "binary_classification")
    
    @test length(final_features) <= 3
    @test 0 <= score <= 1
    @test model == "SimpleCorrelation"
end