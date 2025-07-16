using Test
include("../src/stage2_mcts.jl")

@testset "Stage 2 Tests" begin
    X = randn(50, 10)
    y = randn(50)
    feature_names = ["f$i" for i in 1:10]
    
    X_selected, selected_features, indices = stage2_mcts(X, y, feature_names, 5, iterations=100)
    
    @test size(X_selected, 2) <= 5
    @test length(selected_features) == size(X_selected, 2)
end