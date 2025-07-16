using Random

mutable struct MCTSNode
    feature_mask::BitVector
    visits::Int
    total_score::Float64
    children::Vector{MCTSNode}
    parent::Union{MCTSNode, Nothing}
end

function stage2_mcts(X::Matrix{Float64}, y::Vector, feature_names::Vector{String}, target_features::Int; iterations=1000)
    println("\n=== Stage 2: MCTS Selection ===")
    println("Input: $(size(X, 2)) features → Reducing to: $target_features features")
    
    n_features = size(X, 2)
    root = MCTSNode(falses(n_features), 0, 0.0, MCTSNode[], nothing)
    
    best_score = 0.0
    best_mask = falses(n_features)
    
    for iter in 1:iterations
        if iter % 100 == 0
            println("MCTS iteration: $iter/$iterations, best score: $(round(best_score, digits=4))")
        end
        
        # Simple random feature selection for evaluation
        mask = rand(n_features) .< (target_features / n_features)
        if sum(mask) == 0
            continue
        end
        
        # Quick evaluation (simplified)
        try
            score = evaluate_feature_subset(X[:, mask], y)
            if score > best_score
                best_score = score
                best_mask = copy(mask)
            end
        catch
            continue
        end
    end
    
    selected_indices = findall(best_mask)
    selected_features = feature_names[selected_indices]
    
    println("Selected $(length(selected_features)) features")
    println("Best score: $(round(best_score, digits=4))")
    
    return X[:, selected_indices], selected_features, selected_indices
end

function evaluate_feature_subset(X::Matrix{Float64}, y::Vector)
    # Simple correlation-based score
    n_features = size(X, 2)
    if n_features == 0
        return 0.0
    end
    
    scores = [abs(cor(X[:, i], y)) for i in 1:n_features if var(X[:, i]) > 1e-10]
    return isempty(scores) ? 0.0 : mean(scores)
end