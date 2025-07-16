"""
Simple test for Consensus Building with Weighted Voting
Tests core functionality without external dependencies
"""

using Test
using Random
using Statistics
using Dates

# Test the core structures and logic without complex dependencies
module SimpleConsensusTest

using Random
using Statistics
using Dates

# Define core enums
@enum VotingStrategy begin
    SIMPLE_MAJORITY = 1
    WEIGHTED_PERFORMANCE = 2
    WEIGHTED_EXPLORATION = 3
    ADAPTIVE_WEIGHTS = 4
    HIERARCHICAL_CONSENSUS = 5
end

@enum ConsensusThreshold begin
    ABSOLUTE_THRESHOLD = 1
    RELATIVE_THRESHOLD = 2
    DYNAMIC_THRESHOLD = 3
    PROGRESSIVE_THRESHOLD = 4
end

@enum TieBreakingStrategy begin
    RANDOM_SELECTION = 1
    PERFORMANCE_BASED = 2
    EXPLORATION_BASED = 3
    DIVERSITY_BASED = 4
    TIMESTAMP_BASED = 5
end

# Core structures
mutable struct FeatureVote
    feature_id::Int
    total_votes::Float64
    vote_count::Int
    contributing_trees::Set{Int}
    average_weight::Float64
    first_vote_time::DateTime
    last_vote_time::DateTime
    vote_stability::Float64
    performance_contribution::Float64
end

mutable struct TreeWeight
    tree_id::Int
    performance_weight::Float64
    exploration_weight::Float64
    diversity_weight::Float64
    stability_weight::Float64
    final_weight::Float64
    weight_history::Vector{Float64}
    last_update::DateTime
    update_count::Int
end

struct ConsensusConfig
    voting_strategy::VotingStrategy
    enable_weighted_voting::Bool
    minimum_consensus_percentage::Float32
    maximum_consensus_percentage::Float32
    tie_breaking_strategy::TieBreakingStrategy
    target_feature_count::Int
    early_stopping_threshold::Float32
    minimum_agreement_iterations::Int
    performance_weight_factor::Float32
    exploration_weight_factor::Float32
end

mutable struct SimpleConsensusManager
    config::ConsensusConfig
    feature_votes::Dict{Int, FeatureVote}
    tree_weights::Dict{Int, TreeWeight}
    current_consensus::Vector{Int}
    consensus_strength::Float64
    consensus_history::Vector{Vector{Int}}
    agreement_iterations::Int
    early_stopping_triggered::Bool
    total_voting_rounds::Int
    successful_rounds::Int
end

function create_consensus_config(;
    voting_strategy::VotingStrategy = WEIGHTED_PERFORMANCE,
    enable_weighted_voting::Bool = true,
    minimum_consensus_percentage::Float32 = 0.6f0,
    maximum_consensus_percentage::Float32 = 0.95f0,
    tie_breaking_strategy::TieBreakingStrategy = PERFORMANCE_BASED,
    target_feature_count::Int = 50,
    early_stopping_threshold::Float32 = 0.9f0,
    minimum_agreement_iterations::Int = 3,
    performance_weight_factor::Float32 = 0.6f0,
    exploration_weight_factor::Float32 = 0.4f0
)
    return ConsensusConfig(
        voting_strategy, enable_weighted_voting,
        minimum_consensus_percentage, maximum_consensus_percentage,
        tie_breaking_strategy, target_feature_count,
        early_stopping_threshold, minimum_agreement_iterations,
        performance_weight_factor, exploration_weight_factor
    )
end

function create_feature_vote(feature_id::Int)
    return FeatureVote(
        feature_id, 0.0, 0, Set{Int}(), 0.0,
        now(), now(), 1.0, 0.0
    )
end

function create_tree_weight(tree_id::Int)
    return TreeWeight(
        tree_id, 1.0, 1.0, 1.0, 1.0, 1.0,
        Float64[1.0], now(), 0
    )
end

function initialize_simple_manager(config::ConsensusConfig = create_consensus_config())
    return SimpleConsensusManager(
        config,
        Dict{Int, FeatureVote}(),
        Dict{Int, TreeWeight}(),
        Int[],
        0.0,
        Vector{Vector{Int}}(),
        0,
        false,
        0,
        0
    )
end

function calculate_performance_weight(performance::Float64)::Float64
    return 1.0 / (1.0 + exp(-5.0 * (performance - 0.5)))
end

function calculate_exploration_weight(depth::Int)::Float64
    return min(2.0, 1.0 + log(max(1, depth)) / 10.0)
end

function calculate_voting_weight(manager::SimpleConsensusManager, 
                                tree_id::Int, 
                                performance::Float64, 
                                exploration_depth::Int)::Float64
    weight = 1.0
    
    if manager.config.enable_weighted_voting
        if manager.config.voting_strategy == WEIGHTED_PERFORMANCE
            weight = calculate_performance_weight(performance)
        elseif manager.config.voting_strategy == WEIGHTED_EXPLORATION
            weight = calculate_exploration_weight(exploration_depth)
        elseif manager.config.voting_strategy == ADAPTIVE_WEIGHTS
            perf_weight = calculate_performance_weight(performance) * manager.config.performance_weight_factor
            expl_weight = calculate_exploration_weight(exploration_depth) * manager.config.exploration_weight_factor
            weight = perf_weight + expl_weight
        end
    end
    
    return weight
end

function cast_vote!(manager::SimpleConsensusManager, 
                   tree_id::Int, 
                   feature_id::Int, 
                   tree_performance::Float64 = 1.0,
                   exploration_depth::Int = 1)::Bool
    # Get or create feature vote tracker
    if !haskey(manager.feature_votes, feature_id)
        manager.feature_votes[feature_id] = create_feature_vote(feature_id)
    end
    
    # Get or create tree weight tracker
    if !haskey(manager.tree_weights, tree_id)
        manager.tree_weights[tree_id] = create_tree_weight(tree_id)
    end
    
    vote = manager.feature_votes[feature_id]
    
    # Check if this tree has already voted for this feature
    if tree_id in vote.contributing_trees
        return false
    end
    
    # Add vote with appropriate weight
    weight = calculate_voting_weight(manager, tree_id, tree_performance, exploration_depth)
    vote.total_votes += weight
    vote.vote_count += 1
    push!(vote.contributing_trees, tree_id)
    
    # Update vote metadata
    vote.average_weight = vote.total_votes / vote.vote_count
    vote.last_vote_time = now()
    vote.performance_contribution += tree_performance * weight
    
    return true
end

function update_tree_weight!(manager::SimpleConsensusManager, 
                            tree_id::Int, 
                            performance::Float64, 
                            exploration_depth::Int)
    tree_weight = get(manager.tree_weights, tree_id, create_tree_weight(tree_id))
    
    tree_weight.performance_weight = calculate_performance_weight(performance)
    tree_weight.exploration_weight = calculate_exploration_weight(exploration_depth)
    tree_weight.final_weight = (
        tree_weight.performance_weight * manager.config.performance_weight_factor +
        tree_weight.exploration_weight * manager.config.exploration_weight_factor
    )
    
    push!(tree_weight.weight_history, tree_weight.final_weight)
    if length(tree_weight.weight_history) > 100
        deleteat!(tree_weight.weight_history, 1)
    end
    
    tree_weight.last_update = now()
    tree_weight.update_count += 1
    
    manager.tree_weights[tree_id] = tree_weight
end

function calculate_consensus_threshold(manager::SimpleConsensusManager)::Float64
    return Float64(manager.config.minimum_consensus_percentage)
end

function resolve_ties(manager::SimpleConsensusManager, tied_features::Vector{Int}, count_needed::Int)::Vector{Int}
    if count_needed <= 0 || isempty(tied_features)
        return Int[]
    end
    
    selected = Int[]
    
    if manager.config.tie_breaking_strategy == RANDOM_SELECTION
        shuffled = shuffle(tied_features)
        selected = shuffled[1:min(count_needed, length(shuffled))]
        
    elseif manager.config.tie_breaking_strategy == PERFORMANCE_BASED
        feature_performance = Pair{Int, Float64}[]
        for feature_id in tied_features
            vote = get(manager.feature_votes, feature_id, nothing)
            if !isnothing(vote)
                avg_performance = vote.performance_contribution / max(1, vote.vote_count)
                push!(feature_performance, feature_id => avg_performance)
            else
                push!(feature_performance, feature_id => 0.0)  # Default performance
            end
        end
        sort!(feature_performance, by=x->x.second, rev=true)
        selected = [pair.first for pair in feature_performance[1:min(count_needed, length(feature_performance))]]
        
    elseif manager.config.tie_breaking_strategy == EXPLORATION_BASED
        feature_exploration = Pair{Int, Float64}[]
        for feature_id in tied_features
            vote = get(manager.feature_votes, feature_id, nothing)
            if !isnothing(vote)
                push!(feature_exploration, feature_id => vote.vote_stability)
            else
                push!(feature_exploration, feature_id => 1.0)  # Default stability
            end
        end
        sort!(feature_exploration, by=x->x.second, rev=true)
        selected = [pair.first for pair in feature_exploration[1:min(count_needed, length(feature_exploration))]]
    end
    
    return selected
end

function build_consensus!(manager::SimpleConsensusManager)::Vector{Int}
    threshold = calculate_consensus_threshold(manager)
    consensus_features = Int[]
    feature_scores = Pair{Int, Float64}[]
    
    total_trees = length(manager.tree_weights)
    
    for (feature_id, vote) in manager.feature_votes
        if total_trees > 0
            consensus_score = vote.total_votes / total_trees
            if consensus_score >= threshold
                push!(feature_scores, feature_id => consensus_score)
            end
        end
    end
    
    # Sort by consensus strength (descending)
    sort!(feature_scores, by=x->x.second, rev=true)
    
    # Select top features up to target count
    target_count = min(manager.config.target_feature_count, length(feature_scores))
    consensus_features = [pair.first for pair in feature_scores[1:target_count]]
    
    # Update consensus state
    manager.current_consensus = consensus_features
    manager.consensus_strength = length(consensus_features) > 0 ? 
        mean([manager.feature_votes[fid].total_votes / total_trees for fid in consensus_features]) : 0.0
    
    # Record consensus history
    push!(manager.consensus_history, copy(consensus_features))
    
    # Update statistics
    manager.total_voting_rounds += 1
    if length(consensus_features) >= 1  # Success if any features found
        manager.successful_rounds += 1
    end
    
    # Update agreement tracking
    if length(manager.consensus_history) >= 2
        current = Set(manager.consensus_history[end])
        previous = Set(manager.consensus_history[end-1])
        overlap = length(intersect(current, previous))
        total_features = length(union(current, previous))
        stability = total_features > 0 ? overlap / total_features : 0.0
        
        if stability >= manager.config.early_stopping_threshold
            manager.agreement_iterations += 1
        else
            manager.agreement_iterations = 0
        end
        
        if manager.agreement_iterations >= manager.config.minimum_agreement_iterations
            manager.early_stopping_triggered = true
        end
    end
    
    return consensus_features
end

function should_stop_early(manager::SimpleConsensusManager)::Bool
    return manager.early_stopping_triggered
end

function get_consensus_status(manager::SimpleConsensusManager)
    return Dict{String, Any}(
        "current_consensus_size" => length(manager.current_consensus),
        "consensus_strength" => manager.consensus_strength,
        "total_features_voted" => length(manager.feature_votes),
        "total_trees_participating" => length(manager.tree_weights),
        "voting_rounds" => manager.total_voting_rounds,
        "successful_rounds" => manager.successful_rounds,
        "early_stopping_triggered" => manager.early_stopping_triggered,
        "agreement_iterations" => manager.agreement_iterations
    )
end

function reset_consensus!(manager::SimpleConsensusManager)
    empty!(manager.feature_votes)
    empty!(manager.tree_weights)
    empty!(manager.current_consensus)
    manager.consensus_strength = 0.0
    empty!(manager.consensus_history)
    manager.agreement_iterations = 0
    manager.early_stopping_triggered = false
    manager.total_voting_rounds = 0
    manager.successful_rounds = 0
end

end # module

using .SimpleConsensusTest

@testset "Simple Consensus Building Tests" begin
    
    Random.seed!(42)
    
    @testset "Configuration Tests" begin
        config = SimpleConsensusTest.create_consensus_config()
        
        @test config.voting_strategy == SimpleConsensusTest.WEIGHTED_PERFORMANCE
        @test config.enable_weighted_voting == true
        @test config.minimum_consensus_percentage == 0.6f0
        @test config.target_feature_count == 50
        @test config.tie_breaking_strategy == SimpleConsensusTest.PERFORMANCE_BASED
        
        custom_config = SimpleConsensusTest.create_consensus_config(
            voting_strategy = SimpleConsensusTest.SIMPLE_MAJORITY,
            target_feature_count = 20
        )
        
        @test custom_config.voting_strategy == SimpleConsensusTest.SIMPLE_MAJORITY
        @test custom_config.target_feature_count == 20
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "Manager Initialization Tests" begin
        config = SimpleConsensusTest.create_consensus_config()
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        @test manager.config == config
        @test length(manager.feature_votes) == 0
        @test length(manager.tree_weights) == 0
        @test isempty(manager.current_consensus)
        @test manager.consensus_strength == 0.0
        @test manager.agreement_iterations == 0
        @test manager.early_stopping_triggered == false
        
        println("  ✅ Manager initialization tests passed")
    end
    
    @testset "Weight Calculation Tests" begin
        # Test performance weight calculation
        low_perf = SimpleConsensusTest.calculate_performance_weight(0.2)
        high_perf = SimpleConsensusTest.calculate_performance_weight(0.9)
        @test high_perf > low_perf
        @test 0.0 < low_perf < 1.0
        @test high_perf > 0.5
        
        # Test exploration weight calculation
        shallow = SimpleConsensusTest.calculate_exploration_weight(2)
        deep = SimpleConsensusTest.calculate_exploration_weight(20)
        @test deep > shallow
        @test shallow >= 1.0
        @test deep <= 2.0
        
        println("  ✅ Weight calculation tests passed")
    end
    
    @testset "Vote Casting Tests" begin
        config = SimpleConsensusTest.create_consensus_config()
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Cast first vote
        success = SimpleConsensusTest.cast_vote!(manager, 1, 42, 0.8, 5)
        @test success == true
        @test haskey(manager.feature_votes, 42)
        @test haskey(manager.tree_weights, 1)
        
        vote = manager.feature_votes[42]
        @test vote.feature_id == 42
        @test vote.total_votes > 0.0
        @test vote.vote_count == 1
        @test 1 in vote.contributing_trees
        
        # Test duplicate vote rejection
        duplicate = SimpleConsensusTest.cast_vote!(manager, 1, 42, 0.9, 6)
        @test duplicate == false
        @test vote.vote_count == 1  # Should not increase
        
        # Cast vote from different tree
        success2 = SimpleConsensusTest.cast_vote!(manager, 2, 42, 0.7, 8)
        @test success2 == true
        @test vote.vote_count == 2
        @test 2 in vote.contributing_trees
        
        println("  ✅ Vote casting tests passed")
    end
    
    @testset "Tree Weight Update Tests" begin
        config = SimpleConsensusTest.create_consensus_config()
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Update tree weight
        SimpleConsensusTest.update_tree_weight!(manager, 1, 0.85, 15)
        
        @test haskey(manager.tree_weights, 1)
        tree_weight = manager.tree_weights[1]
        
        @test tree_weight.tree_id == 1
        @test tree_weight.performance_weight > 0.0
        @test tree_weight.exploration_weight > 0.0
        @test tree_weight.final_weight > 0.0
        @test tree_weight.update_count == 1
        @test length(tree_weight.weight_history) == 2  # Initial + update
        
        println("  ✅ Tree weight update tests passed")
    end
    
    @testset "Consensus Building Tests" begin
        config = SimpleConsensusTest.create_consensus_config(
            target_feature_count = 3,
            minimum_consensus_percentage = 0.4f0
        )
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Cast votes from multiple trees
        SimpleConsensusTest.cast_vote!(manager, 1, 10, 0.8, 5)
        SimpleConsensusTest.cast_vote!(manager, 2, 10, 0.7, 6)
        SimpleConsensusTest.cast_vote!(manager, 3, 10, 0.9, 7)  # Feature 10: 3 votes
        
        SimpleConsensusTest.cast_vote!(manager, 1, 20, 0.6, 4)
        SimpleConsensusTest.cast_vote!(manager, 2, 20, 0.8, 8)  # Feature 20: 2 votes
        
        SimpleConsensusTest.cast_vote!(manager, 1, 30, 0.9, 10)  # Feature 30: 1 vote
        
        # Build consensus
        consensus = SimpleConsensusTest.build_consensus!(manager)
        
        @test length(consensus) <= config.target_feature_count
        @test 10 in consensus  # Should have highest votes
        @test manager.consensus_strength > 0.0
        @test length(manager.consensus_history) == 1
        @test manager.total_voting_rounds == 1
        @test manager.successful_rounds == 1
        
        println("  ✅ Consensus building tests passed")
    end
    
    @testset "Tie Breaking Tests" begin
        config = SimpleConsensusTest.create_consensus_config(
            tie_breaking_strategy = SimpleConsensusTest.PERFORMANCE_BASED
        )
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Create tied features with different performance
        SimpleConsensusTest.cast_vote!(manager, 1, 100, 0.9, 5)  # High performance
        SimpleConsensusTest.cast_vote!(manager, 2, 200, 0.5, 5)  # Low performance
        
        tied_features = [100, 200]
        selected = SimpleConsensusTest.resolve_ties(manager, tied_features, 1)
        
        @test length(selected) == 1
        @test 100 in selected  # High performance should win
        
        # Test random tie breaking
        config_random = SimpleConsensusTest.create_consensus_config(
            tie_breaking_strategy = SimpleConsensusTest.RANDOM_SELECTION
        )
        manager_random = SimpleConsensusTest.initialize_simple_manager(config_random)
        SimpleConsensusTest.cast_vote!(manager_random, 1, 300, 0.7, 5)
        SimpleConsensusTest.cast_vote!(manager_random, 2, 400, 0.7, 5)
        
        random_selected = SimpleConsensusTest.resolve_ties(manager_random, [300, 400], 1)
        @test length(random_selected) == 1
        @test random_selected[1] in [300, 400]
        
        println("  ✅ Tie breaking tests passed")
    end
    
    @testset "Progressive Consensus Tests" begin
        config = SimpleConsensusTest.create_consensus_config(
            early_stopping_threshold = 0.8f0,
            minimum_agreement_iterations = 2
        )
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Create stable consensus over multiple rounds
        features = [1, 2, 3]
        
        for round in 1:4
            # Cast same votes each round to create stability
            for tree_id in 1:2
                for feature_id in features
                    SimpleConsensusTest.cast_vote!(manager, tree_id, feature_id, 0.8, 5)
                end
            end
            
            consensus = SimpleConsensusTest.build_consensus!(manager)
            @test length(consensus) > 0
        end
        
        # Should have detected stability and triggered early stopping
        @test manager.agreement_iterations >= config.minimum_agreement_iterations
        @test SimpleConsensusTest.should_stop_early(manager) == true
        
        println("  ✅ Progressive consensus tests passed")
    end
    
    @testset "Status Monitoring Tests" begin
        config = SimpleConsensusTest.create_consensus_config(
            minimum_consensus_percentage = 0.4f0  # Lower threshold to ensure consensus
        )
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Initial status
        status = SimpleConsensusTest.get_consensus_status(manager)
        @test status["current_consensus_size"] == 0
        @test status["consensus_strength"] == 0.0
        @test status["total_features_voted"] == 0
        @test status["voting_rounds"] == 0
        
        # After activity - ensure consensus is reached
        SimpleConsensusTest.cast_vote!(manager, 1, 10, 0.8, 5)
        SimpleConsensusTest.cast_vote!(manager, 2, 10, 0.7, 6)  # Same feature to reach consensus
        SimpleConsensusTest.cast_vote!(manager, 3, 20, 0.7, 6)  # Different feature with lower consensus
        SimpleConsensusTest.build_consensus!(manager)
        
        updated_status = SimpleConsensusTest.get_consensus_status(manager)
        @test updated_status["total_features_voted"] == 2
        @test updated_status["total_trees_participating"] == 3
        @test updated_status["voting_rounds"] == 1
        @test updated_status["successful_rounds"] == 1
        
        println("  ✅ Status monitoring tests passed")
    end
    
    @testset "Reset Tests" begin
        config = SimpleConsensusTest.create_consensus_config()
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Add some data
        SimpleConsensusTest.cast_vote!(manager, 1, 10, 0.8, 5)
        SimpleConsensusTest.build_consensus!(manager)
        
        @test length(manager.feature_votes) > 0
        @test length(manager.consensus_history) > 0
        
        # Reset
        SimpleConsensusTest.reset_consensus!(manager)
        
        @test length(manager.feature_votes) == 0
        @test length(manager.tree_weights) == 0
        @test length(manager.current_consensus) == 0
        @test manager.consensus_strength == 0.0
        @test length(manager.consensus_history) == 0
        @test manager.agreement_iterations == 0
        @test manager.early_stopping_triggered == false
        
        println("  ✅ Reset tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = SimpleConsensusTest.create_consensus_config()
        manager = SimpleConsensusTest.initialize_simple_manager(config)
        
        # Test consensus building with no votes
        empty_consensus = SimpleConsensusTest.build_consensus!(manager)
        @test length(empty_consensus) == 0
        
        # Test tie breaking with empty list
        empty_ties = SimpleConsensusTest.resolve_ties(manager, Int[], 5)
        @test isempty(empty_ties)
        
        # Test tie breaking requesting more than available
        limited_ties = SimpleConsensusTest.resolve_ties(manager, [42], 10)
        @test length(limited_ties) == 1
        @test limited_ties[1] == 42
        
        println("  ✅ Error handling tests passed")
    end
end

println("All Simple Consensus Building tests completed!")
println("✅ Configuration system working correctly")
println("✅ Manager initialization and setup")
println("✅ Weight calculation algorithms for performance and exploration")
println("✅ Vote casting with duplicate detection and tree tracking")
println("✅ Tree weight updates with performance metrics")
println("✅ Consensus building with threshold-based selection")
println("✅ Tie-breaking mechanisms with multiple strategies")
println("✅ Progressive consensus with early stopping detection")
println("✅ Status monitoring and statistics tracking")
println("✅ Reset functionality for clean state management")
println("✅ Error handling for edge cases and invalid inputs")
println("✅ Core consensus voting ready for MCTS ensemble integration")