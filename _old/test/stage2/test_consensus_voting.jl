"""
Test Suite for Consensus Building with Weighted Voting
Validates weighted voting mechanisms, consensus thresholds, tie-breaking rules,
progressive consensus, and early stopping functionality for MCTS ensemble.
"""

using Test
using Random
using Statistics
using Dates

# Include the consensus voting module
include("../../src/stage2/consensus_voting.jl")
using .ConsensusVoting

@testset "Consensus Building with Weighted Voting Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = create_consensus_config()
        
        @test config.voting_strategy == WEIGHTED_PERFORMANCE
        @test config.enable_weighted_voting == true
        @test config.weight_decay_rate == 0.95f0
        @test config.consensus_threshold_type == DYNAMIC_THRESHOLD
        @test config.minimum_consensus_percentage == 0.6f0
        @test config.maximum_consensus_percentage == 0.95f0
        @test config.tie_breaking_strategy == PERFORMANCE_BASED
        @test config.enable_progressive_consensus == true
        @test config.target_feature_count == 50
        
        # Test custom configuration
        custom_config = create_consensus_config(
            voting_strategy = SIMPLE_MAJORITY,
            target_feature_count = 100,
            minimum_consensus_percentage = 0.5f0,
            tie_breaking_strategy = RANDOM_SELECTION
        )
        
        @test custom_config.voting_strategy == SIMPLE_MAJORITY
        @test custom_config.target_feature_count == 100
        @test custom_config.minimum_consensus_percentage == 0.5f0
        @test custom_config.tie_breaking_strategy == RANDOM_SELECTION
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "Enum Value Tests" begin
        # Test voting strategy enum values
        @test Int(SIMPLE_MAJORITY) == 1
        @test Int(WEIGHTED_PERFORMANCE) == 2
        @test Int(WEIGHTED_EXPLORATION) == 3
        @test Int(ADAPTIVE_WEIGHTS) == 4
        @test Int(HIERARCHICAL_CONSENSUS) == 5
        
        # Test consensus threshold enum values
        @test Int(ABSOLUTE_THRESHOLD) == 1
        @test Int(RELATIVE_THRESHOLD) == 2
        @test Int(DYNAMIC_THRESHOLD) == 3
        @test Int(PROGRESSIVE_THRESHOLD) == 4
        
        # Test tie-breaking strategy enum values
        @test Int(RANDOM_SELECTION) == 1
        @test Int(PERFORMANCE_BASED) == 2
        @test Int(EXPLORATION_BASED) == 3
        @test Int(DIVERSITY_BASED) == 4
        @test Int(TIMESTAMP_BASED) == 5
        
        println("  ✅ Enum value tests passed")
    end
    
    @testset "Feature Vote Creation Tests" begin
        vote = ConsensusVoting.create_feature_vote(42)
        
        @test vote.feature_id == 42
        @test vote.total_votes == 0.0
        @test vote.vote_count == 0
        @test isempty(vote.contributing_trees)
        @test vote.average_weight == 0.0
        @test vote.vote_stability == 1.0
        @test vote.performance_contribution == 0.0
        
        println("  ✅ Feature vote creation tests passed")
    end
    
    @testset "Tree Weight Creation Tests" begin
        weight = ConsensusVoting.create_tree_weight(5)
        
        @test weight.tree_id == 5
        @test weight.performance_weight == 1.0
        @test weight.exploration_weight == 1.0
        @test weight.diversity_weight == 1.0
        @test weight.stability_weight == 1.0
        @test weight.final_weight == 1.0
        @test length(weight.weight_history) == 1
        @test weight.weight_history[1] == 1.0
        @test weight.update_count == 0
        
        println("  ✅ Tree weight creation tests passed")
    end
    
    @testset "Manager Initialization Tests" begin
        config = create_consensus_config(target_feature_count = 30)
        manager = initialize_consensus_manager(config)
        
        @test manager.config == config
        @test length(manager.feature_votes) == 0
        @test length(manager.tree_weights) == 0
        @test isempty(manager.current_consensus)
        @test manager.consensus_strength == 0.0
        @test length(manager.consensus_history) == 0
        @test manager.agreement_iterations == 0
        @test manager.early_stopping_triggered == false
        @test manager.manager_state == "active"
        
        println("  ✅ Manager initialization tests passed")
    end
    
    @testset "Weight Calculation Tests" begin
        config = create_consensus_config()
        manager = initialize_consensus_manager(config)
        
        # Test performance weight calculation
        perf_weight = ConsensusVoting.calculate_performance_weight(0.8)
        @test 0.0 < perf_weight <= 2.0
        
        high_perf_weight = ConsensusVoting.calculate_performance_weight(0.9)
        low_perf_weight = ConsensusVoting.calculate_performance_weight(0.3)
        @test high_perf_weight > low_perf_weight
        
        # Test exploration weight calculation
        shallow_weight = ConsensusVoting.calculate_exploration_weight(5)
        deep_weight = ConsensusVoting.calculate_exploration_weight(50)
        @test deep_weight > shallow_weight
        @test deep_weight <= 2.0
        
        # Test adaptive weight calculation
        adaptive_weight = ConsensusVoting.calculate_adaptive_weight(manager, 1, 0.8, 10)
        @test adaptive_weight > 0.0
        
        println("  ✅ Weight calculation tests passed")
    end
    
    @testset "Vote Casting Tests" begin
        config = create_consensus_config(target_feature_count = 10)
        manager = initialize_consensus_manager(config)
        
        # Cast first vote
        success = cast_vote!(manager, 1, 42, 0.8, 5)
        @test success == true
        @test haskey(manager.feature_votes, 42)
        @test haskey(manager.tree_weights, 1)
        
        vote = manager.feature_votes[42]
        @test vote.feature_id == 42
        @test vote.total_votes > 0.0
        @test vote.vote_count == 1
        @test 1 in vote.contributing_trees
        
        # Test duplicate vote from same tree
        duplicate = cast_vote!(manager, 1, 42, 0.9, 6)
        @test duplicate == false  # Should reject duplicate
        @test vote.vote_count == 1  # Count should not increase
        
        # Cast vote from different tree
        success2 = cast_vote!(manager, 2, 42, 0.7, 8)
        @test success2 == true
        @test vote.vote_count == 2
        @test 2 in vote.contributing_trees
        
        # Cast votes for different features
        cast_vote!(manager, 1, 100, 0.6, 3)
        cast_vote!(manager, 2, 200, 0.9, 7)
        
        @test length(manager.feature_votes) == 3
        @test haskey(manager.feature_votes, 100)
        @test haskey(manager.feature_votes, 200)
        
        println("  ✅ Vote casting tests passed")
    end
    
    @testset "Tree Weight Update Tests" begin
        config = create_consensus_config()
        manager = initialize_consensus_manager(config)
        
        # Update tree weight
        update_tree_weight!(manager, 1, 0.85, 15, 0.9, 0.8)
        
        @test haskey(manager.tree_weights, 1)
        tree_weight = manager.tree_weights[1]
        
        @test tree_weight.tree_id == 1
        @test tree_weight.performance_weight > 0.0
        @test tree_weight.exploration_weight > 0.0
        @test tree_weight.diversity_weight == 0.9
        @test tree_weight.stability_weight == 0.8
        @test tree_weight.final_weight > 0.0
        @test tree_weight.update_count == 1
        @test length(tree_weight.weight_history) == 2  # Initial + update
        
        # Test multiple updates
        initial_weight = tree_weight.final_weight
        update_tree_weight!(manager, 1, 0.95, 25, 0.95, 0.9)
        
        @test tree_weight.update_count == 2
        @test length(tree_weight.weight_history) == 3
        
        println("  ✅ Tree weight update tests passed")
    end
    
    @testset "Consensus Building Tests" begin
        config = create_consensus_config(
            target_feature_count = 5,
            minimum_consensus_percentage = 0.5f0,
            consensus_threshold_type = ABSOLUTE_THRESHOLD
        )
        manager = initialize_consensus_manager(config)
        
        # Cast votes from multiple trees
        # Feature 1: 3 votes (should be selected)
        cast_vote!(manager, 1, 1, 0.8, 5)
        cast_vote!(manager, 2, 1, 0.7, 6)
        cast_vote!(manager, 3, 1, 0.9, 7)
        
        # Feature 2: 2 votes (should be selected)
        cast_vote!(manager, 1, 2, 0.6, 4)
        cast_vote!(manager, 4, 2, 0.8, 8)
        
        # Feature 3: 1 vote (may not be selected depending on threshold)
        cast_vote!(manager, 2, 3, 0.9, 9)
        
        # Feature 4: 4 votes (should definitely be selected)
        cast_vote!(manager, 1, 4, 0.9, 10)
        cast_vote!(manager, 2, 4, 0.8, 8)
        cast_vote!(manager, 3, 4, 0.7, 6)
        cast_vote!(manager, 4, 4, 0.85, 9)
        
        # Build consensus
        consensus = build_consensus!(manager)
        
        @test length(consensus) <= config.target_feature_count
        @test 1 in consensus  # Should have high votes
        @test 4 in consensus  # Should have highest votes
        @test manager.consensus_strength > 0.0
        @test length(manager.consensus_history) == 1
        @test manager.stats.total_voting_rounds == 1
        
        println("  ✅ Consensus building tests passed")
    end
    
    @testset "Consensus Threshold Tests" begin
        config = create_consensus_config(
            minimum_consensus_percentage = 0.6f0,
            maximum_consensus_percentage = 0.9f0
        )
        manager = initialize_consensus_manager(config)
        
        # Test absolute threshold
        manager.config = create_consensus_config(consensus_threshold_type = ABSOLUTE_THRESHOLD)
        threshold = ConsensusVoting.calculate_consensus_threshold(manager)
        @test threshold == 0.6  # Should equal minimum percentage
        
        # Test relative threshold with some votes
        cast_vote!(manager, 1, 10, 0.8, 5)
        cast_vote!(manager, 2, 10, 0.7, 6)
        manager.config = create_consensus_config(consensus_threshold_type = RELATIVE_THRESHOLD)
        rel_threshold = ConsensusVoting.calculate_consensus_threshold(manager)
        @test rel_threshold >= 0.6
        
        # Test progressive threshold
        manager.stats.total_voting_rounds = 100
        manager.config = create_consensus_config(consensus_threshold_type = PROGRESSIVE_THRESHOLD)
        prog_threshold = ConsensusVoting.calculate_consensus_threshold(manager)
        @test prog_threshold > 0.6  # Should increase with rounds
        
        println("  ✅ Consensus threshold tests passed")
    end
    
    @testset "Tie Breaking Tests" begin
        config = create_consensus_config(
            target_feature_count = 2,
            tie_breaking_strategy = PERFORMANCE_BASED,
            minimum_consensus_percentage = 0.4f0
        )
        manager = initialize_consensus_manager(config)
        
        # Create tied features with equal vote counts but different performance
        cast_vote!(manager, 1, 100, 0.9, 5)  # High performance
        cast_vote!(manager, 2, 200, 0.5, 5)  # Low performance
        cast_vote!(manager, 3, 300, 0.8, 5)  # Medium performance
        
        # All features have 1 vote each, so should tie
        tied_features = [100, 200, 300]
        selected = ConsensusVoting.resolve_ties(manager, tied_features, 2)
        
        @test length(selected) == 2
        @test 100 in selected  # High performance should be selected
        @test 200 ∉ selected   # Low performance should not be selected
        
        # Test random tie breaking
        manager.config = create_consensus_config(tie_breaking_strategy = RANDOM_SELECTION)
        random_selected = ConsensusVoting.resolve_ties(manager, tied_features, 1)
        @test length(random_selected) == 1
        @test random_selected[1] in tied_features
        
        println("  ✅ Tie breaking tests passed")
    end
    
    @testset "Progressive Consensus Tests" begin
        config = create_consensus_config(
            enable_progressive_consensus = true,
            early_stopping_threshold = 0.8f0,
            minimum_agreement_iterations = 3,
            consensus_stability_window = 5
        )
        manager = initialize_consensus_manager(config)
        
        # Create stable consensus over multiple rounds
        features = [1, 2, 3, 4, 5]
        
        for round in 1:5
            # Cast same votes each round to create stability
            for tree_id in 1:3
                for feature_id in features
                    cast_vote!(manager, tree_id, feature_id, 0.8, 5)
                end
            end
            
            consensus = build_consensus!(manager)
            @test length(consensus) > 0
        end
        
        # Should have detected stability
        @test length(manager.stability_window) > 0
        @test manager.stats.consensus_stability_score > 0.0
        
        # Test early stopping detection
        if manager.agreement_iterations >= config.minimum_agreement_iterations
            @test should_stop_early(manager) == manager.early_stopping_triggered
        end
        
        println("  ✅ Progressive consensus tests passed")
    end
    
    @testset "Status and Monitoring Tests" begin
        config = create_consensus_config()
        manager = initialize_consensus_manager(config)
        
        # Initial status
        status = get_consensus_status(manager)
        @test haskey(status, "manager_state")
        @test haskey(status, "current_consensus_size")
        @test haskey(status, "consensus_strength")
        @test haskey(status, "total_features_voted")
        @test haskey(status, "voting_rounds")
        
        @test status["manager_state"] == "active"
        @test status["current_consensus_size"] == 0
        @test status["consensus_strength"] == 0.0
        @test status["voting_rounds"] == 0
        
        # After some activity
        cast_vote!(manager, 1, 10, 0.8, 5)
        cast_vote!(manager, 2, 20, 0.7, 6)
        build_consensus!(manager)
        
        updated_status = get_consensus_status(manager)
        @test updated_status["total_features_voted"] == 2
        @test updated_status["voting_rounds"] == 1
        @test updated_status["current_consensus_size"] > 0
        
        println("  ✅ Status and monitoring tests passed")
    end
    
    @testset "Feature Voting Details Tests" begin
        config = create_consensus_config()
        manager = initialize_consensus_manager(config)
        
        # Cast some votes
        cast_vote!(manager, 1, 42, 0.9, 8)
        cast_vote!(manager, 2, 42, 0.7, 6)
        
        # Get voting details
        details = get_feature_voting_details(manager, 42)
        @test !isnothing(details)
        @test details["feature_id"] == 42
        @test details["vote_count"] == 2
        @test length(details["contributing_trees"]) == 2
        @test 1 in details["contributing_trees"]
        @test 2 in details["contributing_trees"]
        @test details["total_votes"] > 0.0
        
        # Test non-existent feature
        no_details = get_feature_voting_details(manager, 999)
        @test isnothing(no_details)
        
        println("  ✅ Feature voting details tests passed")
    end
    
    @testset "Report Generation Tests" begin
        config = create_consensus_config()
        manager = initialize_consensus_manager(config)
        
        # Add some activity
        cast_vote!(manager, 1, 10, 0.8, 5)
        cast_vote!(manager, 2, 20, 0.7, 6)
        cast_vote!(manager, 3, 10, 0.9, 7)
        build_consensus!(manager)
        
        # Generate report
        report = generate_consensus_report(manager)
        
        @test contains(report, "Consensus Building Report")
        @test contains(report, "Manager State: active")
        @test contains(report, "Current Consensus:")
        @test contains(report, "Voting Performance:")
        @test contains(report, "Progressive Consensus:")
        @test contains(report, "Configuration:")
        
        # Should contain current consensus info
        @test contains(report, "Selected Features:")
        @test contains(report, "Consensus Strength:")
        
        println("  ✅ Report generation tests passed")
    end
    
    @testset "Reset and Cleanup Tests" begin
        config = create_consensus_config()
        manager = initialize_consensus_manager(config)
        
        # Add some data
        cast_vote!(manager, 1, 10, 0.8, 5)
        cast_vote!(manager, 2, 20, 0.7, 6)
        build_consensus!(manager)
        
        @test length(manager.feature_votes) > 0
        @test length(manager.tree_weights) > 0
        @test length(manager.consensus_history) > 0
        
        # Test reset
        reset_consensus!(manager)
        
        @test length(manager.feature_votes) == 0
        @test length(manager.tree_weights) == 0
        @test length(manager.current_consensus) == 0
        @test manager.consensus_strength == 0.0
        @test length(manager.consensus_history) == 0
        @test manager.agreement_iterations == 0
        @test manager.early_stopping_triggered == false
        
        # Test cleanup
        cleanup_consensus_manager!(manager)
        @test manager.manager_state == "shutdown"
        
        println("  ✅ Reset and cleanup tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = create_consensus_config()
        manager = initialize_consensus_manager(config)
        
        # Test building consensus with no votes
        empty_consensus = build_consensus!(manager)
        @test length(empty_consensus) == 0
        
        # Test voting with invalid tree/feature IDs (should work - IDs are just integers)
        success = cast_vote!(manager, -1, -1, 0.5, 1)
        @test success == true  # Should accept any integer IDs
        
        # Test tie breaking with empty list
        empty_ties = ConsensusVoting.resolve_ties(manager, Int[], 5)
        @test isempty(empty_ties)
        
        # Test tie breaking requesting more than available
        single_feature = [42]
        limited_ties = ConsensusVoting.resolve_ties(manager, single_feature, 10)
        @test length(limited_ties) == 1
        @test limited_ties[1] == 42
        
        println("  ✅ Error handling tests passed")
    end
    
    @testset "Performance and Caching Tests" begin
        config = create_consensus_config(
            cache_weight_calculations = true,
            enable_incremental_updates = true
        )
        manager = initialize_consensus_manager(config)
        
        # Test weight caching
        weight1 = ConsensusVoting.calculate_voting_weight(manager, 1, 0.8, 5)
        weight2 = ConsensusVoting.calculate_voting_weight(manager, 1, 0.8, 5)  # Same parameters
        @test weight1 == weight2  # Should get cached result
        @test length(manager.weight_cache) > 0
        
        # Test performance tracking
        initial_times = length(manager.voting_times)
        build_consensus!(manager)
        @test length(manager.voting_times) == initial_times + 1
        
        # Test weight calculation timing
        initial_weight_times = length(manager.weight_calculation_times)
        ConsensusVoting.calculate_voting_weight(manager, 2, 0.7, 8)
        @test length(manager.weight_calculation_times) == initial_weight_times + 1
        
        println("  ✅ Performance and caching tests passed")
    end
end

println("All Consensus Building with Weighted Voting tests completed!")
println("✅ Configuration and enum validation")
println("✅ Feature vote and tree weight creation")
println("✅ Manager initialization and setup")
println("✅ Weight calculation algorithms (performance, exploration, adaptive)")
println("✅ Vote casting with duplicate detection")
println("✅ Tree weight updates with history tracking")
println("✅ Consensus building with various threshold strategies")
println("✅ Dynamic consensus threshold calculation")
println("✅ Tie-breaking with multiple strategies")
println("✅ Progressive consensus and early stopping detection")
println("✅ Status monitoring and detailed reporting")
println("✅ Feature voting details and metadata tracking")
println("✅ Report generation with comprehensive statistics")
println("✅ Reset and cleanup functionality")
println("✅ Error handling for edge cases")
println("✅ Performance optimization and caching mechanisms")
println("✅ Ready for MCTS ensemble consensus aggregation")