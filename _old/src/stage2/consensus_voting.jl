"""
Consensus Building with Weighted Voting for MCTS Ensemble Feature Selection
Implements consensus mechanism aggregating feature selections from all trees using weighted voting,
with tree performance-based weighting, consensus thresholds, tie-breaking rules, and progressive
consensus for early stopping when strong agreement emerges.

This module provides weighted voting across ensemble trees, consensus threshold detection,
and dynamic weight calculation based on tree performance and exploration metrics.
"""

module ConsensusVoting

using Random
using Statistics
using Dates
using Printf
using LinearAlgebra

# Import ensemble forest for tree management
include("ensemble_forest.jl")
using .EnsembleForest

# Import diversity mechanisms for tree tracking
include("diversity_mechanisms.jl")
using .DiversityMechanisms

"""
Voting strategy types for different consensus approaches
"""
@enum VotingStrategy begin
    SIMPLE_MAJORITY = 1        # Simple majority voting
    WEIGHTED_PERFORMANCE = 2   # Weight by tree performance
    WEIGHTED_EXPLORATION = 3   # Weight by exploration depth
    ADAPTIVE_WEIGHTS = 4       # Dynamic weight adaptation
    HIERARCHICAL_CONSENSUS = 5 # Multi-level consensus building
end

"""
Consensus threshold types for different stopping criteria
"""
@enum ConsensusThreshold begin
    ABSOLUTE_THRESHOLD = 1     # Fixed percentage threshold
    RELATIVE_THRESHOLD = 2     # Threshold relative to best features
    DYNAMIC_THRESHOLD = 3      # Threshold adapts to vote distribution
    PROGRESSIVE_THRESHOLD = 4  # Threshold increases over time
end

"""
Tie-breaking strategy for features with equal votes
"""
@enum TieBreakingStrategy begin
    RANDOM_SELECTION = 1       # Random tie-breaking
    PERFORMANCE_BASED = 2      # Based on contributing tree performance
    EXPLORATION_BASED = 3      # Based on exploration metrics
    DIVERSITY_BASED = 4        # Based on feature diversity contribution
    TIMESTAMP_BASED = 5        # Based on selection time order
end

"""
Feature vote tracking information
"""
mutable struct FeatureVote
    feature_id::Int                    # Feature identifier
    total_votes::Float64              # Total weighted votes
    vote_count::Int                   # Number of trees voting
    contributing_trees::Set{Int}      # Tree IDs that voted for this feature
    average_weight::Float64           # Average weight of voting trees
    first_vote_time::DateTime         # When first vote was cast
    last_vote_time::DateTime          # When last vote was cast
    vote_stability::Float64           # How stable votes are over time
    performance_contribution::Float64  # Performance contribution from this feature
end

"""
Create new feature vote tracker
"""
function create_feature_vote(feature_id::Int)
    return FeatureVote(
        feature_id,
        0.0,  # No votes initially
        0,    # No contributing trees
        Set{Int}(),
        0.0,  # No average weight
        now(), now(),
        1.0,  # Fully stable initially
        0.0   # No performance contribution
    )
end

"""
Tree voting weight based on performance and exploration
"""
mutable struct TreeWeight
    tree_id::Int                    # Tree identifier
    performance_weight::Float64    # Weight based on tree performance
    exploration_weight::Float64    # Weight based on exploration depth
    diversity_weight::Float64      # Weight based on diversity contribution
    stability_weight::Float64      # Weight based on selection stability
    final_weight::Float64         # Combined final weight
    weight_history::Vector{Float64} # Historical weight values
    last_update::DateTime          # Last weight update time
    update_count::Int              # Number of weight updates
end

"""
Create new tree weight tracker
"""
function create_tree_weight(tree_id::Int)
    return TreeWeight(
        tree_id,
        1.0,  # Default equal weight
        1.0,  # Default equal weight
        1.0,  # Default equal weight
        1.0,  # Default equal weight
        1.0,  # Combined default weight
        Float64[1.0],
        now(),
        0
    )
end

"""
Consensus building configuration
"""
struct ConsensusConfig
    # Voting configuration
    voting_strategy::VotingStrategy
    enable_weighted_voting::Bool
    weight_decay_rate::Float32
    
    # Threshold configuration
    consensus_threshold_type::ConsensusThreshold
    minimum_consensus_percentage::Float32
    maximum_consensus_percentage::Float32
    dynamic_threshold_factor::Float32
    
    # Tie-breaking configuration
    tie_breaking_strategy::TieBreakingStrategy
    tie_breaking_randomization::Bool
    tie_breaking_tolerance::Float32
    
    # Progressive consensus
    enable_progressive_consensus::Bool
    early_stopping_threshold::Float32
    consensus_stability_window::Int
    minimum_agreement_iterations::Int
    
    # Weight calculation
    performance_weight_factor::Float32
    exploration_weight_factor::Float32
    diversity_weight_factor::Float32
    stability_weight_factor::Float32
    
    # Feature selection
    target_feature_count::Int
    minimum_feature_count::Int
    maximum_feature_count::Int
    feature_selection_buffer::Int
    
    # Monitoring and validation
    enable_vote_tracking::Bool
    enable_consensus_monitoring::Bool
    vote_history_size::Int
    validation_frequency::Int
    
    # Performance optimization
    enable_incremental_updates::Bool
    batch_update_size::Int
    parallel_weight_calculation::Bool
    cache_weight_calculations::Bool
end

"""
Create default consensus configuration
"""
function create_consensus_config(;
    voting_strategy::VotingStrategy = WEIGHTED_PERFORMANCE,
    enable_weighted_voting::Bool = true,
    weight_decay_rate::Float32 = 0.95f0,
    consensus_threshold_type::ConsensusThreshold = DYNAMIC_THRESHOLD,
    minimum_consensus_percentage::Float32 = 0.6f0,
    maximum_consensus_percentage::Float32 = 0.95f0,
    dynamic_threshold_factor::Float32 = 0.8f0,
    tie_breaking_strategy::TieBreakingStrategy = PERFORMANCE_BASED,
    tie_breaking_randomization::Bool = false,
    tie_breaking_tolerance::Float32 = 0.01f0,
    enable_progressive_consensus::Bool = true,
    early_stopping_threshold::Float32 = 0.9f0,
    consensus_stability_window::Int = 100,
    minimum_agreement_iterations::Int = 50,
    performance_weight_factor::Float32 = 0.4f0,
    exploration_weight_factor::Float32 = 0.3f0,
    diversity_weight_factor::Float32 = 0.2f0,
    stability_weight_factor::Float32 = 0.1f0,
    target_feature_count::Int = 50,
    minimum_feature_count::Int = 10,
    maximum_feature_count::Int = 100,
    feature_selection_buffer::Int = 5,
    enable_vote_tracking::Bool = true,
    enable_consensus_monitoring::Bool = true,
    vote_history_size::Int = 1000,
    validation_frequency::Int = 100,
    enable_incremental_updates::Bool = true,
    batch_update_size::Int = 10,
    parallel_weight_calculation::Bool = true,
    cache_weight_calculations::Bool = true
)
    return ConsensusConfig(
        voting_strategy, enable_weighted_voting, weight_decay_rate,
        consensus_threshold_type, minimum_consensus_percentage, maximum_consensus_percentage, dynamic_threshold_factor,
        tie_breaking_strategy, tie_breaking_randomization, tie_breaking_tolerance,
        enable_progressive_consensus, early_stopping_threshold, consensus_stability_window, minimum_agreement_iterations,
        performance_weight_factor, exploration_weight_factor, diversity_weight_factor, stability_weight_factor,
        target_feature_count, minimum_feature_count, maximum_feature_count, feature_selection_buffer,
        enable_vote_tracking, enable_consensus_monitoring, vote_history_size, validation_frequency,
        enable_incremental_updates, batch_update_size, parallel_weight_calculation, cache_weight_calculations
    )
end

"""
Consensus statistics tracking
"""
mutable struct ConsensusStats
    total_voting_rounds::Int
    successful_consensus_rounds::Int
    failed_consensus_rounds::Int
    early_stopping_triggers::Int
    tie_breaking_instances::Int
    average_consensus_strength::Float64
    average_voting_time::Float64
    consensus_stability_score::Float64
    feature_turnover_rate::Float64
    last_consensus_time::DateTime
end

"""
Initialize consensus statistics
"""
function initialize_consensus_stats()
    return ConsensusStats(
        0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, now()
    )
end

"""
Consensus building manager
Coordinates weighted voting across all trees and manages consensus detection
"""
mutable struct ConsensusManager
    # Configuration
    config::ConsensusConfig
    
    # Vote tracking
    feature_votes::Dict{Int, FeatureVote}        # Feature ID -> vote info
    tree_weights::Dict{Int, TreeWeight}          # Tree ID -> weight info
    
    # Consensus state
    current_consensus::Vector{Int}               # Currently selected features
    consensus_strength::Float64                 # Strength of current consensus
    consensus_history::Vector{Vector{Int}}      # Historical consensus selections
    consensus_timestamps::Vector{DateTime}      # Timestamps of consensus updates
    
    # Progressive consensus tracking
    stability_window::Vector{Float64}           # Recent consensus stability scores
    agreement_iterations::Int                   # Consecutive agreement iterations
    early_stopping_triggered::Bool             # Whether early stopping was triggered
    
    # Performance monitoring
    stats::ConsensusStats
    voting_times::Vector{Float64}              # Historical voting computation times
    weight_calculation_times::Vector{Float64}   # Historical weight calculation times
    
    # Synchronization and state
    consensus_lock::ReentrantLock
    weight_cache::Dict{Int, Float64}           # Cached weight calculations
    last_weight_update::DateTime               # Last time weights were updated
    
    # Status and logging
    manager_state::String
    error_log::Vector{String}
    creation_time::DateTime
end

"""
Initialize consensus manager
"""
function initialize_consensus_manager(config::ConsensusConfig = create_consensus_config())
    manager = ConsensusManager(
        config,
        Dict{Int, FeatureVote}(),
        Dict{Int, TreeWeight}(),
        Int[],
        0.0,
        Vector{Vector{Int}}(),
        DateTime[],
        Float64[],
        0,
        false,
        initialize_consensus_stats(),
        Float64[],
        Float64[],
        ReentrantLock(),
        Dict{Int, Float64}(),
        now(),
        "active",
        String[],
        now()
    )
    
    @info "Consensus manager initialized with strategy: $(config.voting_strategy)"
    return manager
end

"""
Cast vote for feature from specific tree
"""
function cast_vote!(manager::ConsensusManager, 
                   tree_id::Int, 
                   feature_id::Int, 
                   tree_performance::Float64 = 1.0,
                   exploration_depth::Int = 1)
    lock(manager.consensus_lock) do
        # Get or create feature vote tracker
        if !haskey(manager.feature_votes, feature_id)
            manager.feature_votes[feature_id] = create_feature_vote(feature_id)
        end
        
        # Get or create tree weight tracker
        if !haskey(manager.tree_weights, tree_id)
            manager.tree_weights[tree_id] = create_tree_weight(tree_id)
            # Update tree weight based on performance and exploration
            update_tree_weight!(manager, tree_id, tree_performance, exploration_depth)
        end
        
        vote = manager.feature_votes[feature_id]
        tree_weight = manager.tree_weights[tree_id]
        
        # Check if this tree has already voted for this feature
        if tree_id in vote.contributing_trees
            @debug "Tree $tree_id has already voted for feature $feature_id"
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
        
        # Update vote stability (simple moving average)
        current_strength = vote.total_votes / length(manager.tree_weights)
        if length(manager.consensus_history) > 1
            previous_selections = length(manager.consensus_history) > 0 ? manager.consensus_history[end] : Int[]
            was_selected = feature_id in previous_selections
            is_selected = current_strength >= manager.config.minimum_consensus_percentage
            
            if was_selected == is_selected
                vote.vote_stability = min(1.0, vote.vote_stability + 0.1)
            else
                vote.vote_stability = max(0.0, vote.vote_stability - 0.2)
            end
        end
        
        @debug "Vote cast: Tree $tree_id -> Feature $feature_id (weight: $weight, total: $(vote.total_votes))"
        return true
    end
end

"""
Calculate voting weight for tree based on performance and configuration
"""
function calculate_voting_weight(manager::ConsensusManager, 
                                tree_id::Int, 
                                tree_performance::Float64, 
                                exploration_depth::Int)::Float64
    
    # Check cache first if enabled
    if manager.config.cache_weight_calculations
        cache_key = hash((tree_id, tree_performance, exploration_depth))
        cached_weight = get(manager.weight_cache, cache_key, nothing)
        if !isnothing(cached_weight)
            return cached_weight
        end
    end
    
    start_time = time()
    
    weight = 1.0  # Base weight
    
    if manager.config.enable_weighted_voting
        tree_weight = get(manager.tree_weights, tree_id, nothing)
        
        if !isnothing(tree_weight)
            # Use pre-calculated weights
            weight = tree_weight.final_weight
        else
            # Calculate weight on-the-fly based on strategy
            if manager.config.voting_strategy == WEIGHTED_PERFORMANCE
                weight = calculate_performance_weight(tree_performance)
            elseif manager.config.voting_strategy == WEIGHTED_EXPLORATION
                weight = calculate_exploration_weight(exploration_depth)
            elseif manager.config.voting_strategy == ADAPTIVE_WEIGHTS
                weight = calculate_adaptive_weight(manager, tree_id, tree_performance, exploration_depth)
            end
        end
    end
    
    # Apply weight decay if configured
    if manager.config.weight_decay_rate < 1.0f0
        iterations_since_start = length(manager.consensus_history)
        decay_factor = manager.config.weight_decay_rate ^ iterations_since_start
        weight *= decay_factor
    end
    
    # Cache the result if enabled
    if manager.config.cache_weight_calculations
        cache_key = hash((tree_id, tree_performance, exploration_depth))
        manager.weight_cache[cache_key] = weight
    end
    
    # Track computation time
    computation_time = (time() - start_time) * 1000
    push!(manager.weight_calculation_times, computation_time)
    if length(manager.weight_calculation_times) > 1000
        deleteat!(manager.weight_calculation_times, 1)
    end
    
    return weight
end

"""
Calculate performance-based weight
"""
function calculate_performance_weight(performance::Float64)::Float64
    # Sigmoid transformation to emphasize high-performing trees
    return 1.0 / (1.0 + exp(-5.0 * (performance - 0.5)))
end

"""
Calculate exploration-based weight
"""
function calculate_exploration_weight(depth::Int)::Float64
    # Logarithmic weight favoring deeper exploration
    return min(2.0, 1.0 + log(max(1, depth)) / 10.0)
end

"""
Calculate adaptive weight combining multiple factors
"""
function calculate_adaptive_weight(manager::ConsensusManager, 
                                 tree_id::Int, 
                                 performance::Float64, 
                                 exploration_depth::Int)::Float64
    perf_weight = calculate_performance_weight(performance) * manager.config.performance_weight_factor
    expl_weight = calculate_exploration_weight(exploration_depth) * manager.config.exploration_weight_factor
    
    # Add diversity and stability weights if tree weight exists
    diversity_weight = manager.config.diversity_weight_factor
    stability_weight = manager.config.stability_weight_factor
    
    tree_weight = get(manager.tree_weights, tree_id, nothing)
    if !isnothing(tree_weight)
        diversity_weight *= tree_weight.diversity_weight
        stability_weight *= tree_weight.stability_weight
    end
    
    return perf_weight + expl_weight + diversity_weight + stability_weight
end

"""
Update tree weight based on performance metrics
"""
function update_tree_weight!(manager::ConsensusManager, 
                            tree_id::Int, 
                            performance::Float64, 
                            exploration_depth::Int,
                            diversity_score::Float64 = 1.0,
                            stability_score::Float64 = 1.0)
    tree_weight = get(manager.tree_weights, tree_id, create_tree_weight(tree_id))
    
    # Update individual weight components
    tree_weight.performance_weight = calculate_performance_weight(performance)
    tree_weight.exploration_weight = calculate_exploration_weight(exploration_depth)
    tree_weight.diversity_weight = diversity_score
    tree_weight.stability_weight = stability_score
    
    # Calculate final combined weight
    tree_weight.final_weight = (
        tree_weight.performance_weight * manager.config.performance_weight_factor +
        tree_weight.exploration_weight * manager.config.exploration_weight_factor +
        tree_weight.diversity_weight * manager.config.diversity_weight_factor +
        tree_weight.stability_weight * manager.config.stability_weight_factor
    )
    
    # Update history and metadata
    push!(tree_weight.weight_history, tree_weight.final_weight)
    if length(tree_weight.weight_history) > 100  # Keep recent history
        deleteat!(tree_weight.weight_history, 1)
    end
    
    tree_weight.last_update = now()
    tree_weight.update_count += 1
    
    # Store updated weight
    manager.tree_weights[tree_id] = tree_weight
    
    @debug "Updated weight for tree $tree_id: $(tree_weight.final_weight)"
end

"""
Build consensus from current votes
"""
function build_consensus!(manager::ConsensusManager)::Vector{Int}
    start_time = time()
    
    lock(manager.consensus_lock) do
        # Calculate consensus threshold
        threshold = calculate_consensus_threshold(manager)
        
        # Get features that meet consensus threshold
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
        
        # Handle ties if we need to break to reach exact target
        if length(feature_scores) > target_count
            tie_threshold = feature_scores[target_count].second
            tied_features = [pair.first for pair in feature_scores 
                           if abs(pair.second - tie_threshold) <= manager.config.tie_breaking_tolerance 
                           && pair.first ∉ consensus_features]
            
            if !isempty(tied_features)
                additional_features = resolve_ties(manager, tied_features, 
                                                 max(0, manager.config.target_feature_count - length(consensus_features)))
                append!(consensus_features, additional_features)
            end
        end
        
        # Ensure minimum feature count
        if length(consensus_features) < manager.config.minimum_feature_count
            # Add next best features to reach minimum
            remaining_features = [pair.first for pair in feature_scores 
                                if pair.first ∉ consensus_features]
            needed = manager.config.minimum_feature_count - length(consensus_features)
            append!(consensus_features, remaining_features[1:min(needed, length(remaining_features))])
        end
        
        # Update consensus state
        manager.current_consensus = consensus_features
        manager.consensus_strength = length(consensus_features) > 0 ? 
            mean([manager.feature_votes[fid].total_votes / total_trees for fid in consensus_features]) : 0.0
        
        # Record consensus history
        push!(manager.consensus_history, copy(consensus_features))
        push!(manager.consensus_timestamps, now())
        
        # Limit history size
        if length(manager.consensus_history) > manager.config.vote_history_size
            deleteat!(manager.consensus_history, 1)
            deleteat!(manager.consensus_timestamps, 1)
        end
        
        # Update progressive consensus tracking
        update_consensus_stability!(manager)
        
        # Update statistics
        manager.stats.total_voting_rounds += 1
        if length(consensus_features) >= manager.config.minimum_feature_count
            manager.stats.successful_consensus_rounds += 1
        else
            manager.stats.failed_consensus_rounds += 1
        end
        
        # Track timing
        voting_time = (time() - start_time) * 1000
        push!(manager.voting_times, voting_time)
        if length(manager.voting_times) > 1000
            deleteat!(manager.voting_times, 1)
        end
        manager.stats.average_voting_time = mean(manager.voting_times)
        manager.stats.last_consensus_time = now()
        
        @info "Consensus built: $(length(consensus_features)) features, strength: $(round(manager.consensus_strength, digits=3))"
        return consensus_features
    end
end

"""
Calculate dynamic consensus threshold
"""
function calculate_consensus_threshold(manager::ConsensusManager)::Float64
    base_threshold = manager.config.minimum_consensus_percentage
    
    if manager.config.consensus_threshold_type == ABSOLUTE_THRESHOLD
        return Float64(base_threshold)
        
    elseif manager.config.consensus_threshold_type == RELATIVE_THRESHOLD
        # Adjust based on vote distribution
        if !isempty(manager.feature_votes)
            total_trees = length(manager.tree_weights)
            max_votes = maximum(vote.total_votes for vote in values(manager.feature_votes))
            relative_factor = total_trees > 0 ? (max_votes / total_trees) * manager.config.dynamic_threshold_factor : 1.0
            return min(manager.config.maximum_consensus_percentage, base_threshold * relative_factor)
        end
        return Float64(base_threshold)
        
    elseif manager.config.consensus_threshold_type == DYNAMIC_THRESHOLD
        # Adapt based on consensus history stability
        if length(manager.stability_window) > 10
            stability = mean(manager.stability_window)
            # Higher stability allows lower threshold, encouraging convergence
            dynamic_factor = 1.0 - (stability * 0.3)
            return max(base_threshold * 0.5, base_threshold * dynamic_factor)
        end
        return Float64(base_threshold)
        
    elseif manager.config.consensus_threshold_type == PROGRESSIVE_THRESHOLD
        # Increase threshold over time to ensure convergence
        rounds = manager.stats.total_voting_rounds
        progress_factor = min(1.5, 1.0 + (rounds / 1000.0) * 0.5)
        return min(manager.config.maximum_consensus_percentage, base_threshold * progress_factor)
    end
    
    return Float64(base_threshold)
end

"""
Resolve ties using configured strategy
"""
function resolve_ties(manager::ConsensusManager, tied_features::Vector{Int}, count_needed::Int)::Vector{Int}
    if count_needed <= 0 || isempty(tied_features)
        return Int[]
    end
    
    selected = Int[]
    
    if manager.config.tie_breaking_strategy == RANDOM_SELECTION
        # Random selection
        shuffled = shuffle(tied_features)
        selected = shuffled[1:min(count_needed, length(shuffled))]
        
    elseif manager.config.tie_breaking_strategy == PERFORMANCE_BASED
        # Select based on contributing tree performance
        feature_performance = Pair{Int, Float64}[]
        for feature_id in tied_features
            vote = manager.feature_votes[feature_id]
            avg_performance = vote.performance_contribution / max(1, vote.vote_count)
            push!(feature_performance, feature_id => avg_performance)
        end
        sort!(feature_performance, by=x->x.second, rev=true)
        selected = [pair.first for pair in feature_performance[1:min(count_needed, length(feature_performance))]]
        
    elseif manager.config.tie_breaking_strategy == EXPLORATION_BASED
        # Select based on exploration metrics (use vote stability as proxy)
        feature_exploration = Pair{Int, Float64}[]
        for feature_id in tied_features
            vote = manager.feature_votes[feature_id]
            push!(feature_exploration, feature_id => vote.vote_stability)
        end
        sort!(feature_exploration, by=x->x.second, rev=true)
        selected = [pair.first for pair in feature_exploration[1:min(count_needed, length(feature_exploration))]]
        
    elseif manager.config.tie_breaking_strategy == TIMESTAMP_BASED
        # Select based on earliest vote time
        feature_timestamps = Pair{Int, DateTime}[]
        for feature_id in tied_features
            vote = manager.feature_votes[feature_id]
            push!(feature_timestamps, feature_id => vote.first_vote_time)
        end
        sort!(feature_timestamps, by=x->x.second)
        selected = [pair.first for pair in feature_timestamps[1:min(count_needed, length(feature_timestamps))]]
    end
    
    if !isempty(selected)
        manager.stats.tie_breaking_instances += 1
        @debug "Tie broken using $(manager.config.tie_breaking_strategy): $(length(selected)) features selected"
    end
    
    return selected
end

"""
Update consensus stability tracking
"""
function update_consensus_stability!(manager::ConsensusManager)
    if length(manager.consensus_history) < 2
        return
    end
    
    # Calculate stability as feature overlap between recent consensus rounds
    current = Set(manager.consensus_history[end])
    previous = Set(manager.consensus_history[end-1])
    
    overlap = length(intersect(current, previous))
    total_features = length(union(current, previous))
    stability = total_features > 0 ? overlap / total_features : 0.0
    
    # Update stability window
    push!(manager.stability_window, stability)
    if length(manager.stability_window) > manager.config.consensus_stability_window
        deleteat!(manager.stability_window, 1)
    end
    
    # Update agreement iterations
    if stability >= manager.config.early_stopping_threshold
        manager.agreement_iterations += 1
    else
        manager.agreement_iterations = 0
    end
    
    # Check for early stopping
    if manager.config.enable_progressive_consensus && 
       manager.agreement_iterations >= manager.config.minimum_agreement_iterations
        manager.early_stopping_triggered = true
        manager.stats.early_stopping_triggers += 1
        @info "Early stopping triggered after $(manager.agreement_iterations) stable iterations"
    end
    
    # Update overall stability score
    if length(manager.stability_window) > 0
        manager.stats.consensus_stability_score = mean(manager.stability_window)
    end
end

"""
Check if early stopping should be triggered
"""
function should_stop_early(manager::ConsensusManager)::Bool
    return manager.config.enable_progressive_consensus && manager.early_stopping_triggered
end

"""
Get current consensus status
"""
function get_consensus_status(manager::ConsensusManager)
    lock(manager.consensus_lock) do
        return Dict{String, Any}(
            "manager_state" => manager.manager_state,
            "current_consensus_size" => length(manager.current_consensus),
            "consensus_strength" => manager.consensus_strength,
            "total_features_voted" => length(manager.feature_votes),
            "total_trees_participating" => length(manager.tree_weights),
            "voting_rounds" => manager.stats.total_voting_rounds,
            "successful_rounds" => manager.stats.successful_consensus_rounds,
            "failed_rounds" => manager.stats.failed_consensus_rounds,
            "early_stopping_triggered" => manager.early_stopping_triggered,
            "agreement_iterations" => manager.agreement_iterations,
            "consensus_stability" => manager.stats.consensus_stability_score,
            "average_voting_time_ms" => manager.stats.average_voting_time,
            "tie_breaking_instances" => manager.stats.tie_breaking_instances,
            "last_consensus_time" => manager.stats.last_consensus_time
        )
    end
end

"""
Get detailed feature voting information
"""
function get_feature_voting_details(manager::ConsensusManager, feature_id::Int)
    vote = get(manager.feature_votes, feature_id, nothing)
    if isnothing(vote)
        return nothing
    end
    
    return Dict{String, Any}(
        "feature_id" => vote.feature_id,
        "total_votes" => vote.total_votes,
        "vote_count" => vote.vote_count,
        "contributing_trees" => collect(vote.contributing_trees),
        "average_weight" => vote.average_weight,
        "vote_stability" => vote.vote_stability,
        "performance_contribution" => vote.performance_contribution,
        "first_vote_time" => vote.first_vote_time,
        "last_vote_time" => vote.last_vote_time
    )
end

"""
Clear all votes and reset consensus state
"""
function reset_consensus!(manager::ConsensusManager)
    lock(manager.consensus_lock) do
        empty!(manager.feature_votes)
        empty!(manager.tree_weights)
        empty!(manager.current_consensus)
        manager.consensus_strength = 0.0
        empty!(manager.consensus_history)
        empty!(manager.consensus_timestamps)
        empty!(manager.stability_window)
        manager.agreement_iterations = 0
        manager.early_stopping_triggered = false
        empty!(manager.weight_cache)
        
        @info "Consensus manager reset"
    end
end

"""
Generate consensus building report
"""
function generate_consensus_report(manager::ConsensusManager)
    status = get_consensus_status(manager)
    
    report = String[]
    
    push!(report, "=== Consensus Building Report ===")
    push!(report, "Generated: $(now())")
    push!(report, "Manager State: $(status["manager_state"])")
    push!(report, "")
    
    # Current consensus
    push!(report, "Current Consensus:")
    push!(report, "  Selected Features: $(status["current_consensus_size"])")
    push!(report, "  Consensus Strength: $(round(status["consensus_strength"], digits=3))")
    push!(report, "  Total Features Voted: $(status["total_features_voted"])")
    push!(report, "  Participating Trees: $(status["total_trees_participating"])")
    push!(report, "")
    
    # Voting performance
    push!(report, "Voting Performance:")
    push!(report, "  Total Voting Rounds: $(status["voting_rounds"])")
    push!(report, "  Successful Rounds: $(status["successful_rounds"])")
    push!(report, "  Failed Rounds: $(status["failed_rounds"])")
    if status["voting_rounds"] > 0
        success_rate = status["successful_rounds"] / status["voting_rounds"] * 100
        push!(report, "  Success Rate: $(round(success_rate, digits=1))%")
    end
    push!(report, "  Average Voting Time: $(round(status["average_voting_time_ms"], digits=2))ms")
    push!(report, "")
    
    # Progressive consensus
    push!(report, "Progressive Consensus:")
    push!(report, "  Early Stopping: $(status["early_stopping_triggered"] ? "Yes" : "No")")
    push!(report, "  Agreement Iterations: $(status["agreement_iterations"])")
    push!(report, "  Consensus Stability: $(round(status["consensus_stability"], digits=3))")
    push!(report, "  Tie Breaking Instances: $(status["tie_breaking_instances"])")
    push!(report, "")
    
    # Configuration summary
    push!(report, "Configuration:")
    push!(report, "  Voting Strategy: $(manager.config.voting_strategy)")
    push!(report, "  Consensus Threshold: $(manager.config.consensus_threshold_type)")
    push!(report, "  Tie Breaking: $(manager.config.tie_breaking_strategy)")
    push!(report, "  Target Features: $(manager.config.target_feature_count)")
    push!(report, "  Progressive Consensus: $(manager.config.enable_progressive_consensus)")
    push!(report, "")
    
    # Top features if available
    if !isempty(manager.current_consensus)
        push!(report, "Current Top Features:")
        for (i, feature_id) in enumerate(manager.current_consensus[1:min(10, end)])
            vote = manager.feature_votes[feature_id]
            push!(report, "  $(i). Feature $feature_id: $(round(vote.total_votes, digits=2)) votes ($(vote.vote_count) trees)")
        end
    end
    
    push!(report, "=== End Consensus Report ===")
    
    return join(report, "\n")
end

"""
Cleanup consensus manager
"""
function cleanup_consensus_manager!(manager::ConsensusManager)
    lock(manager.consensus_lock) do
        reset_consensus!(manager)
        empty!(manager.voting_times)
        empty!(manager.weight_calculation_times)
        empty!(manager.error_log)
    end
    
    manager.manager_state = "shutdown"
    @info "Consensus manager cleaned up"
end

# Export main types and functions
export VotingStrategy, ConsensusThreshold, TieBreakingStrategy
export SIMPLE_MAJORITY, WEIGHTED_PERFORMANCE, WEIGHTED_EXPLORATION, ADAPTIVE_WEIGHTS, HIERARCHICAL_CONSENSUS
export ABSOLUTE_THRESHOLD, RELATIVE_THRESHOLD, DYNAMIC_THRESHOLD, PROGRESSIVE_THRESHOLD
export RANDOM_SELECTION, PERFORMANCE_BASED, EXPLORATION_BASED, DIVERSITY_BASED, TIMESTAMP_BASED
export FeatureVote, TreeWeight, ConsensusConfig, ConsensusManager
export create_consensus_config, initialize_consensus_manager
export cast_vote!, build_consensus!, update_tree_weight!
export should_stop_early, get_consensus_status, get_feature_voting_details
export reset_consensus!, generate_consensus_report, cleanup_consensus_manager!

end # module ConsensusVoting