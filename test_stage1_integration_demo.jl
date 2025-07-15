"""
Stage 1 Integration System Demonstration
Shows how to use the Stage 1 output integration and feature indexing system
"""

using Random
using Statistics
using Dates
using Printf

Random.seed!(42)

# Include the Stage 1 integration module
include("src/stage2/stage1_integration.jl")
using .Stage1Integration

println("🚀 Stage 1 Integration System Demonstration")
println("=" ^ 60)

"""
Create a realistic mock Stage 1 output for demonstration
"""
function create_demo_stage1_output()
    println("📊 Creating mock Stage 1 output (500 features)...")
    
    features = FeatureMetadata[]
    
    # Create 500 diverse features
    for i in 1:500
        # Generate realistic feature properties
        original_id = 10000 + rand(1:90000)  # Random original IDs
        importance = rand() * 0.7 + 0.3      # Importance 0.3-1.0
        confidence = rand() * 0.4 + 0.6      # Confidence 0.6-1.0
        
        # Distribute feature types realistically
        feature_type = if i <= 300
            "numeric"
        elseif i <= 450
            "categorical" 
        else
            "binary"
        end
        
        # Group some features for correlation analysis
        correlation_group = if i <= 50
            1  # High correlation group
        elseif i <= 100
            2  # Medium correlation group
        elseif i <= 120
            3  # Small correlation group
        else
            nothing  # Uncorrelated
        end
        
        # Vary source stages
        source_stage = if i <= 200
            "xgboost"
        elseif i <= 400
            "random_forest"
        else
            "correlation_analysis"
        end
        
        feature = FeatureMetadata(
            original_id,
            i,
            "feature_$(lpad(i, 3, '0'))",
            feature_type,
            importance,
            confidence,
            correlation_group,
            Dict("scaled" => true, "outliers_removed" => rand(Bool)),
            source_stage,
            now() - Minute(rand(1:500))
        )
        
        push!(features, feature)
    end
    
    # Sort by importance for realistic ranking
    sort!(features, by = f -> f.importance_score, rev = true)
    
    # Update stage1_rank to match sorted order
    for (i, feature) in enumerate(features)
        features[i] = FeatureMetadata(
            feature.original_id, i, feature.name, feature.feature_type,
            feature.importance_score, feature.selection_confidence,
            feature.correlation_group, feature.preprocessing_info,
            feature.source_stage, feature.timestamp_selected
        )
    end
    
    return Stage1Output(
        features,
        Dict(
            "total_original_features" => 50000,
            "selection_method" => "ensemble_voting",
            "selection_threshold" => 0.75,
            "feature_stability_score" => 0.89
        ),
        Dict(
            "dataset_name" => "financial_features_v2",
            "n_samples" => 250000,
            "n_original_features" => 50000,
            "data_time_range" => "2020-2023"
        ),
        Dict(
            "selection_time_seconds" => 1847.3,
            "cross_validation_score" => 0.923,
            "feature_importance_variance" => 0.15,
            "selection_stability" => 0.91
        ),
        nothing,  # No feature matrix in this demo
        Dict(
            "holdout_validation_score" => 0.918,
            "feature_redundancy_score" => 0.12,
            "correlation_analysis_complete" => true
        ),
        now() - Hour(2),
        "1.2.0"
    )
end

"""
Demonstrate Stage 1 integration workflow
"""
function demonstrate_integration_workflow()
    println("\n🔧 Stage 1 Integration Workflow Demonstration")
    println("-" ^ 50)
    
    # Step 1: Initialize interface
    println("1️⃣  Initializing Stage 1 integration interface...")
    config = create_stage1_integration_config(
        validation_enabled = true,
        caching_enabled = true,
        performance_monitoring = true
    )
    interface = initialize_stage1_interface(config)
    
    @assert interface.initialization_status == "ready"
    println("   ✅ Interface initialized successfully")
    
    # Step 2: Create and load Stage 1 output
    println("\n2️⃣  Loading Stage 1 output...")
    stage1_output = create_demo_stage1_output()
    interface.stage1_output = stage1_output
    
    println("   📊 Loaded $(length(stage1_output.features)) features")
    println("   📈 Top importance score: $(round(maximum(f.importance_score for f in stage1_output.features), digits=3))")
    println("   📉 Min importance score: $(round(minimum(f.importance_score for f in stage1_output.features), digits=3))")
    
    # Step 3: Build feature indexer
    println("\n3️⃣  Building feature indexer...")
    indexer = build_feature_indexer(interface)
    
    println("   🗂️  Built mappings for $(indexer.total_features) features")
    println("   📋 Feature types: $(length(indexer.type_groups)) types")
    println("   🔗 Correlation groups: $(length(indexer.correlation_groups)) groups")
    
    # Step 4: Initialize state tracker
    println("\n4️⃣  Initializing feature state tracker...")
    tracker = initialize_state_tracker(interface)
    
    selected_count = sum(tracker.feature_states)
    println("   📊 Initialized state for $(length(tracker.feature_states)) features")
    println("   ✅ Currently selected: $selected_count features")
    
    return interface
end

"""
Demonstrate feature lookup operations
"""
function demonstrate_feature_operations(interface)
    println("\n🔍 Feature Lookup Operations Demo")
    println("-" ^ 40)
    
    # Get top features by importance
    println("🏆 Top 10 most important features:")
    top_features = get_top_features(interface, 10)
    for (i, (idx, metadata)) in enumerate(top_features[1:5])  # Show first 5
        @printf("   %2d. %s (importance: %.3f, type: %s)\n", 
                i, metadata.name, metadata.importance_score, metadata.feature_type)
    end
    println("   ... (showing 5 of 10)")
    
    # Demonstrate lookup by original ID
    println("\n🔍 Feature lookup examples:")
    sample_feature = interface.stage1_output.features[10]
    
    println("   Original ID $(sample_feature.original_id) → Stage 2 index: $(get_stage2_index(interface, sample_feature.original_id))")
    println("   Stage 2 index 10 → Original ID: $(get_original_id(interface, 10))")
    println("   Feature name '$(sample_feature.name)' → Stage 2 index: $(get_stage2_index_by_name(interface, sample_feature.name))")
    
    # Show feature type distribution
    println("\n📊 Feature type distribution:")
    for feature_type in ["numeric", "categorical", "binary"]
        type_features = get_features_by_type(interface, feature_type)
        println("   $feature_type: $(length(type_features)) features")
    end
    
    # Show correlation groups
    println("\n🔗 Correlation group analysis:")
    for group_id in 1:3
        group_features = get_correlation_group(interface, group_id)
        if !isempty(group_features)
            avg_importance = mean(f[2].importance_score for f in group_features)
            println("   Group $group_id: $(length(group_features)) features (avg importance: $(round(avg_importance, digits=3)))")
        end
    end
end

"""
Demonstrate MCTS-style state management
"""
function demonstrate_mcts_operations(interface)
    println("\n🌳 MCTS-Style State Management Demo")
    println("-" ^ 45)
    
    # Simulate MCTS tree operations
    println("🎯 Simulating MCTS feature selection operations...")
    
    # Tree 1: Select top features
    println("\n🌲 Tree 1 - Selecting top importance features:")
    top_features = get_top_features(interface, 20)
    
    for (i, (idx, metadata)) in enumerate(top_features[1:10])
        update_feature_state!(interface, idx, true, tree_id=1)
        if i <= 3
            println("   ✅ Selected feature $(idx): $(metadata.name) (importance: $(round(metadata.importance_score, digits=3)))")
        end
    end
    
    tree1_selected = sum(get_feature_states(interface, tree_id=1))
    println("   📊 Tree 1 total selected: $tree1_selected features")
    
    # Tree 2: Select by correlation group
    println("\n🌲 Tree 2 - Selecting correlation group 1:")
    group1_features = get_correlation_group(interface, 1)
    
    for (i, (idx, metadata)) in enumerate(group1_features[1:15])
        update_feature_state!(interface, idx, true, tree_id=2)
        if i <= 3
            println("   ✅ Selected feature $(idx): $(metadata.name) (group: $(metadata.correlation_group))")
        end
    end
    
    tree2_selected = sum(get_feature_states(interface, tree_id=2))
    println("   📊 Tree 2 total selected: $tree2_selected features")
    
    # Tree 3: Mixed selection strategy
    println("\n🌲 Tree 3 - Mixed selection strategy:")
    numeric_features = get_features_by_type(interface, "numeric")
    categorical_features = get_features_by_type(interface, "categorical")
    
    # Select some numeric and categorical features
    for (idx, metadata) in [numeric_features[1:5]; categorical_features[1:5]]
        update_feature_state!(interface, idx, true, tree_id=3)
    end
    
    tree3_selected = sum(get_feature_states(interface, tree_id=3))
    println("   📊 Tree 3 total selected: $tree3_selected features")
    
    # Show state tracking metrics
    tracker = interface.state_tracker
    println("\n📈 State tracking summary:")
    println("   🔄 Total state changes: $(tracker.state_change_count)")
    println("   🌳 Active trees: $(length(tracker.tree_feature_maps))")
    println("   💾 Cache hits: $(tracker.cache_hit_count)")
    println("   🔍 Cache misses: $(tracker.cache_miss_count)")
end

"""
Demonstrate performance monitoring and validation
"""
function demonstrate_monitoring_validation(interface)
    println("\n📊 Performance Monitoring & Validation Demo")
    println("-" ^ 50)
    
    # Run consistency validation
    println("🔍 Running feature consistency validation...")
    validation_result = validate_feature_consistency!(interface)
    
    if validation_result
        println("   ✅ All consistency checks passed")
    else
        println("   ❌ Validation issues detected")
    end
    
    # Show performance metrics
    println("\n⚡ Performance metrics:")
    for (metric, value) in interface.performance_metrics
        @printf("   %s: %.4f seconds\n", metric, value)
    end
    
    # Generate and show status report excerpt
    println("\n📋 System status summary:")
    report = generate_status_report(interface)
    
    # Extract key lines from report
    report_lines = split(report, '\n')
    key_sections = [
        "Initialization Status:",
        "Component Status:",
        "Total Features:",
        "Feature Types:",
        "State Changes:",
        "Cache Hits:"
    ]
    
    for line in report_lines
        for section in key_sections
            if contains(line, section)
                println("   $(strip(line))")
                break
            end
        end
    end
end

"""
Main demonstration function
"""
function main()
    try
        # Run demonstration workflow
        interface = demonstrate_integration_workflow()
        
        # Demonstrate operations
        demonstrate_feature_operations(interface)
        demonstrate_mcts_operations(interface)
        demonstrate_monitoring_validation(interface)
        
        println("\n" * "=" ^ 60)
        println("🎉 Stage 1 Integration Demo Completed Successfully!")
        println("✨ Key achievements:")
        println("   ✅ Loaded and indexed 500 features from Stage 1")
        println("   ✅ Built efficient lookup system for MCTS operations")
        println("   ✅ Demonstrated feature state tracking across multiple trees")
        println("   ✅ Validated system consistency and performance")
        println("   ✅ Ready for Stage 2 GPU-MCTS integration")
        
        return interface
        
    catch e
        println("\n❌ Demo failed with error: $e")
        rethrow(e)
    end
end

# Run the demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end