"""
Test Suite for Stage 1 Output Integration and Feature Indexing System
Validates interface layer, feature indexing, state tracking, and MCTS integration
"""

using Test
using Random
using Statistics
using Dates
using JSON3
using JLD2

# Include the Stage 1 integration module
include("../../src/stage2/stage1_integration.jl")
using .Stage1Integration

@testset "Stage 1 Integration Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration and Initialization Tests" begin
        # Test default configuration
        config = create_stage1_integration_config()
        
        @test config["validation_enabled"] == true
        @test config["caching_enabled"] == true
        @test config["expected_feature_count"] == 500
        @test config["cache_size_limit"] == 10000
        
        # Test custom configuration
        custom_config = create_stage1_integration_config(
            validation_enabled = false,
            cache_size_limit = 5000
        )
        
        @test custom_config["validation_enabled"] == false
        @test custom_config["cache_size_limit"] == 5000
        
        # Test interface initialization
        interface = initialize_stage1_interface(config)
        
        @test interface.initialization_status == "ready"
        @test interface.validation_enabled == true
        @test interface.caching_enabled == true
        @test interface.operation_count == 0
        @test isempty(interface.error_log)
        
        println("  ✅ Configuration and initialization tests passed")
    end
    
    @testset "Feature Metadata Creation Tests" begin
        # Create sample feature metadata
        feature = FeatureMetadata(
            12345,                          # original_id
            1,                              # stage1_rank
            "feature_001",                  # name
            "numeric",                      # feature_type
            0.85,                          # importance_score
            0.92,                          # selection_confidence
            101,                           # correlation_group
            Dict("normalized" => true),     # preprocessing_info
            "xgboost",                     # source_stage
            now()                          # timestamp_selected
        )
        
        @test feature.original_id == 12345
        @test feature.stage1_rank == 1
        @test feature.name == "feature_001"
        @test feature.feature_type == "numeric"
        @test feature.importance_score == 0.85
        @test feature.selection_confidence == 0.92
        @test feature.correlation_group == 101
        @test feature.source_stage == "xgboost"
        
        println("  ✅ Feature metadata creation tests passed")
    end
    
    @testset "Mock Stage 1 Output Generation Tests" begin
        # Generate mock Stage 1 output for testing
        function create_mock_stage1_output(n_features::Int = 500)
            features = FeatureMetadata[]
            
            for i in 1:n_features
                feature = FeatureMetadata(
                    10000 + i,                    # original_id
                    i,                            # stage1_rank
                    "feature_$(lpad(i, 3, '0'))", # name
                    rand(["numeric", "categorical", "binary"]), # feature_type
                    rand() * 0.5 + 0.5,          # importance_score (0.5-1.0)
                    rand() * 0.3 + 0.7,          # selection_confidence (0.7-1.0)
                    rand(1:50),                   # correlation_group
                    Dict("normalized" => rand(Bool)), # preprocessing_info
                    rand(["xgboost", "random_forest", "correlation"]), # source_stage
                    now() - Minute(rand(1:1000))  # timestamp_selected
                )
                push!(features, feature)
            end
            
            return Stage1Output(
                features,
                Dict("total_original_features" => 50000, "selection_method" => "ensemble"),
                Dict("dataset_name" => "test_dataset", "n_samples" => 100000),
                Dict("selection_time" => 300.5, "validation_score" => 0.92),
                nothing,  # No feature matrix for testing
                Dict("cross_validation_score" => 0.89, "stability_score" => 0.94),
                now(),
                "1.0.0"
            )
        end
        
        mock_output = create_mock_stage1_output(500)
        
        @test length(mock_output.features) == 500
        @test all(f.stage1_rank == i for (i, f) in enumerate(mock_output.features))
        @test all(0.5 <= f.importance_score <= 1.0 for f in mock_output.features)
        @test all(0.7 <= f.selection_confidence <= 1.0 for f in mock_output.features)
        @test mock_output.version == "1.0.0"
        
        println("  ✅ Mock Stage 1 output generation tests passed")
    end
    
    @testset "Stage 1 Output Validation Tests" begin
        interface = initialize_stage1_interface()
        
        # Test valid output
        valid_output = Stage1Output(
            [FeatureMetadata(i, i, "feat_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500],
            Dict("method" => "test"),
            Dict("name" => "test"),
            Dict("time" => 100.0),
            nothing,
            Dict("score" => 0.9),
            now(),
            "1.0.0"
        )
        
        @test Stage1Integration.validate_stage1_output(valid_output, interface.config) == true
        
        # Test invalid output - wrong feature count
        invalid_output_count = Stage1Output(
            [FeatureMetadata(i, i, "feat_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:400],
            Dict("method" => "test"),
            Dict("name" => "test"),
            Dict("time" => 100.0),
            nothing,
            Dict("score" => 0.9),
            now(),
            "1.0.0"
        )
        
        @test_throws ErrorException Stage1Integration.validate_stage1_output(invalid_output_count, interface.config)
        
        # Test invalid output - wrong importance score
        invalid_features = [FeatureMetadata(i, i, "feat_$i", "numeric", i == 1 ? 1.5 : 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        invalid_output_importance = Stage1Output(
            invalid_features,
            Dict("method" => "test"),
            Dict("name" => "test"),
            Dict("time" => 100.0),
            nothing,
            Dict("score" => 0.9),
            now(),
            "1.0.0"
        )
        
        @test_throws ErrorException Stage1Integration.validate_stage1_output(invalid_output_importance, interface.config)
        
        println("  ✅ Stage 1 output validation tests passed")
    end
    
    @testset "Feature Indexer Building Tests" begin
        interface = initialize_stage1_interface()
        
        # Create mock Stage 1 output
        mock_features = [
            FeatureMetadata(100+i, i, "feature_$i", i <= 250 ? "numeric" : "categorical", 
                          1.0 - (i-1)/1000, 0.9, i <= 10 ? 1 : nothing, Dict{String,Any}(), "test", now()) 
            for i in 1:500
        ]
        
        mock_output = Stage1Output(
            mock_features,
            Dict("method" => "test"), Dict("name" => "test"), Dict("time" => 100.0),
            nothing, Dict("score" => 0.9), now(), "1.0.0"
        )
        
        interface.stage1_output = mock_output
        
        # Build indexer
        indexer = build_feature_indexer(interface)
        
        @test indexer.total_features == 500
        @test length(indexer.original_to_stage2) == 500
        @test length(indexer.stage2_to_original) == 500
        @test length(indexer.name_to_stage2) == 500
        @test length(indexer.stage2_to_name) == 500
        
        # Test mappings consistency
        for i in 1:500
            original_id = 100 + i
            @test indexer.original_to_stage2[original_id] == i
            @test indexer.stage2_to_original[i] == original_id
            @test indexer.name_to_stage2["feature_$i"] == i
            @test indexer.stage2_to_name[i] == "feature_$i"
        end
        
        # Test importance ranking (should be sorted by importance descending)
        importance_scores = [mock_features[i].importance_score for i in indexer.importance_ranking]
        @test issorted(importance_scores, rev=true)
        
        # Test correlation groups
        @test haskey(indexer.correlation_groups, 1)
        @test length(indexer.correlation_groups[1]) == 10  # First 10 features
        
        # Test type groups
        @test haskey(indexer.type_groups, "numeric")
        @test haskey(indexer.type_groups, "categorical")
        @test length(indexer.type_groups["numeric"]) == 250
        @test length(indexer.type_groups["categorical"]) == 250
        
        println("  ✅ Feature indexer building tests passed")
    end
    
    @testset "State Tracker Initialization Tests" begin
        interface = initialize_stage1_interface()
        
        # Set up mock data
        mock_features = [FeatureMetadata(100+i, i, "feature_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        
        # Initialize state tracker
        tracker = initialize_state_tracker(interface)
        
        @test length(tracker.feature_states) == 500
        @test all(.!tracker.feature_states)  # All should be unselected initially
        @test length(tracker.selection_history) == 1  # Initial state
        @test length(tracker.state_timestamps) == 1
        @test tracker.state_change_count == 0
        @test tracker.cache_hit_count == 0
        @test tracker.cache_miss_count == 0
        @test isempty(tracker.tree_feature_maps)
        @test isempty(tracker.validation_errors)
        
        println("  ✅ State tracker initialization tests passed")
    end
    
    @testset "Feature Lookup Operations Tests" begin
        interface = initialize_stage1_interface()
        
        # Set up test data
        mock_features = [FeatureMetadata(2000+i, i, "test_feature_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        initialize_state_tracker(interface)
        
        # Test get_stage2_index
        @test get_stage2_index(interface, 2001) == 1
        @test get_stage2_index(interface, 2250) == 250
        @test get_stage2_index(interface, 2500) == 500
        
        # Test get_original_id
        @test get_original_id(interface, 1) == 2001
        @test get_original_id(interface, 250) == 2250
        @test get_original_id(interface, 500) == 2500
        
        # Test get_stage2_index_by_name
        @test get_stage2_index_by_name(interface, "test_feature_1") == 1
        @test get_stage2_index_by_name(interface, "test_feature_100") == 100
        
        # Test get_feature_metadata
        metadata = get_feature_metadata(interface, 1)
        @test metadata.name == "test_feature_1"
        @test metadata.original_id == 2001
        @test metadata.stage1_rank == 1
        
        # Test error cases
        @test_throws ErrorException get_stage2_index(interface, 9999)  # Non-existent original ID
        @test_throws ErrorException get_original_id(interface, 501)    # Out of range
        @test_throws ErrorException get_stage2_index_by_name(interface, "nonexistent") # Non-existent name
        
        println("  ✅ Feature lookup operations tests passed")
    end
    
    @testset "Feature State Management Tests" begin
        interface = initialize_stage1_interface()
        
        # Set up test data
        mock_features = [FeatureMetadata(3000+i, i, "state_test_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        initialize_state_tracker(interface)
        
        # Test initial state
        states = get_feature_states(interface)
        @test length(states) == 500
        @test all(.!states)  # All unselected
        
        # Test state updates
        update_feature_state!(interface, 1, true)
        update_feature_state!(interface, 10, true)
        update_feature_state!(interface, 100, true)
        
        states = get_feature_states(interface)
        @test states[1] == true
        @test states[10] == true
        @test states[100] == true
        @test sum(states) == 3
        
        # Test state change tracking
        tracker = interface.state_tracker
        @test tracker.state_change_count == 3
        @test length(tracker.selection_history) == 4  # Initial + 3 changes
        
        # Test tree-specific states
        update_feature_state!(interface, 50, true, tree_id=1)
        update_feature_state!(interface, 51, true, tree_id=1)
        
        tree1_states = get_feature_states(interface, tree_id=1)
        @test tree1_states[50] == true
        @test tree1_states[51] == true
        @test sum(tree1_states) == 5  # Previous 3 + new 2
        
        # Test state deselection
        update_feature_state!(interface, 1, false)
        states = get_feature_states(interface)
        @test states[1] == false
        @test sum(states) == 4  # One deselected
        
        println("  ✅ Feature state management tests passed")
    end
    
    @testset "Feature Query Operations Tests" begin
        interface = initialize_stage1_interface()
        
        # Create diverse test features
        mock_features = FeatureMetadata[]
        for i in 1:500
            importance = 1.0 - (i-1)/1000  # Decreasing importance
            feature_type = i <= 200 ? "numeric" : (i <= 400 ? "categorical" : "binary")
            correlation_group = i <= 50 ? 1 : (i <= 100 ? 2 : nothing)
            
            feature = FeatureMetadata(
                4000+i, i, "query_test_$i", feature_type, importance, 0.9,
                correlation_group, Dict{String,Any}(), "test", now()
            )
            push!(mock_features, feature)
        end
        
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        
        # Test get_top_features
        top_10 = get_top_features(interface, 10)
        @test length(top_10) == 10
        @test all(top_10[i][1] == i for i in 1:10)  # Should be indices 1-10 (highest importance)
        
        # Test importance ordering
        importances = [top_10[i][2].importance_score for i in 1:10]
        @test issorted(importances, rev=true)
        
        # Test get_correlation_group
        group1_features = get_correlation_group(interface, 1)
        @test length(group1_features) == 50
        @test all(f[2].correlation_group == 1 for f in group1_features)
        
        group2_features = get_correlation_group(interface, 2)
        @test length(group2_features) == 50
        @test all(f[2].correlation_group == 2 for f in group2_features)
        
        # Test get_features_by_type
        numeric_features = get_features_by_type(interface, "numeric")
        @test length(numeric_features) == 200
        @test all(f[2].feature_type == "numeric" for f in numeric_features)
        
        categorical_features = get_features_by_type(interface, "categorical")
        @test length(categorical_features) == 200
        @test all(f[2].feature_type == "categorical" for f in categorical_features)
        
        binary_features = get_features_by_type(interface, "binary")
        @test length(binary_features) == 100
        @test all(f[2].feature_type == "binary" for f in binary_features)
        
        println("  ✅ Feature query operations tests passed")
    end
    
    @testset "Caching and Performance Tests" begin
        interface = initialize_stage1_interface(create_stage1_integration_config(
            caching_enabled = true,
            performance_monitoring = true
        ))
        
        # Set up test data
        mock_features = [FeatureMetadata(5000+i, i, "cache_test_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        initialize_state_tracker(interface)
        
        # Test cache behavior
        initial_cache_misses = interface.state_tracker.cache_miss_count
        
        # First lookup - should be cache miss
        idx1 = get_stage2_index(interface, 5001)
        @test idx1 == 1
        @test interface.state_tracker.cache_miss_count == initial_cache_misses + 1
        
        # Second lookup - should be cache hit
        initial_cache_hits = interface.state_tracker.cache_hit_count
        idx2 = get_stage2_index(interface, 5001)
        @test idx2 == 1
        @test interface.state_tracker.cache_hit_count == initial_cache_hits + 1
        
        # Test performance monitoring
        @test haskey(interface.performance_metrics, "indexer_build_time")
        @test interface.performance_metrics["indexer_build_time"] > 0
        
        println("  ✅ Caching and performance tests passed")
    end
    
    @testset "Validation and Consistency Tests" begin
        interface = initialize_stage1_interface()
        
        # Set up test data
        mock_features = [FeatureMetadata(6000+i, i, "validation_test_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        initialize_state_tracker(interface)
        
        # Test successful validation
        @test validate_feature_consistency!(interface) == true
        @test interface.state_tracker.state_consistency_checks == 1
        @test isempty(interface.state_tracker.validation_errors)
        
        # Modify some states and validate again
        update_feature_state!(interface, 1, true)
        update_feature_state!(interface, 50, true, tree_id=1)
        
        @test validate_feature_consistency!(interface) == true
        @test interface.state_tracker.state_consistency_checks == 2
        
        println("  ✅ Validation and consistency tests passed")
    end
    
    @testset "Status Reporting Tests" begin
        interface = initialize_stage1_interface()
        
        # Set up test data
        mock_features = [FeatureMetadata(7000+i, i, "report_test_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        initialize_state_tracker(interface)
        
        # Generate status report
        report = generate_status_report(interface)
        
        @test contains(report, "Stage 1 Integration Interface Status Report")
        @test contains(report, "Initialization Status: ready")
        @test contains(report, "Stage 1 Output: Loaded")
        @test contains(report, "Feature Indexer: Ready")
        @test contains(report, "State Tracker: Active")
        @test contains(report, "Total Features: 500")
        @test contains(report, "Currently Selected: 0 / 500")
        
        # Test with some state changes
        update_feature_state!(interface, 1, true)
        update_feature_state!(interface, 2, true)
        
        report2 = generate_status_report(interface)
        @test contains(report2, "Currently Selected: 2 / 500")
        @test contains(report2, "State Changes: 2")
        
        println("  ✅ Status reporting tests passed")
    end
    
    @testset "File I/O Tests" begin
        interface = initialize_stage1_interface()
        
        # Set up test data
        mock_features = [FeatureMetadata(8000+i, i, "io_test_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        initialize_state_tracker(interface)
        
        # Test state saving
        temp_file = tempname() * ".jld2"
        save_interface_state(interface, temp_file)
        @test isfile(temp_file)
        
        # Load and verify
        saved_data = JLD2.load(temp_file, "interface_state")
        @test haskey(saved_data, "initialization_status")
        @test saved_data["initialization_status"] == "ready"
        @test haskey(saved_data, "state_tracker")
        
        # Cleanup
        rm(temp_file, force=true)
        
        println("  ✅ File I/O tests passed")
    end
    
    @testset "Error Handling Tests" begin
        interface = initialize_stage1_interface()
        
        # Test operations before setup
        @test_throws ErrorException build_feature_indexer(interface)  # No Stage 1 output
        @test_throws ErrorException initialize_state_tracker(interface)  # No indexer
        
        # Set up minimal data
        mock_features = [FeatureMetadata(9000+i, i, "error_test_$i", "numeric", 0.8, 0.9, nothing, Dict{String,Any}(), "test", now()) for i in 1:500]
        interface.stage1_output = Stage1Output(mock_features, Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}(), nothing, Dict{String,Any}(), now(), "1.0.0")
        build_feature_indexer(interface)
        initialize_state_tracker(interface)
        
        # Test invalid lookups
        @test_throws ErrorException get_stage2_index(interface, 99999)
        @test_throws ErrorException get_original_id(interface, 501)
        @test_throws ErrorException get_stage2_index_by_name(interface, "nonexistent")
        @test_throws ErrorException get_feature_metadata(interface, 0)
        @test_throws ErrorException get_feature_metadata(interface, 501)
        
        # Test invalid state updates
        @test_throws ErrorException update_feature_state!(interface, 0, true)
        @test_throws ErrorException update_feature_state!(interface, 501, true)
        
        println("  ✅ Error handling tests passed")
    end
end

println("All Stage 1 Integration tests passed successfully!")
println("✅ Feature metadata structures and validation")
println("✅ Stage 1 output loading and processing")
println("✅ Feature indexing system with efficient lookups")
println("✅ Feature state tracking for MCTS operations")
println("✅ Caching and performance optimization")
println("✅ Consistency validation and error handling")
println("✅ Status reporting and file I/O operations")
println("✅ Integration readiness for Stage 2 MCTS system")