using Test
using SQLite
using DataFrames
using JSON3
using Dates
using UUIDs

# Include the SQLite result writer module
include("../../src/stage2/sqlite_result_writer.jl")

@testset "SQLite Result Writer Tests" begin
    
    # Create temporary database for testing
    temp_dir = "/tmp/test_sqlite_$(uuid4())"
    mkpath(temp_dir)
    db_path = joinpath(temp_dir, "test_results.sqlite")
    
    # Create a separate database for each test to avoid conflicts
    test_counter = 0
    
    @testset "SQLiteResultWriter Creation and Schema" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        @test writer.db_path == db_path
        @test writer.dataset_name == "test_dataset"
        @test writer.current_run_id === nothing
        @test !writer.transaction_active
        @test isempty(writer.feature_cache)
        
        # Verify tables were created
        tables = SQLite.tables(writer.db_connection)
        expected_tables = ["result_runs", "ensemble_config", "selected_features", 
                          "ensemble_metrics", "tree_statistics", "consensus_history",
                          "performance_metrics", "feature_validation"]
        
        table_names = [table.name for table in tables]
        
        for table in expected_tables
            @test table in table_names
        end
        
        close_database!(writer)
    end
    
    @testset "Run Initialization and Configuration" begin
        db_path2 = joinpath(temp_dir, "test_results2.sqlite")
        writer = SQLiteResultWriter(db_path2, "test_dataset")
        
        config = Dict{String, Any}(
            "total_features_input" => 500,
            "tree_count" => 100,
            "gpu_count" => 2,
            "exploration_constant" => 1.414,
            "convergence_threshold" => 0.95,
            "use_metamodel" => true,
            "feature_subset_size" => [10, 20, 30],
            "random_seed" => 42
        )
        
        run_id = initialize_run!(writer, config)
        
        @test writer.current_run_id == run_id
        @test length(run_id) == 36  # UUID length
        
        # Verify run was stored
        runs = get_run_history(writer)
        @test nrow(runs) == 1
        @test runs.run_id[1] == run_id
        @test runs.dataset_name[1] == "test_dataset"
        @test runs.completion_status[1] == "RUNNING"
        @test runs.total_features_input[1] == 500
        
        # Verify configuration was stored
        config_query = SQLite.execute(writer.db_connection, """
            SELECT config_key, config_value, config_type 
            FROM ensemble_config WHERE run_id = ?
        """, [run_id])
        
        config_rows = collect(config_query)
        @test length(config_rows) == 8
        
        # Check specific configuration values
        config_dict = Dict(row[1] => (row[2], row[3]) for row in config_rows)
        @test config_dict["tree_count"] == ("100", "INTEGER")
        @test config_dict["exploration_constant"] == ("1.414", "FLOAT")
        @test config_dict["use_metamodel"] == ("true", "BOOLEAN")
        @test config_dict["feature_subset_size"] == ("[10,20,30]", "ARRAY")
        
        close_database!(writer)
    end
    
    @testset "Selected Features Writing" begin
        db_path3 = joinpath(temp_dir, "test_results3.sqlite")
        writer = SQLiteResultWriter(db_path3, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test selected features
        features = [
            create_selected_feature(1, "feature_1", 101, 1, 0.95, 0.92, 0.88, 0.85),
            create_selected_feature(2, "feature_2", 205, 2, 0.93, 0.90, 0.85, 0.82),
            create_selected_feature(3, "feature_3", 350, 3, 0.91, 0.88, 0.82, 0.79),
            create_selected_feature(4, "feature_4", 467, 4, 0.89, 0.86, 0.79, 0.76),
            create_selected_feature(5, "feature_5", 123, 5, 0.87, 0.84, 0.76, nothing)
        ]
        
        write_selected_features!(writer, features)
        
        # Verify features were written
        stored_features = get_selected_features(writer, run_id)
        @test nrow(stored_features) == 5
        @test stored_features.feature_id == [1, 2, 3, 4, 5]
        @test stored_features.feature_name == ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]
        @test stored_features.original_feature_id == [101, 205, 350, 467, 123]
        @test stored_features.selection_rank == [1, 2, 3, 4, 5]
        @test stored_features.selection_score ≈ [0.95, 0.93, 0.91, 0.89, 0.87]
        @test stored_features.selection_confidence ≈ [0.92, 0.90, 0.88, 0.86, 0.84]
        @test stored_features.selection_frequency ≈ [0.88, 0.85, 0.82, 0.79, 0.76]
        
        # Check that run was updated with feature count
        runs = get_run_history(writer)
        @test runs.total_features_output[1] == 5
        
        close_database!(writer)
    end
    
    @testset "Ensemble Metrics Writing" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test ensemble metrics
        metrics = [
            create_ensemble_metric("convergence_score", 0.92, "CONVERGENCE"),
            create_ensemble_metric("diversity_index", 0.75, "DIVERSITY"),
            create_ensemble_metric("consensus_strength", 0.88, "CONSENSUS"),
            create_ensemble_metric("execution_time", 1234.5, "PERFORMANCE"),
            create_ensemble_metric("memory_usage", 0.65, "RESOURCE")
        ]
        
        write_ensemble_metrics!(writer, metrics)
        
        # Verify metrics were written
        stored_metrics = get_ensemble_metrics(writer, run_id)
        @test nrow(stored_metrics) == 5
        @test Set(stored_metrics.metric_name) == Set(["convergence_score", "diversity_index", 
                                                     "consensus_strength", "execution_time", 
                                                     "memory_usage"])
        @test stored_metrics.metric_value[1] ≈ 0.92
        @test stored_metrics.metric_type[1] == "CONVERGENCE"
        
        close_database!(writer)
    end
    
    @testset "Tree Statistics Writing" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test tree statistics
        stats = [
            create_tree_statistic(1, 1, 5000, 15, 0.85, 4500, 0.3, 50, 12.5),
            create_tree_statistic(2, 1, 4800, 14, 0.87, 4200, 0.35, 48, 11.8),
            create_tree_statistic(3, 2, 5200, 16, 0.83, nothing, 0.28, 52, 13.2),
            create_tree_statistic(4, 2, 4900, 13, 0.89, 4300, 0.32, 46, 12.1)
        ]
        
        write_tree_statistics!(writer, stats)
        
        # Verify statistics were written
        stored_stats = get_tree_statistics(writer, run_id)
        @test nrow(stored_stats) == 4
        @test stored_stats.tree_id == [1, 2, 3, 4]
        @test stored_stats.gpu_id == [1, 1, 2, 2]
        @test stored_stats.total_iterations == [5000, 4800, 5200, 4900]
        @test stored_stats.max_depth == [15, 14, 16, 13]
        @test stored_stats.final_score ≈ [0.85, 0.87, 0.83, 0.89]
        @test stored_stats.exploration_ratio ≈ [0.3, 0.35, 0.28, 0.32]
        @test stored_stats.unique_features_explored == [50, 48, 52, 46]
        @test stored_stats.execution_time_seconds ≈ [12.5, 11.8, 13.2, 12.1]
        
        close_database!(writer)
    end
    
    @testset "Consensus History Writing" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test consensus history
        history = [
            create_consensus_entry(1, 101, 25, 0.65, 0.25),
            create_consensus_entry(1, 205, 30, 0.78, 0.30),
            create_consensus_entry(1, 350, 35, 0.82, 0.35),
            create_consensus_entry(2, 101, 28, 0.68, 0.28),
            create_consensus_entry(2, 205, 32, 0.80, 0.32),
            create_consensus_entry(2, 350, 38, 0.85, 0.38)
        ]
        
        write_consensus_history!(writer, history)
        
        # Verify history was written
        stored_history = get_consensus_history(writer, run_id)
        @test nrow(stored_history) == 6
        @test stored_history.iteration == [1, 1, 1, 2, 2, 2]
        @test stored_history.feature_id == [101, 205, 350, 101, 205, 350]
        @test stored_history.vote_count == [25, 30, 35, 28, 32, 38]
        @test stored_history.weighted_score ≈ [0.65, 0.78, 0.82, 0.68, 0.80, 0.85]
        @test stored_history.selection_probability ≈ [0.25, 0.30, 0.35, 0.28, 0.32, 0.38]
        
        close_database!(writer)
    end
    
    @testset "Performance Metrics Writing" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test performance metrics
        metrics = [
            create_performance_metric(1, 0.85, 8192.0, 12288.0, 75.5, 2.3),
            create_performance_metric(2, 0.82, 7680.0, 12288.0, 73.2, 2.1),
            create_performance_metric(1, 0.88, 8448.0, 12288.0, 76.8, 2.4),
            create_performance_metric(2, 0.79, 7424.0, 12288.0, 72.9, 2.0)
        ]
        
        write_performance_metrics!(writer, metrics)
        
        # Verify metrics were written
        stored_metrics = get_performance_metrics(writer, run_id)
        @test nrow(stored_metrics) == 4
        @test stored_metrics.gpu_id == [1, 2, 1, 2]
        @test stored_metrics.gpu_utilization ≈ [0.85, 0.82, 0.88, 0.79]
        @test stored_metrics.gpu_memory_used ≈ [8192.0, 7680.0, 8448.0, 7424.0]
        @test stored_metrics.gpu_memory_total ≈ [12288.0, 12288.0, 12288.0, 12288.0]
        @test stored_metrics.gpu_temperature ≈ [75.5, 73.2, 76.8, 72.9]
        @test stored_metrics.throughput_trees_per_second ≈ [2.3, 2.1, 2.4, 2.0]
        
        close_database!(writer)
    end
    
    @testset "Feature Validation" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test selected features
        features = [
            create_selected_feature(1, "feature_1", 101, 1, 0.95, 0.92, 0.88, 0.85),
            create_selected_feature(2, "feature_2", 205, 2, 0.93, 0.90, 0.85, 0.82),
            create_selected_feature(3, "feature_3", 350, 3, 0.91, 0.88, 0.82, 0.79)
        ]
        
        write_selected_features!(writer, features)
        
        # Test validation with valid features
        original_features = [101, 205, 350, 467, 123]
        validation = validate_features!(writer, original_features)
        
        @test validation.original_feature_count == 5
        @test validation.validated_feature_count == 3
        @test validation.missing_features == [467, 123]
        @test isempty(validation.duplicate_features)
        @test isempty(validation.invalid_features)
        @test validation.validation_status == "WARNING"
        
        # Test validation with invalid features
        features_with_invalid = [
            create_selected_feature(1, "feature_1", 101, 1, 0.95, 0.92, 0.88, 0.85),
            create_selected_feature(2, "feature_2", 999, 2, 0.93, 0.90, 0.85, 0.82)  # Invalid feature
        ]
        
        write_selected_features!(writer, features_with_invalid)
        validation = validate_features!(writer, original_features)
        
        @test validation.invalid_features == [999]
        @test validation.validation_status == "INVALID"
        
        close_database!(writer)
    end
    
    @testset "Run Completion" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Complete the run
        execution_time = 1234.56
        complete_run!(writer, COMPLETED, execution_time)
        
        @test writer.current_run_id === nothing
        
        # Verify run was updated
        runs = get_run_history(writer)
        @test runs.completion_status[1] == "COMPLETED"
        @test runs.execution_time_seconds[1] ≈ execution_time
        
        close_database!(writer)
    end
    
    @testset "Transaction Rollback on Error" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create invalid feature (missing required field)
        # This should cause a rollback
        try
            SQLite.execute(writer.db_connection, """
                INSERT INTO selected_features (run_id, feature_id) VALUES (?, ?)
            """, [run_id, 1])  # Missing required fields
            
            @test false  # Should not reach here
        catch e
            @test true  # Expected to fail
        end
        
        # Verify transaction was rolled back
        @test !writer.transaction_active
        
        close_database!(writer)
    end
    
    @testset "Configuration Serialization" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}(
            "int_val" => 42,
            "float_val" => 3.14159,
            "bool_val" => true,
            "string_val" => "hello world",
            "array_val" => [1, 2, 3, 4, 5],
            "mixed_array" => ["a", 1, true, 2.5]
        )
        
        run_id = initialize_run!(writer, config)
        
        # Verify serialization
        config_query = SQLite.execute(writer.db_connection, """
            SELECT config_key, config_value, config_type 
            FROM ensemble_config WHERE run_id = ?
        """, [run_id])
        
        config_dict = Dict(row[1] => (row[2], row[3]) for row in collect(config_query))
        
        @test config_dict["int_val"] == ("42", "INTEGER")
        @test config_dict["float_val"] == ("3.14159", "FLOAT")
        @test config_dict["bool_val"] == ("true", "BOOLEAN")
        @test config_dict["string_val"] == ("hello world", "STRING")
        @test config_dict["array_val"] == ("[1,2,3,4,5]", "ARRAY")
        @test config_dict["mixed_array"] == ("[\"a\",1,true,2.5]", "ARRAY")
        
        close_database!(writer)
    end
    
    @testset "Multiple Runs on Same Dataset" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        # Create first run
        config1 = Dict{String, Any}("total_features_input" => 500, "run_name" => "run1")
        run_id1 = initialize_run!(writer, config1)
        features1 = [create_selected_feature(1, "feature_1", 101, 1, 0.95, 0.92, 0.88, 0.85)]
        write_selected_features!(writer, features1)
        complete_run!(writer, COMPLETED, 100.0)
        
        # Create second run
        config2 = Dict{String, Any}("total_features_input" => 500, "run_name" => "run2")
        run_id2 = initialize_run!(writer, config2)
        features2 = [create_selected_feature(1, "feature_2", 205, 1, 0.93, 0.90, 0.85, 0.82)]
        write_selected_features!(writer, features2)
        complete_run!(writer, COMPLETED, 120.0)
        
        # Verify both runs exist
        runs = get_run_history(writer)
        @test nrow(runs) == 2
        @test Set(runs.run_id) == Set([run_id1, run_id2])
        
        # Verify features are separate
        features1_stored = get_selected_features(writer, run_id1)
        features2_stored = get_selected_features(writer, run_id2)
        
        @test features1_stored.feature_name[1] == "feature_1"
        @test features2_stored.feature_name[1] == "feature_2"
        
        close_database!(writer)
    end
    
    @testset "Database Maintenance" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        # Create multiple runs with different timestamps
        old_timestamp = now() - Day(45)
        recent_timestamp = now() - Day(15)
        
        # Create old run
        config = Dict{String, Any}("total_features_input" => 500)
        run_id_old = initialize_run!(writer, config)
        
        # Manually update timestamp to make it old
        SQLite.execute(writer.db_connection, """
            UPDATE result_runs SET run_timestamp = ? WHERE run_id = ?
        """, [old_timestamp, run_id_old])
        
        complete_run!(writer, COMPLETED, 100.0)
        
        # Create recent run
        run_id_recent = initialize_run!(writer, config)
        SQLite.execute(writer.db_connection, """
            UPDATE result_runs SET run_timestamp = ? WHERE run_id = ?
        """, [recent_timestamp, run_id_recent])
        
        complete_run!(writer, COMPLETED, 120.0)
        
        # Verify both runs exist
        runs = get_run_history(writer)
        @test nrow(runs) == 2
        
        # Cleanup old runs (keep 30 days)
        cleanup_old_runs!(writer, 30)
        
        # Verify only recent run remains
        runs = get_run_history(writer)
        @test nrow(runs) == 1
        @test runs.run_id[1] == run_id_recent
        
        # Test vacuum
        vacuum_database!(writer)
        
        close_database!(writer)
    end
    
    @testset "Error Handling" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        # Test operations without initialized run
        @test_throws ArgumentError write_selected_features!(writer, SelectedFeature[])
        @test_throws ArgumentError write_ensemble_metrics!(writer, EnsembleMetric[])
        @test_throws ArgumentError write_tree_statistics!(writer, TreeStatistic[])
        @test_throws ArgumentError write_consensus_history!(writer, ConsensusEntry[])
        @test_throws ArgumentError write_performance_metrics!(writer, PerformanceMetric[])
        @test_throws ArgumentError validate_features!(writer, Int[])
        @test_throws ArgumentError complete_run!(writer, COMPLETED, 100.0)
        
        close_database!(writer)
    end
    
    @testset "Helper Functions" begin
        # Test create_selected_feature
        feature = create_selected_feature(1, "test_feature", 101, 1, 0.95, 0.92, 0.88, 0.85)
        @test feature.feature_id == 1
        @test feature.feature_name == "test_feature"
        @test feature.original_feature_id == 101
        @test feature.selection_rank == 1
        @test feature.selection_score == 0.95
        @test feature.selection_confidence == 0.92
        @test feature.selection_frequency == 0.88
        @test feature.feature_importance == 0.85
        
        # Test create_ensemble_metric
        metric = create_ensemble_metric("test_metric", 0.75, "TEST")
        @test metric.metric_name == "test_metric"
        @test metric.metric_value == 0.75
        @test metric.metric_type == "TEST"
        @test isa(metric.measurement_timestamp, DateTime)
        
        # Test create_tree_statistic
        stat = create_tree_statistic(1, 1, 5000, 15, 0.85, 4500, 0.3, 50, 12.5)
        @test stat.tree_id == 1
        @test stat.gpu_id == 1
        @test stat.total_iterations == 5000
        @test stat.max_depth == 15
        @test stat.final_score == 0.85
        @test stat.convergence_iteration == 4500
        @test stat.exploration_ratio == 0.3
        @test stat.unique_features_explored == 50
        @test stat.execution_time_seconds == 12.5
        
        # Test create_consensus_entry
        entry = create_consensus_entry(1, 101, 25, 0.65, 0.25)
        @test entry.iteration == 1
        @test entry.feature_id == 101
        @test entry.vote_count == 25
        @test entry.weighted_score == 0.65
        @test entry.selection_probability == 0.25
        @test isa(entry.timestamp, DateTime)
        
        # Test create_performance_metric
        perf = create_performance_metric(1, 0.85, 8192.0, 12288.0, 75.5, 2.3)
        @test perf.gpu_id == 1
        @test perf.gpu_utilization == 0.85
        @test perf.gpu_memory_used == 8192.0
        @test perf.gpu_memory_total == 12288.0
        @test perf.gpu_temperature == 75.5
        @test perf.throughput_trees_per_second == 2.3
        @test isa(perf.metric_timestamp, DateTime)
    end
    
    @testset "Database Schema Validation" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        # Test table existence
        tables = SQLite.tables(writer.db_connection)
        table_names = [table.name for table in tables]
        
        required_tables = ["result_runs", "ensemble_config", "selected_features", 
                          "ensemble_metrics", "tree_statistics", "consensus_history",
                          "performance_metrics", "feature_validation"]
        
        for table in required_tables
            @test table in table_names
        end
        
        # Test column existence for key tables
        columns = SQLite.columns(writer.db_connection, "result_runs")
        column_names = [col.name for col in columns]
        
        required_columns = ["run_id", "dataset_name", "run_timestamp", "schema_version",
                           "completion_status", "total_features_input", "total_features_output",
                           "execution_time_seconds", "created_at"]
        
        for col in required_columns
            @test col in column_names
        end
        
        close_database!(writer)
    end
    
    @testset "Concurrent Access Safety" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Test concurrent writes (simulated)
        metrics = [create_ensemble_metric("test_metric_$i", Float64(i), "TEST") for i in 1:10]
        
        # This should work without errors due to locking
        write_ensemble_metrics!(writer, metrics)
        
        stored_metrics = get_ensemble_metrics(writer, run_id)
        @test nrow(stored_metrics) == 10
        
        close_database!(writer)
    end
    
    @testset "Large Data Handling" begin
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 5000)
        run_id = initialize_run!(writer, config)
        
        # Create large number of features
        features = [create_selected_feature(i, "feature_$i", 1000+i, i, 
                                          0.95 - i*0.001, 0.92 - i*0.001, 
                                          0.88 - i*0.001, 0.85 - i*0.001) 
                   for i in 1:1000]
        
        write_selected_features!(writer, features)
        
        # Verify all features were written
        stored_features = get_selected_features(writer, run_id)
        @test nrow(stored_features) == 1000
        @test stored_features.feature_id[1] == 1
        @test stored_features.feature_id[end] == 1000
        
        close_database!(writer)
    end
    
    @testset "Result Status Enum" begin
        @test PENDING == ResultStatus(1)
        @test RUNNING == ResultStatus(2)
        @test COMPLETED == ResultStatus(3)
        @test FAILED == ResultStatus(4)
        @test CANCELLED == ResultStatus(5)
        
        @test string(COMPLETED) == "COMPLETED"
        @test string(FAILED) == "FAILED"
    end
    
    # Cleanup
    rm(temp_dir, recursive=true)
end