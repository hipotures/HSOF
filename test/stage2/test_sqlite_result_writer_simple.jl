using Test
using SQLite
using DataFrames
using JSON3
using Dates
using UUIDs

# Include the SQLite result writer module
include("../../src/stage2/sqlite_result_writer.jl")

@testset "SQLite Result Writer Simple Tests" begin
    
    # Create temporary database for testing
    temp_dir = "/tmp/test_sqlite_$(uuid4())"
    mkpath(temp_dir)
    
    @testset "Basic Writer Creation" begin
        db_path = joinpath(temp_dir, "test1.sqlite")
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        @test writer.db_path == db_path
        @test writer.dataset_name == "test_dataset"
        @test writer.current_run_id === nothing
        @test !writer.transaction_active
        @test isempty(writer.feature_cache)
        
        close_database!(writer)
    end
    
    @testset "Schema Creation" begin
        db_path = joinpath(temp_dir, "test2.sqlite")
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        # Check if main tables exist
        tables = SQLite.tables(writer.db_connection)
        table_names = [table.name for table in tables]
        
        @test "result_runs" in table_names
        @test "ensemble_config" in table_names
        @test "selected_features" in table_names
        
        close_database!(writer)
    end
    
    @testset "Run Initialization" begin
        db_path = joinpath(temp_dir, "test3.sqlite")
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}(
            "total_features_input" => 500,
            "tree_count" => 100
        )
        
        run_id = initialize_run!(writer, config)
        
        @test writer.current_run_id == run_id
        @test length(run_id) == 36  # UUID length
        
        close_database!(writer)
    end
    
    @testset "Selected Features Writing" begin
        db_path = joinpath(temp_dir, "test4.sqlite")
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create simple test features
        features = [
            create_selected_feature(1, "feature_1", 101, 1, 0.95, 0.92, 0.88, 0.85),
            create_selected_feature(2, "feature_2", 205, 2, 0.93, 0.90, 0.85, 0.82)
        ]
        
        write_selected_features!(writer, features)
        
        # Verify features were written
        stored_features = get_selected_features(writer, run_id)
        @test nrow(stored_features) == 2
        @test stored_features.feature_id == [1, 2]
        @test stored_features.feature_name == ["feature_1", "feature_2"]
        
        close_database!(writer)
    end
    
    @testset "Ensemble Metrics Writing" begin
        db_path = joinpath(temp_dir, "test5.sqlite")
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test metrics
        metrics = [
            create_ensemble_metric("convergence_score", 0.92, "CONVERGENCE"),
            create_ensemble_metric("diversity_index", 0.75, "DIVERSITY")
        ]
        
        write_ensemble_metrics!(writer, metrics)
        
        # Verify metrics were written
        stored_metrics = get_ensemble_metrics(writer, run_id)
        @test nrow(stored_metrics) == 2
        @test stored_metrics.metric_name[1] == "convergence_score"
        @test stored_metrics.metric_value[1] ≈ 0.92
        
        close_database!(writer)
    end
    
    @testset "Tree Statistics Writing" begin
        db_path = joinpath(temp_dir, "test6.sqlite")
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test tree statistics
        stats = [
            create_tree_statistic(1, 1, 5000, 15, 0.85, 4500, 0.3, 50, 12.5),
            create_tree_statistic(2, 1, 4800, 14, 0.87, 4200, 0.35, 48, 11.8)
        ]
        
        write_tree_statistics!(writer, stats)
        
        # Verify statistics were written
        stored_stats = get_tree_statistics(writer, run_id)
        @test nrow(stored_stats) == 2
        @test stored_stats.tree_id == [1, 2]
        @test stored_stats.gpu_id == [1, 1]
        @test stored_stats.total_iterations == [5000, 4800]
        
        close_database!(writer)
    end
    
    @testset "Run Completion" begin
        db_path = joinpath(temp_dir, "test7.sqlite")
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
    
    @testset "Result Status Enum" begin
        @test PENDING == ResultStatus(1)
        @test RUNNING == ResultStatus(2)
        @test COMPLETED == ResultStatus(3)
        @test FAILED == ResultStatus(4)
        @test CANCELLED == ResultStatus(5)
        
        @test string(COMPLETED) == "COMPLETED"
        @test string(FAILED) == "FAILED"
    end
    
    @testset "Feature Validation" begin
        db_path = joinpath(temp_dir, "test8.sqlite")
        writer = SQLiteResultWriter(db_path, "test_dataset")
        
        config = Dict{String, Any}("total_features_input" => 500)
        run_id = initialize_run!(writer, config)
        
        # Create test features
        features = [
            create_selected_feature(1, "feature_1", 101, 1, 0.95, 0.92, 0.88, 0.85),
            create_selected_feature(2, "feature_2", 205, 2, 0.93, 0.90, 0.85, 0.82)
        ]
        
        write_selected_features!(writer, features)
        
        # Test validation
        original_features = [101, 205, 350, 467, 123]
        validation = validate_features!(writer, original_features)
        
        @test validation.original_feature_count == 5
        @test validation.validated_feature_count == 2
        @test validation.missing_features == [350, 467, 123]
        @test isempty(validation.duplicate_features)
        @test isempty(validation.invalid_features)
        @test validation.validation_status == "WARNING"
        
        close_database!(writer)
    end
    
    # Cleanup
    rm(temp_dir, recursive=true)
end