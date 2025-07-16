using Test
using SQLite
using DataFrames
using Random
using Dates
using Statistics

# Include all database modules
include("../../src/database/Database.jl")
using .Database

@testset "Database Integration Tests" begin
    
    # Create test database
    test_db_path = tempname() * ".db"
    
    # Initialize database with sample data
    function setup_test_database()
        db = SQLite.DB(test_db_path)
        
        # Create large test dataset
        SQLite.execute(db, """
            CREATE TABLE test_dataset (
                id INTEGER PRIMARY KEY,
                patient_id INTEGER,
                $(join(["feature_$i REAL" for i in 1:500], ", ")),
                target INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Insert data in batches
        Random.seed!(42)
        n_samples = 100_000
        batch_size = 1000
        
        println("Creating test dataset with $n_samples samples...")
        
        for batch_start in 1:batch_size:n_samples
            batch_end = min(batch_start + batch_size - 1, n_samples)
            
            # Prepare batch data
            for i in batch_start:batch_end
                values = [i, i]  # id, patient_id
                
                # Generate features (some informative, some noise)
                for j in 1:500
                    if j <= 50  # First 50 are informative
                        value = randn() + (i % 2 == 0 ? 0.5 : -0.5)
                    else  # Rest are noise
                        value = randn()
                    end
                    push!(values, value)
                end
                
                push!(values, i % 2)  # target (binary)
                push!(values, "2024-01-01")  # created_at
                push!(values, "2024-01-01")  # updated_at
                
                # Insert row
                placeholders = join(["?" for _ in 1:505], ", ")
                SQLite.execute(db, 
                    "INSERT INTO test_dataset VALUES ($placeholders)", values)
            end
            
            if batch_end % 10_000 == 0
                println("  Inserted $batch_end/$n_samples rows...")
            end
        end
        
        # Create metadata table
        Database.create_metadata_table(db)
        
        # Insert metadata
        Database.insert_metadata(db, Database.DatasetMetadata(
            table_name = "test_dataset",
            excluded_columns = ["created_at", "updated_at"],
            id_columns = ["id", "patient_id"],
            target_column = "target",
            feature_count = 500,
            row_count = n_samples,
            metadata_version = "1.0",
            additional_info = Dict("test" => true, "features" => "synthetic")
        ))
        
        SQLite.close(db)
        println("Test database created successfully!")
    end
    
    # Setup
    setup_test_database()
    
    @testset "End-to-End Data Loading Pipeline" begin
        # Create connection pool
        pool = Database.create_database_connection(test_db_path)
        
        # Parse metadata
        metadata = Database.parse_metadata(pool, "test_dataset")
        @test metadata.table_name == "test_dataset"
        @test length(metadata.excluded_columns) == 2
        @test metadata.feature_count == 500
        @test metadata.row_count == 100_000
        
        # Validate columns
        validation = Database.validate_dataset(pool, "test_dataset")
        @test validation.valid == true
        @test length(validation.column_info) >= 500  # All features + target
        
        # Create data loader with progress tracking
        println("\nTesting data loading with progress tracking...")
        
        progress_updates = []
        progress_callback = function(info)
            push!(progress_updates, info)
            if length(progress_updates) % 5 == 0
                println("  Progress: $(round(info.percentage, digits=1))% " *
                       "($(info.rows_processed)/$(info.total_rows) rows)")
            end
        end
        
        iterator = Database.load_dataset_lazy(
            pool, "test_dataset",
            chunk_size = 5000,
            show_progress = false,  # Use custom callback instead
            progress_callback = progress_callback
        )
        
        # Process chunks
        chunk_count = 0
        total_rows = 0
        feature_sums = zeros(500)
        
        for chunk in iterator
            chunk_count += 1
            total_rows += size(chunk.data, 1)
            
            # Extract features (skip id columns and target)
            feature_data = Matrix(chunk.data[:, 3:502])
            feature_sums .+= vec(sum(feature_data, dims=1))
            
            # Validate chunk
            @test size(chunk.data, 2) == 504  # 2 ids + 500 features + 1 target + 1 excluded
            @test chunk.chunk_index == chunk_count
        end
        
        @test chunk_count == 20  # 100k rows / 5k chunk size
        @test total_rows == 100_000
        @test length(progress_updates) > 0
        @test progress_updates[end].percentage == 100.0
        
        # Check feature statistics
        feature_means = feature_sums ./ total_rows
        @test abs(mean(feature_means[1:50])) > 0.01  # Informative features
        @test abs(mean(feature_means[51:500])) < 0.01  # Noise features
        
        Database.close_pool(pool)
    end
    
    @testset "MCTS Result Writing and Querying" begin
        pool = Database.create_database_connection(test_db_path)
        
        # Create result writer
        writer = Database.ResultWriter(pool, 
                                     table_prefix="test_mcts",
                                     batch_size=100)
        
        println("\nSimulating MCTS iterations...")
        
        # Simulate MCTS iterations
        for iter in 1:1000
            # Generate random feature selection
            n_features = rand(5:20)
            selected = sort(randperm(500)[1:n_features])
            scores = rand(n_features) .* (1.0 - 0.5 * iter / 1000)  # Decreasing scores
            
            result = Database.MCTSResult(
                iter,
                selected,
                scores,
                mean(scores) + 0.1 * randn(),  # Objective with noise
                Dict("algorithm" => "UCB", "temperature" => 1.0 / sqrt(iter)),
                now()
            )
            
            Database.write_results!(writer, result)
            
            # Update feature importance periodically
            if iter % 100 == 0
                for (idx, feature_idx) in enumerate(selected[1:min(5, end)])
                    importance = Database.FeatureImportance(
                        feature_idx,
                        "feature_$feature_idx",
                        scores[idx] * sqrt(iter),
                        iter ÷ 10,
                        scores[idx]
                    )
                    Database.write_feature_importance!(writer, importance)
                end
            end
        end
        
        # Ensure all data is written
        close(writer)
        
        # Query best results
        writer2 = Database.ResultWriter(pool, table_prefix="test_mcts")
        best_results = Database.get_best_results(writer2, 10)
        
        @test size(best_results, 1) == 10
        @test issorted(best_results.objective_value, rev=true)
        
        # Query feature importance
        importance = Database.get_feature_importance(writer2)
        @test size(importance, 1) > 0
        @test issorted(importance.importance_score, rev=true)
        
        println("  Top 5 important features:")
        for i in 1:min(5, size(importance, 1))
            println("    $(importance.feature_name[i]): " * 
                   "$(round(importance.importance_score[i], digits=3))")
        end
        
        close(writer2)
        Database.close_pool(pool)
    end
    
    @testset "Checkpoint Management" begin
        pool = Database.create_database_connection(test_db_path)
        
        # Create checkpoint manager
        manager = Database.CheckpointManager(pool,
                                           compression_level=6,
                                           retention_count=5)
        
        println("\nTesting checkpoint save/load...")
        
        # Save checkpoints
        for iter in [100, 200, 500, 1000, 2000, 5000]
            # Simulate MCTS tree state
            tree_state = Dict(
                "nodes" => rand(1:1000, iter),
                "values" => rand(iter),
                "visits" => rand(1:100, iter)
            )
            
            checkpoint = Database.Checkpoint(
                iter,
                tree_state,
                sort(randperm(500)[1:20]),  # Selected features
                rand(20),                    # Scores
                Dict("iteration" => iter, "best_value" => maximum(rand(20))),
                now()
            )
            
            saved = Database.save_checkpoint!(manager, checkpoint, force=true)
            @test saved == true
        end
        
        # List checkpoints
        checkpoints = Database.list_checkpoints(manager)
        @test size(checkpoints, 1) <= 5  # Retention policy
        
        # Load latest checkpoint
        latest = Database.load_checkpoint(manager)
        @test latest.iteration == 5000
        @test length(latest.feature_selection) == 20
        
        # Get statistics
        stats = Database.get_checkpoint_stats(manager)
        @test stats["total_checkpoints"] <= 5
        @test stats["space_saved_mb"] > 0
        @test stats["avg_compression_ratio"] > 1.0
        
        println("  Checkpoint compression ratio: " * 
               "$(round(stats["avg_compression_ratio"], digits=2))")
        println("  Space saved: $(round(stats["space_saved_mb"], digits=2)) MB")
        
        Database.close_pool(pool)
    end
    
    @testset "Concurrent Operations" begin
        pool = Database.create_database_connection(test_db_path, max_size=10)
        
        println("\nTesting concurrent database operations...")
        
        # Concurrent reading
        tasks = []
        results = Channel{Int}(10)
        
        for i in 1:10
            task = Threads.@spawn begin
                try
                    # Each task loads a chunk
                    iterator = Database.create_chunk_iterator(
                        pool, "test_dataset",
                        chunk_size = 10_000,
                        show_progress = false
                    )
                    
                    chunk = first(iterator)
                    put!(results, size(chunk.data, 1))
                    close(iterator)
                catch e
                    @error "Task error" exception=e
                    put!(results, 0)
                end
            end
            push!(tasks, task)
        end
        
        # Wait for all tasks
        for task in tasks
            wait(task)
        end
        
        close(results)
        
        # Check results
        chunk_sizes = collect(results)
        @test all(chunk_sizes .== 10_000)
        @test length(chunk_sizes) == 10
        
        Database.close_pool(pool)
    end
    
    @testset "Performance Benchmarks" begin
        pool = Database.create_database_connection(test_db_path)
        
        println("\nRunning performance benchmarks...")
        
        # Benchmark data loading
        loading_time = @elapsed begin
            iterator = Database.create_chunk_iterator(
                pool, "test_dataset",
                chunk_size = 20_000,
                show_progress = false
            )
            
            row_count = 0
            for chunk in iterator
                row_count += size(chunk.data, 1)
            end
            
            @test row_count == 100_000
        end
        
        throughput = 100_000 / loading_time
        println("  Data loading: $(round(throughput, digits=0)) rows/second")
        
        # Benchmark result writing
        writer = Database.ResultWriter(pool,
                                     table_prefix="bench",
                                     batch_size=1000,
                                     async_writing=true)
        
        write_time = @elapsed begin
            results = []
            for i in 1:10_000
                push!(results, Database.MCTSResult(
                    i,
                    rand(1:500, 10),
                    rand(10),
                    rand(),
                    Dict("benchmark" => true),
                    now()
                ))
            end
            
            Database.batch_write_results!(writer, results)
            close(writer)
        end
        
        write_throughput = 10_000 / write_time
        println("  Result writing: $(round(write_throughput, digits=0)) results/second")
        
        # Benchmark checkpoint compression
        manager = Database.CheckpointManager(pool,
                                           table_name="bench_checkpoint",
                                           compression_level=6)
        
        large_state = Dict("data" => rand(1000, 1000))  # ~8MB uncompressed
        
        checkpoint_time = @elapsed begin
            checkpoint = Database.Checkpoint(
                1,
                large_state,
                collect(1:50),
                rand(50),
                Dict(),
                now()
            )
            
            Database.save_checkpoint!(manager, checkpoint, force=true)
        end
        
        stats = Database.get_checkpoint_stats(manager)
        compression_ratio = stats["avg_compression_ratio"]
        
        println("  Checkpoint save: $(round(checkpoint_time * 1000, digits=1)) ms")
        println("  Compression ratio: $(round(compression_ratio, digits=2))x")
        
        Database.close_pool(pool)
    end
    
    @testset "Error Recovery" begin
        pool = Database.create_database_connection(test_db_path)
        
        # Test recovery from missing metadata
        @test_throws ErrorException Database.parse_metadata(pool, "nonexistent_table")
        
        # Test handling of corrupted data
        iterator = Database.create_chunk_iterator(
            pool, "test_dataset",
            chunk_size = 1000,
            where_clause = "feature_1 > 1000"  # No matches
        )
        
        chunks = collect(iterator)
        @test isempty(chunks)
        
        # Test checkpoint recovery
        manager = Database.CheckpointManager(pool, table_name="recovery_test")
        
        # Try to load non-existent checkpoint
        missing = Database.load_checkpoint(manager, 99999)
        @test isnothing(missing)
        
        Database.close_pool(pool)
    end
    
    # Cleanup
    rm(test_db_path, force=true)
    
    println("\n✅ All integration tests passed!")
end