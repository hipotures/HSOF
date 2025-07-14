using Test
using SQLite
using DataFrames
using JSON
using Dates
using Random

# Include modules
include("../../src/database/connection_pool.jl")
include("../../src/database/result_writer.jl")
using .ConnectionPool
using .ResultWriter

@testset "Result Writer Tests" begin
    # Create test database
    test_db_path = tempname() * ".db"
    db = SQLite.DB(test_db_path)
    SQLite.close(db)
    
    # Create connection pool
    pool = SQLitePool(test_db_path)
    
    @testset "Table Creation" begin
        writer = ResultWriter.ResultWriter(pool, table_prefix="test")
        
        # Check tables exist
        with_connection(pool) do conn
            tables = DBInterface.execute(conn.db, 
                "SELECT name FROM sqlite_master WHERE type='table'") |> DataFrame
            
            table_names = Set(tables.name)
            @test "test_results" in table_names
            @test "test_feature_importance" in table_names
            @test "test_checkpoints" in table_names
            
            # Check indices exist
            indices = DBInterface.execute(conn.db,
                "SELECT name FROM sqlite_master WHERE type='index'") |> DataFrame
            
            index_names = Set(indices.name)
            @test "idx_test_results_iteration" in index_names
            @test "idx_test_results_objective" in index_names
        end
        
        close(writer)
    end
    
    @testset "Single Result Writing" begin
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="single",
                                         batch_size=5,
                                         async_writing=false)
        
        # Create test result
        result = ResultWriter.MCTSResult(
            1,                              # iteration
            [1, 5, 7, 12],                 # selected_features
            [0.9, 0.8, 0.7, 0.6],         # feature_scores
            0.95,                          # objective_value
            Dict("method" => "greedy"),    # metadata
            now()                          # timestamp
        )
        
        # Write result
        ResultWriter.write_results!(writer, result)
        
        # Should not be written yet (batch size = 5)
        stats = ResultWriter.get_write_statistics(writer)
        @test stats["results_buffered"] == 1
        @test stats["total_results_written"] == 0
        
        # Write more to trigger batch
        for i in 2:5
            result = ResultWriter.MCTSResult(
                i,
                rand(1:100, rand(3:10)),
                rand(rand(3:10)),
                rand(),
                Dict("iteration" => i),
                now()
            )
            ResultWriter.write_results!(writer, result)
        end
        
        # Should be written now
        stats = ResultWriter.get_write_statistics(writer)
        @test stats["results_buffered"] == 0
        @test stats["total_results_written"] == 5
        
        # Verify in database
        with_connection(pool) do conn
            count_result = DBInterface.execute(conn.db,
                "SELECT COUNT(*) as count FROM single_results") |> DataFrame
            @test count_result.count[1] == 5
        end
        
        close(writer)
    end
    
    @testset "Batch Writing" begin
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="batch",
                                         batch_size=100,
                                         async_writing=false)
        
        # Create batch of results
        results = []
        for i in 1:50
            push!(results, ResultWriter.MCTSResult(
                i,
                sort(randperm(100)[1:10]),
                rand(10),
                rand() * 0.5 + 0.5,  # 0.5 to 1.0
                Dict("batch" => true, "index" => i),
                now()
            ))
        end
        
        # Batch write
        ResultWriter.batch_write_results!(writer, results)
        
        # Should be written immediately
        stats = ResultWriter.get_write_statistics(writer)
        @test stats["total_results_written"] == 50
        
        close(writer)
    end
    
    @testset "Feature Importance" begin
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="importance",
                                         async_writing=false)
        
        # Write feature importance
        for i in 1:20
            importance = ResultWriter.FeatureImportance(
                i,                           # feature_index
                "feature_$i",               # feature_name
                rand(),                     # importance_score
                rand(1:100),               # selection_count
                rand() * 0.1               # average_contribution
            )
            ResultWriter.write_feature_importance!(writer, importance)
        end
        
        # Force flush
        ResultWriter.flush_importance(writer)
        
        # Query importance
        importance_df = ResultWriter.get_feature_importance(writer)
        @test size(importance_df, 1) == 20
        @test issorted(importance_df.importance_score, rev=true)
        
        # Update existing feature
        updated = ResultWriter.FeatureImportance(
            5,                    # existing index
            "feature_5",
            0.99,                # new high score
            200,
            0.15
        )
        ResultWriter.write_feature_importance!(writer, updated)
        ResultWriter.flush_importance(writer)
        
        # Verify update
        importance_df = ResultWriter.get_feature_importance(writer)
        @test importance_df[1, :feature_index] == 5  # Should be first now
        @test importance_df[1, :importance_score] == 0.99
        
        close(writer)
    end
    
    @testset "Async Writing" begin
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="async",
                                         batch_size=10,
                                         flush_interval=0.5,
                                         async_writing=true)
        
        # Write results rapidly
        for i in 1:25
            result = ResultWriter.MCTSResult(
                i,
                rand(1:100, 5),
                rand(5),
                rand(),
                Dict("async" => true),
                now()
            )
            ResultWriter.write_results!(writer, result)
            sleep(0.01)  # Small delay
        end
        
        # Wait for async writes
        sleep(1.0)
        
        # Check all written
        with_connection(pool) do conn
            count_result = DBInterface.execute(conn.db,
                "SELECT COUNT(*) as count FROM async_results") |> DataFrame
            @test count_result.count[1] == 25
        end
        
        close(writer)
    end
    
    @testset "Query Functions" begin
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="query",
                                         async_writing=false)
        
        # Write results with varying objective values
        for i in 1:100
            result = ResultWriter.MCTSResult(
                i,
                rand(1:50, rand(3:8)),
                rand(rand(3:8)),
                rand() * 0.8 + 0.1,  # 0.1 to 0.9
                Dict("test" => true),
                now()
            )
            ResultWriter.write_results!(writer, result)
        end
        
        # Force flush
        close(writer)
        
        # Create new writer for querying
        writer2 = ResultWriter.ResultWriter(pool, 
                                          table_prefix="query",
                                          async_writing=false)
        
        # Query best results
        best_results = ResultWriter.get_best_results(writer2, 10)
        @test size(best_results, 1) == 10
        @test issorted(best_results.objective_value, rev=true)
        @test all(best_results.objective_value .>= 0.1)
        
        close(writer2)
    end
    
    @testset "Transaction Safety" begin
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="trans",
                                         batch_size=50,
                                         async_writing=false)
        
        # Simulate failure during batch write
        results = []
        for i in 1:30
            push!(results, ResultWriter.MCTSResult(
                i,
                [1, 2, 3],
                [0.1, 0.2, 0.3],
                0.5,
                Dict(),
                now()
            ))
        end
        
        # Add invalid result that will cause error
        push!(results, ResultWriter.MCTSResult(
            31,
            [1, 2, 3],
            [0.1, 0.2],  # Mismatched lengths - would cause JSON error
            0.5,
            Dict(),
            now()
        ))
        
        # This should handle the error gracefully
        try
            ResultWriter.batch_write_results!(writer, results)
        catch
            # Expected
        end
        
        # First 30 should still be written
        stats = ResultWriter.get_write_statistics(writer)
        @test stats["total_results_written"] == 30
        
        close(writer)
    end
    
    @testset "WAL Mode" begin
        # Test with WAL enabled
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="wal",
                                         wal_enabled=true,
                                         async_writing=false)
        
        # Check WAL mode is enabled
        with_connection(pool) do conn
            mode_result = DBInterface.execute(conn.db,
                "PRAGMA journal_mode") |> DataFrame
            @test mode_result[1, 1] == "wal"
        end
        
        # Write some data
        for i in 1:10
            result = ResultWriter.MCTSResult(
                i,
                [i, i+1, i+2],
                [0.1, 0.2, 0.3],
                rand(),
                Dict("wal" => true),
                now()
            )
            ResultWriter.write_results!(writer, result)
        end
        
        close(writer)
    end
    
    @testset "Concurrent Writing" begin
        writer = ResultWriter.ResultWriter(pool, 
                                         table_prefix="concurrent",
                                         batch_size=100,
                                         async_writing=true)
        
        # Multiple threads writing
        tasks = []
        for thread in 1:5
            task = Threads.@spawn begin
                for i in 1:20
                    result = ResultWriter.MCTSResult(
                        thread * 100 + i,
                        rand(1:100, 5),
                        rand(5),
                        rand(),
                        Dict("thread" => thread),
                        now()
                    )
                    ResultWriter.write_results!(writer, result)
                    sleep(0.001)
                end
            end
            push!(tasks, task)
        end
        
        # Wait for all tasks
        for task in tasks
            wait(task)
        end
        
        # Wait for async writes
        sleep(1.0)
        close(writer)
        
        # Verify all written
        with_connection(pool) do conn
            count_result = DBInterface.execute(conn.db,
                "SELECT COUNT(*) as count FROM concurrent_results") |> DataFrame
            @test count_result.count[1] == 100  # 5 threads * 20 results
        end
    end
    
    # Cleanup
    close_pool(pool)
    rm(test_db_path, force=true)
end