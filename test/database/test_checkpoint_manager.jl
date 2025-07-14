using Test
using SQLite
using DataFrames
using Dates
using Random

# Include modules
include("../../src/database/connection_pool.jl")
include("../../src/database/result_writer.jl")
include("../../src/database/checkpoint_manager.jl")
using .ConnectionPool
using .ResultWriter
using .CheckpointManager

# Mock MCTS tree state
struct MockTreeState
    nodes::Dict{Int, Vector{Int}}
    values::Dict{Int, Float64}
    visits::Dict{Int, Int}
end

@testset "Checkpoint Manager Tests" begin
    # Create test database
    test_db_path = tempname() * ".db"
    db = SQLite.DB(test_db_path)
    SQLite.close(db)
    
    # Create connection pool
    pool = SQLitePool(test_db_path)
    
    @testset "Table Creation" begin
        manager = CheckpointManager.CheckpointManager(pool, table_name="test_checkpoints")
        
        # Check table exists
        with_connection(pool) do conn
            tables = DBInterface.execute(conn.db, 
                "SELECT name FROM sqlite_master WHERE type='table'") |> DataFrame
            
            @test "test_checkpoints" in tables.name
            
            # Check indices
            indices = DBInterface.execute(conn.db,
                "SELECT name FROM sqlite_master WHERE type='index'") |> DataFrame
            
            index_names = Set(indices.name)
            @test "idx_test_checkpoints_iteration" in index_names
            @test "idx_test_checkpoints_timestamp" in index_names
        end
    end
    
    @testset "Basic Checkpoint Save/Load" begin
        manager = CheckpointManager.CheckpointManager(pool, 
                                                    table_name="basic",
                                                    compression_level=6)
        
        # Create test checkpoint
        tree_state = MockTreeState(
            Dict(1 => [2, 3], 2 => [4, 5], 3 => [6, 7]),
            Dict(1 => 0.5, 2 => 0.6, 3 => 0.4),
            Dict(1 => 100, 2 => 50, 3 => 50)
        )
        
        checkpoint = CheckpointManager.Checkpoint(
            100,                        # iteration
            tree_state,                 # tree_state
            [1, 5, 10, 15],            # feature_selection
            [0.9, 0.8, 0.7, 0.6],      # scores
            Dict("algorithm" => "UCB"), # metadata
            now()                       # timestamp
        )
        
        # Save checkpoint
        saved = CheckpointManager.save_checkpoint!(manager, checkpoint, force=true)
        @test saved == true
        
        # Load checkpoint
        loaded = CheckpointManager.load_checkpoint(manager, 100)
        @test !isnothing(loaded)
        @test loaded.iteration == 100
        @test loaded.feature_selection == [1, 5, 10, 15]
        @test loaded.scores == [0.9, 0.8, 0.7, 0.6]
        @test loaded.tree_state.nodes == tree_state.nodes
        
        # Load latest (should be same)
        latest = CheckpointManager.load_checkpoint(manager)
        @test latest.iteration == 100
    end
    
    @testset "Compression" begin
        # Test with compression
        manager_compressed = CheckpointManager.CheckpointManager(pool, 
                                                               table_name="compressed",
                                                               compression_level=9)
        
        # Test without compression
        manager_uncompressed = CheckpointManager.CheckpointManager(pool,
                                                                 table_name="uncompressed",
                                                                 compression_level=0)
        
        # Create large checkpoint
        large_data = Dict(i => rand(1000) for i in 1:100)
        checkpoint = CheckpointManager.Checkpoint(
            1,
            large_data,
            collect(1:50),
            rand(50),
            Dict("size" => "large"),
            now()
        )
        
        # Save with compression
        CheckpointManager.save_checkpoint!(manager_compressed, checkpoint, force=true)
        
        # Save without compression
        CheckpointManager.save_checkpoint!(manager_uncompressed, checkpoint, force=true)
        
        # Compare sizes
        stats_compressed = CheckpointManager.get_checkpoint_stats(manager_compressed)
        stats_uncompressed = CheckpointManager.get_checkpoint_stats(manager_uncompressed)
        
        @test stats_compressed["total_size_mb"] < stats_uncompressed["total_size_mb"]
        @test stats_compressed["avg_compression_ratio"] > 1.0
        @test stats_compressed["space_saved_mb"] > 0
        
        # Verify both can be loaded correctly
        loaded_compressed = CheckpointManager.load_checkpoint(manager_compressed, 1)
        loaded_uncompressed = CheckpointManager.load_checkpoint(manager_uncompressed, 1)
        
        @test loaded_compressed.feature_selection == loaded_uncompressed.feature_selection
    end
    
    @testset "Auto Checkpointing" begin
        manager = CheckpointManager.CheckpointManager(pool,
                                                    table_name="auto",
                                                    auto_checkpoint_interval=10,
                                                    auto_checkpoint_time=0.001)  # 0.06 seconds
        
        # Test iteration-based auto checkpoint
        for i in 1:25
            checkpoint = CheckpointManager.Checkpoint(
                i,
                Dict("iter" => i),
                [i],
                [Float64(i) / 100],
                Dict(),
                now()
            )
            
            CheckpointManager.save_checkpoint!(manager, checkpoint)
        end
        
        # Should have saved at iterations 10 and 20
        checkpoints = CheckpointManager.list_checkpoints(manager)
        saved_iterations = Set(checkpoints.iteration)
        @test 10 in saved_iterations
        @test 20 in saved_iterations
        @test !(5 in saved_iterations)  # Not saved
        
        # Test time-based auto checkpoint
        sleep(0.1)  # Wait for time interval
        
        checkpoint = CheckpointManager.Checkpoint(
            26,
            Dict("time_based" => true),
            [26],
            [0.26],
            Dict(),
            now()
        )
        
        saved = CheckpointManager.save_checkpoint!(manager, checkpoint)
        @test saved == true  # Should save due to time
    end
    
    @testset "Retention Policy" begin
        manager = CheckpointManager.CheckpointManager(pool,
                                                    table_name="retention",
                                                    retention_count=5,
                                                    retention_days=0.0001)  # Very short for testing
        
        # Create many checkpoints
        for i in 1:10
            checkpoint = CheckpointManager.Checkpoint(
                i,
                Dict("i" => i),
                [i],
                [Float64(i)],
                Dict(),
                now() - Minute(10 - i)  # Older checkpoints have earlier timestamps
            )
            
            CheckpointManager.save_checkpoint!(manager, checkpoint, force=true)
        end
        
        # Apply retention policy
        CheckpointManager.apply_retention_policy!(manager)
        
        # Should only keep 5 most recent
        remaining = CheckpointManager.list_checkpoints(manager)
        @test size(remaining, 1) <= 5
        @test all(remaining.iteration .>= 6)  # Should keep iterations 6-10
    end
    
    @testset "Checkpoint Listing" begin
        manager = CheckpointManager.CheckpointManager(pool, table_name="listing")
        
        # Create checkpoints
        for i in [1, 5, 10, 15, 20]
            checkpoint = CheckpointManager.Checkpoint(
                i,
                Dict("data" => "test_$i"),
                collect(1:i),
                rand(i),
                Dict("index" => i),
                now()
            )
            
            CheckpointManager.save_checkpoint!(manager, checkpoint, force=true)
        end
        
        # List all
        all_checkpoints = CheckpointManager.list_checkpoints(manager)
        @test size(all_checkpoints, 1) == 5
        @test issorted(all_checkpoints.iteration, rev=true)
        
        # List limited
        limited = CheckpointManager.list_checkpoints(manager, limit=3)
        @test size(limited, 1) == 3
        @test limited.iteration == [20, 15, 10]
    end
    
    @testset "Checkpoint Statistics" begin
        manager = CheckpointManager.CheckpointManager(pool, table_name="stats")
        
        # Create checkpoints of varying sizes
        for i in 1:5
            data_size = i * 100
            checkpoint = CheckpointManager.Checkpoint(
                i,
                Dict("data" => rand(data_size)),
                collect(1:10),
                rand(10),
                Dict("size" => data_size),
                now()
            )
            
            CheckpointManager.save_checkpoint!(manager, checkpoint, force=true)
        end
        
        # Get statistics
        stats = CheckpointManager.get_checkpoint_stats(manager)
        
        @test stats["total_checkpoints"] == 5
        @test stats["total_size_mb"] > 0
        @test stats["space_saved_mb"] > 0
        @test stats["avg_compression_ratio"] > 1.0
        @test stats["iteration_range"] == (1, 5)
    end
    
    @testset "Export/Import" begin
        manager = CheckpointManager.CheckpointManager(pool, table_name="export")
        
        # Create and save checkpoint
        original = CheckpointManager.Checkpoint(
            999,
            Dict("exported" => true, "data" => rand(100)),
            [1, 2, 3, 4, 5],
            [0.9, 0.8, 0.7, 0.6, 0.5],
            Dict("version" => "1.0"),
            now()
        )
        
        CheckpointManager.save_checkpoint!(manager, original, force=true)
        
        # Export to file
        export_path = tempname()
        filepath, metadata_path = CheckpointManager.export_checkpoint(manager, 999, export_path)
        
        @test isfile(filepath)
        @test isfile(metadata_path)
        
        # Read metadata
        metadata = JSON.parsefile(metadata_path)
        @test metadata["iteration"] == 999
        @test metadata["feature_count"] == 5
        
        # Import to new manager
        manager2 = CheckpointManager.CheckpointManager(pool, table_name="import")
        iteration = CheckpointManager.import_checkpoint!(manager2, filepath)
        
        @test iteration == 999
        
        # Verify imported checkpoint
        imported = CheckpointManager.load_checkpoint(manager2, 999)
        @test imported.feature_selection == original.feature_selection
        @test imported.scores == original.scores
        
        # Cleanup
        rm(filepath, force=true)
        rm(metadata_path, force=true)
    end
    
    @testset "Error Handling" begin
        manager = CheckpointManager.CheckpointManager(pool, table_name="errors")
        
        # Try to load non-existent checkpoint
        result = CheckpointManager.load_checkpoint(manager, 9999)
        @test isnothing(result)
        
        # Try to export non-existent checkpoint
        @test_throws ErrorException CheckpointManager.export_checkpoint(
            manager, 9999, tempname()
        )
    end
    
    @testset "Incremental Checkpoints" begin
        manager = CheckpointManager.CheckpointManager(pool, table_name="incremental")
        
        # Create base checkpoint
        base = CheckpointManager.Checkpoint(
            100,
            Dict("base" => true, "nodes" => 1000),
            collect(1:20),
            rand(20),
            Dict("type" => "full"),
            now()
        )
        
        CheckpointManager.save_checkpoint!(manager, base, force=true)
        
        # Create incremental checkpoint
        incremental = CheckpointManager.Checkpoint(
            110,
            Dict("base" => true, "nodes" => 1100),  # Only 100 new nodes
            collect(1:22),  # 2 new features
            rand(22),
            Dict("type" => "incremental"),
            now()
        )
        
        CheckpointManager.save_incremental_checkpoint!(manager, incremental, 100)
        
        # Verify incremental was saved
        loaded = CheckpointManager.load_checkpoint(manager, 110)
        @test loaded.metadata["is_incremental"] == true
        @test loaded.metadata["base_iteration"] == 100
    end
    
    # Cleanup
    close_pool(pool)
    rm(test_db_path, force=true)
end