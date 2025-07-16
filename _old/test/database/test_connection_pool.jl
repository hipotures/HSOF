using Test
using SQLite
using DataFrames
using Random

# Include the connection pool module
include("../../src/database/connection_pool.jl")
using .ConnectionPool

@testset "SQLite Connection Pool Tests" begin
    # Create a test database
    test_db_path = tempname() * ".db"
    
    # Initialize test database
    db = SQLite.DB(test_db_path)
    SQLite.execute(db, """
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            value REAL,
            name TEXT
        )
    """)
    
    # Insert test data
    for i in 1:1000
        SQLite.execute(db, "INSERT INTO test_table (value, name) VALUES (?, ?)", 
                      [rand(), "test_$i"])
    end
    SQLite.close(db)
    
    @testset "Pool Creation" begin
        # Test basic creation
        pool = SQLitePool(test_db_path)
        @test !isnothing(pool)
        @test pool.min_size == 5
        @test pool.max_size == 20
        @test length(pool.connections) == 5
        close_pool(pool)
        
        # Test custom parameters
        pool = SQLitePool(test_db_path, 
                         min_size=2, 
                         max_size=10,
                         timeout_seconds=5.0)
        @test pool.min_size == 2
        @test pool.max_size == 10
        @test length(pool.connections) == 2
        close_pool(pool)
        
        # Test invalid database path
        @test_throws ArgumentError SQLitePool("nonexistent.db")
        
        # Test invalid pool sizes
        @test_throws ArgumentError SQLitePool(test_db_path, min_size=0)
        @test_throws ArgumentError SQLitePool(test_db_path, min_size=10, max_size=5)
    end
    
    @testset "Connection Acquisition" begin
        pool = SQLitePool(test_db_path, min_size=2, max_size=5)
        
        # Test basic acquisition
        conn = acquire_connection(pool)
        @test !isnothing(conn)
        @test conn.in_use == true
        @test conn.use_count == 1
        
        # Test read-only mode
        @test_throws SQLite.SQLiteException SQLite.execute(conn.db, 
            "INSERT INTO test_table (value, name) VALUES (1.0, 'test')")
        
        # Release connection
        release_connection(pool, conn)
        @test conn.in_use == false
        
        # Test multiple acquisitions
        conns = []
        for i in 1:5
            push!(conns, acquire_connection(pool))
        end
        
        stats = pool_stats(pool)
        @test stats["total_connections"] == 5
        @test stats["in_use"] == 5
        @test stats["available"] == 0
        
        # Release all
        for conn in conns
            release_connection(pool, conn)
        end
        
        stats = pool_stats(pool)
        @test stats["available"] == 5
        
        close_pool(pool)
    end
    
    @testset "Connection Recycling" begin
        pool = SQLitePool(test_db_path, 
                         min_size=1, 
                         max_size=3,
                         max_uses=5,
                         max_lifetime_hours=0.0001)  # Very short lifetime for testing
        
        # Use connection beyond max_uses
        conn = acquire_connection(pool)
        conn_id = conn.id
        
        for i in 1:5
            release_connection(pool, conn)
            conn = acquire_connection(pool)
        end
        
        # Should have been recycled
        @test conn.use_count == 1
        @test conn.id == conn_id  # Same slot, new connection
        
        release_connection(pool, conn)
        
        # Test lifetime recycling
        sleep(0.5)  # Ensure lifetime exceeded
        conn = acquire_connection(pool)
        @test conn.use_count == 1
        
        release_connection(pool, conn)
        close_pool(pool)
    end
    
    @testset "Health Checking" begin
        pool = SQLitePool(test_db_path, min_size=1, max_size=2)
        
        conn = acquire_connection(pool)
        
        # Verify healthy connection works
        result = DBInterface.execute(conn.db, "SELECT COUNT(*) as count FROM test_table") |> DataFrame
        @test result.count[1] == 1000
        
        release_connection(pool, conn)
        close_pool(pool)
    end
    
    @testset "Concurrent Access" begin
        pool = SQLitePool(test_db_path, min_size=5, max_size=10)
        
        # Test concurrent acquisition and release
        tasks = []
        results = Channel{Bool}(100)
        
        for i in 1:20
            task = Threads.@spawn begin
                try
                    conn = acquire_connection(pool)
                    
                    # Perform query
                    result = DBInterface.execute(conn.db, 
                        "SELECT COUNT(*) as count FROM test_table") |> DataFrame
                    
                    # Simulate work
                    sleep(0.1 * rand())
                    
                    release_connection(pool, conn)
                    put!(results, result.count[1] == 1000)
                catch e
                    @error "Task error" exception=e
                    put!(results, false)
                end
            end
            push!(tasks, task)
        end
        
        # Wait for all tasks
        for task in tasks
            wait(task)
        end
        
        # Check results
        close(results)
        all_succeeded = all(collect(results))
        @test all_succeeded
        
        # Verify pool state
        stats = pool_stats(pool)
        @test stats["available"] + stats["in_use"] == stats["total_connections"]
        @test stats["total_connections"] <= 10
        
        close_pool(pool)
    end
    
    @testset "Timeout Handling" begin
        pool = SQLitePool(test_db_path, 
                         min_size=1, 
                         max_size=1,
                         timeout_seconds=1.0)
        
        # Acquire the only connection
        conn1 = acquire_connection(pool)
        
        # Try to acquire another (should timeout)
        @test_throws ErrorException acquire_connection(pool)
        
        release_connection(pool, conn1)
        close_pool(pool)
    end
    
    @testset "with_connection Helper" begin
        pool = SQLitePool(test_db_path, min_size=2, max_size=5)
        
        # Test basic usage
        result = with_connection(pool) do conn
            DBInterface.execute(conn.db, "SELECT COUNT(*) as count FROM test_table") |> DataFrame
        end
        @test result.count[1] == 1000
        
        # Test exception handling
        @test_throws SQLite.SQLiteException with_connection(pool) do conn
            SQLite.execute(conn.db, "INSERT INTO test_table VALUES (1, 2, 3)")
        end
        
        # Verify connection was released even after exception
        stats = pool_stats(pool)
        @test stats["available"] == stats["total_connections"]
        
        close_pool(pool)
    end
    
    @testset "Pool Closure" begin
        pool = SQLitePool(test_db_path, min_size=2, max_size=5)
        
        # Close pool
        close_pool(pool)
        @test pool.closed == true
        @test isempty(pool.connections)
        
        # Verify cannot acquire after closure
        @test_throws ErrorException acquire_connection(pool)
    end
    
    # Cleanup
    rm(test_db_path, force=true)
end