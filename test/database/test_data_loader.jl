using Test
using SQLite
using DataFrames
using Statistics
using Random

# Include modules
include("../../src/database/connection_pool.jl")
include("../../src/database/metadata_parser.jl")
include("../../src/database/data_loader.jl")
using .ConnectionPool
using .MetadataParser
using .DataLoader

@testset "Data Loader Tests" begin
    # Create test database
    test_db_path = tempname() * ".db"
    db = SQLite.DB(test_db_path)
    
    # Create test table with many rows
    SQLite.execute(db, """
        CREATE TABLE large_features (
            id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            $(join(["feature_$i REAL" for i in 1:50], ", ")),
            target INTEGER,
            created_at TEXT
        )
    """)
    
    # Insert test data
    n_rows = 50_000
    batch_size = 1000
    
    for batch_start in 1:batch_size:n_rows
        batch_end = min(batch_start + batch_size - 1, n_rows)
        
        values = []
        for i in batch_start:batch_end
            row_values = [i, i]  # id, patient_id
            append!(row_values, [rand() for _ in 1:50])  # features
            push!(row_values, rand() > 0.5 ? 1 : 0)  # target
            push!(row_values, "2024-01-01")  # created_at
            push!(values, row_values)
        end
        
        # Batch insert
        placeholders = "(" * join(["?" for _ in 1:54], ", ") * ")"
        values_clause = join([placeholders for _ in 1:length(values)], ", ")
        
        flat_values = vcat(values...)
        
        SQLite.execute(db, """
            INSERT INTO large_features VALUES $values_clause
        """, flat_values)
    end
    
    # Create metadata
    MetadataParser.create_metadata_table(db)
    
    SQLite.execute(db, """
        INSERT INTO dataset_metadata 
        (table_name, excluded_columns, id_columns, target_column)
        VALUES (?, ?, ?, ?)
    """, [
        "large_features",
        JSON.json(["created_at"]),
        JSON.json(["id", "patient_id"]),
        "target"
    ])
    
    SQLite.close(db)
    
    # Create connection pool
    pool = SQLitePool(test_db_path)
    
    @testset "Chunk Iterator Creation" begin
        # Test basic creation
        iterator = create_chunk_iterator(pool, "large_features")
        
        @test iterator.total_rows == n_rows
        @test iterator.total_chunks > 0
        @test length(iterator.feature_columns) == 50
        @test iterator.current_chunk == 0
        @test !iterator.closed
        
        close(iterator)
        
        # Test with custom chunk size
        iterator = create_chunk_iterator(pool, "large_features", chunk_size=5000)
        @test iterator.total_chunks == cld(n_rows, 5000)
        close(iterator)
        
        # Test with WHERE clause
        iterator = create_chunk_iterator(pool, "large_features", 
                                       where_clause="target = 1")
        @test iterator.total_rows < n_rows
        close(iterator)
    end
    
    @testset "Adaptive Chunk Sizing" begin
        # Test adaptive sizing calculation
        chunk_size = adaptive_chunk_size(50, 100_000, 8.0)
        @test 1000 <= chunk_size <= 100_000
        
        # Test with small dataset
        chunk_size = adaptive_chunk_size(50, 1000, 8.0)
        @test chunk_size == 1000  # Should not exceed total rows
        
        # Test memory estimation
        mem_gb = estimate_memory_usage(10_000, 50)
        @test 0.001 < mem_gb < 0.1  # Reasonable range
    end
    
    @testset "Chunk Iteration" begin
        iterator = create_chunk_iterator(pool, "large_features", chunk_size=10_000)
        
        chunks_processed = 0
        total_rows_seen = 0
        
        for chunk in iterator
            @test chunk isa DataChunk
            @test size(chunk.data, 2) == 54  # 2 ids + 50 features + 1 target + 1 created_at
            @test chunk.chunk_index == chunks_processed + 1
            @test chunk.total_chunks == iterator.total_chunks
            
            # Verify columns
            @test "id" in names(chunk.data)
            @test "patient_id" in names(chunk.data)
            @test "target" in names(chunk.data)
            @test "feature_1" in names(chunk.data)
            @test "feature_50" in names(chunk.data)
            
            chunks_processed += 1
            total_rows_seen += size(chunk.data, 1)
            
            # Check last chunk flag
            if chunk.chunk_index == chunk.total_chunks
                @test chunk.is_last
            else
                @test !chunk.is_last
            end
        end
        
        @test chunks_processed == iterator.total_chunks
        @test total_rows_seen == n_rows
        @test iterator.closed
    end
    
    @testset "Streaming with Processing" begin
        # Test stream_features helper
        results = stream_features(pool, "large_features", chunk_size=10_000) do chunk
            # Calculate mean of features for each chunk
            feature_cols = filter(name -> startswith(name, "feature_"), names(chunk.data))
            feature_data = Matrix(chunk.data[:, feature_cols])
            return vec(mean(feature_data, dims=1))
        end
        
        @test length(results) == cld(n_rows, 10_000)
        @test all(r -> length(r) == 50, results)  # 50 features
        
        # Test with early termination
        chunks_processed = 0
        iterator = create_chunk_iterator(pool, "large_features", chunk_size=10_000)
        
        for chunk in iterator
            chunks_processed += 1
            if chunks_processed >= 3
                close(iterator)
                break
            end
        end
        
        @test chunks_processed == 3
    end
    
    @testset "Memory Management" begin
        # Test memory monitoring
        monitor = DataLoader.MemoryMonitor(16.0)
        @test DataLoader.check_memory(monitor)  # Should pass with high limit
        
        # Test with very small memory limit
        monitor_small = DataLoader.MemoryMonitor(0.001)
        # May or may not trigger depending on system
        DataLoader.check_memory(monitor_small)
    end
    
    @testset "Prefetching" begin
        # Test that prefetching works
        iterator = create_chunk_iterator(pool, "large_features", 
                                       chunk_size=5000,
                                       prefetch_chunks=3)
        
        # First chunk should be available immediately
        chunk1 = first(iterator)
        @test chunk1.chunk_index == 1
        
        # Check that prefetch buffer has data
        @test iterator.prefetch_buffer.n_avail > 0
        
        close(iterator)
    end
    
    @testset "Data Integrity" begin
        # Collect all chunks
        iterator = create_chunk_iterator(pool, "large_features", 
                                       chunk_size=10_000,
                                       order_by="id")
        
        all_chunks = DataChunk[]
        for chunk in iterator
            push!(all_chunks, chunk)
        end
        
        # Validate chunk integrity
        @test DataLoader.validate_chunk_integrity(all_chunks)
        
        # Check no data loss or duplication
        all_ids = Int[]
        for chunk in all_chunks
            append!(all_ids, chunk.data.id)
        end
        
        @test length(all_ids) == n_rows
        @test all_ids == 1:n_rows  # Should be in order
    end
    
    @testset "Error Handling" begin
        # Test with non-existent table
        @test_throws ErrorException create_chunk_iterator(pool, "nonexistent_table")
        
        # Test iterator after closing
        iterator = create_chunk_iterator(pool, "large_features", chunk_size=10_000)
        close(iterator)
        
        # Should return nothing when iterating closed iterator
        @test isnothing(iterate(iterator))
    end
    
    @testset "Custom Filtering" begin
        # Test with WHERE clause filtering
        iterator = create_chunk_iterator(pool, "large_features",
                                       where_clause="feature_1 > 0.5",
                                       chunk_size=5000)
        
        total_filtered = 0
        for chunk in iterator
            # Verify all rows match filter
            @test all(chunk.data.feature_1 .> 0.5)
            total_filtered += size(chunk.data, 1)
        end
        
        @test total_filtered < n_rows  # Should have filtered some rows
        @test total_filtered > 0  # Should have some results
    end
    
    @testset "Concurrent Usage" begin
        # Test multiple iterators simultaneously
        iterators = []
        
        for i in 1:3
            iterator = create_chunk_iterator(pool, "large_features",
                                           chunk_size=10_000)
            push!(iterators, iterator)
        end
        
        # Process first chunk from each
        results = []
        for iterator in iterators
            chunk = first(iterator)
            push!(results, size(chunk.data, 1))
        end
        
        @test all(r -> r > 0, results)
        
        # Clean up
        for iterator in iterators
            close(iterator)
        end
    end
    
    # Cleanup
    close_pool(pool)
    rm(test_db_path, force=true)
end