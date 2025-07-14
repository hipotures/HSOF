using Test
using SQLite
using DataFrames
using JSON
using Dates

# Include modules
include("../../src/database/connection_pool.jl")
include("../../src/database/metadata_parser.jl")
using .ConnectionPool
using .MetadataParser

@testset "Metadata Parser Tests" begin
    # Create test database
    test_db_path = tempname() * ".db"
    db = SQLite.DB(test_db_path)
    
    # Create test tables
    SQLite.execute(db, """
        CREATE TABLE test_features (
            id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            feature_1 REAL,
            feature_2 REAL,
            feature_3 REAL,
            feature_4 REAL,
            feature_5 REAL,
            target INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    
    # Insert test data
    for i in 1:1000
        SQLite.execute(db, """
            INSERT INTO test_features 
            (patient_id, feature_1, feature_2, feature_3, feature_4, feature_5, target, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        """, [i, rand(), rand(), rand(), rand(), rand(), rand() > 0.5 ? 1 : 0])
    end
    
    # Create metadata table
    MetadataParser.create_metadata_table(db)
    
    # Insert test metadata
    SQLite.execute(db, """
        INSERT INTO dataset_metadata 
        (table_name, excluded_columns, id_columns, target_column, metadata_version, additional_info)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        "test_features",
        JSON.json(["created_at", "updated_at"]),
        JSON.json(["id", "patient_id"]),
        "target",
        "1.0",
        JSON.json(Dict("description" => "Test dataset", "source" => "synthetic"))
    ])
    
    SQLite.close(db)
    
    # Create connection pool
    pool = SQLitePool(test_db_path)
    
    @testset "Metadata Parsing" begin
        # Test successful parsing
        metadata = parse_metadata(pool, "test_features")
        
        @test metadata.table_name == "test_features"
        @test metadata.excluded_columns == ["created_at", "updated_at"]
        @test metadata.id_columns == ["id", "patient_id"]
        @test metadata.target_column == "target"
        @test metadata.feature_count == 5  # feature_1 through feature_5
        @test metadata.row_count == 1000
        @test metadata.metadata_version == "1.0"
        @test metadata.additional_info["description"] == "Test dataset"
        @test metadata.additional_info["source"] == "synthetic"
        
        # Test error for non-existent table
        @test_throws ErrorException parse_metadata(pool, "nonexistent_table")
    end
    
    @testset "Metadata Validation" begin
        metadata = parse_metadata(pool, "test_features")
        
        with_connection(pool) do conn
            # Should validate successfully
            @test MetadataParser.validate_metadata(metadata, conn.db) == true
            
            # Test with invalid metadata
            invalid_metadata = DatasetMetadata(
                table_name = "test_features",
                excluded_columns = ["nonexistent_column"],
                id_columns = ["id"],
                target_column = "target"
            )
            
            @test_throws ErrorException MetadataParser.validate_metadata(invalid_metadata, conn.db)
        end
    end
    
    @testset "Feature Column Extraction" begin
        metadata = parse_metadata(pool, "test_features")
        
        with_connection(pool) do conn
            feature_cols = MetadataParser.get_feature_columns(metadata, conn.db)
            
            @test length(feature_cols) == 5
            @test "feature_1" in feature_cols
            @test "feature_2" in feature_cols
            @test "feature_3" in feature_cols
            @test "feature_4" in feature_cols
            @test "feature_5" in feature_cols
            
            # Excluded columns should not be in features
            @test !("id" in feature_cols)
            @test !("patient_id" in feature_cols)
            @test !("target" in feature_cols)
            @test !("created_at" in feature_cols)
            @test !("updated_at" in feature_cols)
        end
    end
    
    @testset "Metadata Caching" begin
        # Clear cache first
        MetadataParser.clear_cache()
        
        # First call should parse from database
        t1 = @elapsed metadata1 = parse_metadata(pool, "test_features")
        
        # Second call should be from cache (much faster)
        t2 = @elapsed metadata2 = parse_metadata(pool, "test_features")
        
        @test t2 < t1 / 10  # Cache should be at least 10x faster
        @test metadata1 == metadata2
        
        # Test cache clearing
        MetadataParser.clear_cache("test_features")
        
        # Should parse from database again
        t3 = @elapsed metadata3 = parse_metadata(pool, "test_features")
        @test t3 > t2  # Should be slower than cached version
        
        # Test clearing all cache
        MetadataParser.clear_cache()
        cached = MetadataParser.get_cached_metadata("test_features")
        @test isnothing(cached)
    end
    
    @testset "Edge Cases" begin
        # Test table with no metadata
        db = SQLite.DB(test_db_path)
        
        # Create table without metadata entry
        SQLite.execute(db, """
            CREATE TABLE no_metadata_table (
                id INTEGER PRIMARY KEY,
                value REAL
            )
        """)
        
        SQLite.close(db)
        
        @test_throws ErrorException parse_metadata(pool, "no_metadata_table")
        
        # Test with empty JSON arrays
        db = SQLite.DB(test_db_path)
        
        SQLite.execute(db, """
            INSERT INTO dataset_metadata 
            (table_name, excluded_columns, id_columns, target_column)
            VALUES ('no_metadata_table', '[]', '[]', NULL)
        """)
        
        SQLite.close(db)
        
        metadata = parse_metadata(pool, "no_metadata_table")
        @test isempty(metadata.excluded_columns)
        @test isempty(metadata.id_columns)
        @test isnothing(metadata.target_column)
        @test metadata.feature_count == 1  # Only 'value' column
    end
    
    @testset "Metadata Insertion" begin
        db = SQLite.DB(test_db_path)
        
        # Create new metadata
        new_metadata = DatasetMetadata(
            table_name = "new_dataset",
            excluded_columns = ["timestamp"],
            id_columns = ["record_id"],
            target_column = "outcome",
            metadata_version = "2.0",
            additional_info = Dict("notes" => "Test insertion")
        )
        
        MetadataParser.insert_metadata(db, new_metadata)
        
        # Verify insertion
        result = DBInterface.execute(db, 
            "SELECT * FROM dataset_metadata WHERE table_name = 'new_dataset'") |> DataFrame
        
        @test size(result, 1) == 1
        @test JSON.parse(result.excluded_columns[1]) == ["timestamp"]
        @test JSON.parse(result.id_columns[1]) == ["record_id"]
        @test result.target_column[1] == "outcome"
        @test result.metadata_version[1] == "2.0"
        
        SQLite.close(db)
    end
    
    @testset "Large Table Handling" begin
        # Test with a larger table to verify row count estimation
        db = SQLite.DB(test_db_path)
        
        SQLite.execute(db, """
            CREATE TABLE large_table (
                id INTEGER PRIMARY KEY,
                $(join(["col_$i REAL" for i in 1:100], ", "))
            )
        """)
        
        # Insert metadata
        SQLite.execute(db, """
            INSERT INTO dataset_metadata 
            (table_name, excluded_columns, id_columns, target_column)
            VALUES ('large_table', '[]', '["id"]', 'col_100')
        """)
        
        SQLite.close(db)
        
        metadata = parse_metadata(pool, "large_table")
        @test metadata.feature_count == 99  # 100 columns - 1 id - 1 target
        @test metadata.table_name == "large_table"
    end
    
    @testset "Concurrent Access" begin
        # Test concurrent metadata parsing
        tasks = []
        results = Channel{DatasetMetadata}(10)
        
        for i in 1:10
            task = Threads.@spawn begin
                metadata = parse_metadata(pool, "test_features")
                put!(results, metadata)
            end
            push!(tasks, task)
        end
        
        # Wait for all tasks
        for task in tasks
            wait(task)
        end
        
        close(results)
        
        # All results should be identical
        all_metadata = collect(results)
        @test length(all_metadata) == 10
        @test all(m -> m == all_metadata[1], all_metadata)
    end
    
    # Cleanup
    close_pool(pool)
    rm(test_db_path, force=true)
end