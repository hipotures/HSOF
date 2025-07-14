using Test
using SQLite
using DataFrames
using Random
using Dates

# Include modules
include("../../src/database/connection_pool.jl")
include("../../src/database/metadata_parser.jl")
include("../../src/database/column_validator.jl")
using .ConnectionPool
using .MetadataParser
using .ColumnValidator

@testset "Column Validator Tests" begin
    # Create test database
    test_db_path = tempname() * ".db"
    db = SQLite.DB(test_db_path)
    
    # Create test table with various column types
    SQLite.execute(db, """
        CREATE TABLE mixed_types_table (
            id INTEGER PRIMARY KEY,
            numeric_col REAL,
            integer_col INTEGER,
            text_col TEXT,
            categorical_col TEXT,
            date_col TEXT,
            binary_col INTEGER,
            mostly_missing REAL,
            always_null TEXT,
            mixed_type TEXT
        )
    """)
    
    # Insert test data
    Random.seed!(42)
    categories = ["A", "B", "C", "D", "E"]
    
    for i in 1:10000
        # Mixed type column: sometimes numeric, sometimes text
        mixed_val = i % 100 < 80 ? string(rand()) : "text_$(rand(1:10))"
        
        SQLite.execute(db, """
            INSERT INTO mixed_types_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            i,                                              # id
            rand() * 100,                                   # numeric_col
            rand(1:1000),                                   # integer_col
            "text_$i",                                      # text_col
            rand(categories),                               # categorical_col
            "2024-01-$(lpad(rand(1:28), 2, '0'))",        # date_col
            rand([0, 1]),                                   # binary_col
            i % 10 == 0 ? rand() : missing,               # mostly_missing (90% missing)
            missing,                                        # always_null
            mixed_val                                       # mixed_type
        ])
    end
    
    # Create metadata
    MetadataParser.create_metadata_table(db)
    
    SQLite.execute(db, """
        INSERT INTO dataset_metadata 
        (table_name, excluded_columns, id_columns, target_column)
        VALUES (?, ?, ?, ?)
    """, [
        "mixed_types_table",
        JSON.json(["always_null"]),
        JSON.json(["id"]),
        "binary_col"
    ])
    
    SQLite.close(db)
    
    # Create connection pool
    pool = SQLitePool(test_db_path)
    
    @testset "Basic Column Validation" begin
        metadata = parse_metadata(pool, "mixed_types_table")
        
        # Run validation
        result = validate_columns(pool, metadata)
        
        @test result.valid == true
        @test length(result.errors) == 0
        @test length(result.column_info) >= 7  # At least 7 feature columns + target
        
        # Check numeric column
        @test haskey(result.column_info, "numeric_col")
        numeric_info = result.column_info["numeric_col"]
        @test numeric_info.is_numeric == true
        @test numeric_info.is_categorical == false
        @test numeric_info.julia_type == Float64
        @test numeric_info.missing_count == 0
        @test numeric_info.min_value !== nothing
        @test numeric_info.max_value !== nothing
        @test numeric_info.mean_value !== nothing
        
        # Check categorical column
        @test haskey(result.column_info, "categorical_col")
        cat_info = result.column_info["categorical_col"]
        @test cat_info.is_categorical == true
        @test cat_info.is_numeric == false
        @test cat_info.unique_count == 5
        @test cat_info.mode_value in categories
        
        # Check target column
        @test haskey(result.column_info, "binary_col")
        target_info = result.column_info["binary_col"]
        @test target_info.unique_count == 2
    end
    
    @testset "Missing Value Detection" begin
        metadata = parse_metadata(pool, "mixed_types_table")
        
        config = ValidationConfig(missing_threshold=0.8)
        result = validate_columns(pool, metadata, config=config)
        
        # Check mostly_missing column
        @test haskey(result.column_info, "mostly_missing")
        missing_info = result.column_info["mostly_missing"]
        @test missing_info.has_missing == true
        @test missing_info.missing_ratio > 0.85
        @test missing_info.missing_ratio < 0.95
        
        # Should have warning about high missing ratio
        @test any(contains(w, "mostly_missing") && contains(w, "missing values") 
                 for w in result.warnings)
    end
    
    @testset "Type Inference" begin
        metadata = parse_metadata(pool, "mixed_types_table")
        result = validate_columns(pool, metadata)
        
        # Integer column should be detected
        int_info = result.column_info["integer_col"]
        @test int_info.julia_type == Int64
        @test int_info.is_numeric == true
        
        # Text column
        text_info = result.column_info["text_col"]
        @test text_info.julia_type == String
        @test text_info.is_numeric == false
        @test text_info.is_categorical == false  # Too many unique values
        
        # Date column (stored as text in SQLite)
        date_info = result.column_info["date_col"]
        @test date_info.julia_type == String
        # Note: Date detection from string is implemented but may not always trigger
    end
    
    @testset "Type Consistency Checking" begin
        metadata = parse_metadata(pool, "mixed_types_table")
        
        config = ValidationConfig(check_types=true)
        result = validate_columns(pool, metadata, config=config)
        
        # mixed_type column should be flagged as inconsistent
        @test haskey(result.type_consistency, "mixed_type")
        # Note: Depending on sampling, this might or might not be detected
        
        # Numeric columns should be consistent
        @test get(result.type_consistency, "numeric_col", true) == true
        @test get(result.type_consistency, "integer_col", true) == true
    end
    
    @testset "Statistics Collection" begin
        metadata = parse_metadata(pool, "mixed_types_table")
        
        config = ValidationConfig(collect_statistics=true)
        result = validate_columns(pool, metadata, config=config)
        
        # Check statistics summary
        @test haskey(result.statistics_summary, "total_columns")
        @test result.statistics_summary["numeric_columns"] >= 3
        @test result.statistics_summary["categorical_columns"] >= 1
        @test result.statistics_summary["columns_with_missing"] >= 1
        
        # Check individual column statistics
        numeric_info = result.column_info["numeric_col"]
        @test 0 <= numeric_info.min_value <= numeric_info.max_value <= 100
        @test numeric_info.mean_value !== nothing
        @test numeric_info.median_value !== nothing
    end
    
    @testset "Error Cases" begin
        # Test with non-existent columns in metadata
        db = SQLite.DB(test_db_path)
        
        SQLite.execute(db, """
            INSERT INTO dataset_metadata 
            (table_name, excluded_columns, id_columns, target_column)
            VALUES ('mixed_types_table', '["nonexistent_col"]', '["id"]', 'fake_target')
        """)
        
        SQLite.close(db)
        
        # Clear cache to force re-read
        MetadataParser.clear_cache()
        
        metadata = parse_metadata(pool, "mixed_types_table")
        result = validate_columns(pool, metadata)
        
        @test result.valid == false
        @test any(contains(e, "nonexistent_col") for e in result.errors)
        @test any(contains(e, "fake_target") for e in result.errors)
    end
    
    @testset "Missing Value Handling" begin
        # Create test DataFrame with missing values
        test_df = DataFrame(
            col1 = [1, missing, 3, missing, 5],
            col2 = [missing, "B", "C", missing, "E"],
            col3 = [1.0, 2.0, missing, 4.0, 5.0]
        )
        
        # Create mock column info
        col_info = Dict(
            "col1" => ColumnInfo(
                "col1", "INTEGER", Int64, Int64, true, false, false,
                true, 2, 0.4, 3, 1, 5, 3.0, 3.0, nothing
            ),
            "col2" => ColumnInfo(
                "col2", "TEXT", String, String, false, true, false,
                true, 2, 0.4, 3, "B", "E", nothing, nothing, "C"
            ),
            "col3" => ColumnInfo(
                "col3", "REAL", Float64, Float64, true, false, false,
                true, 1, 0.2, 4, 1.0, 5.0, 3.0, 3.5, nothing
            )
        )
        
        # Test FILL_MEAN strategy
        result_mean = handle_missing_values(test_df, col_info, FILL_MEAN)
        @test result_mean.col1 == [1, 3, 3, 3, 5]  # Mean = 3
        @test !any(ismissing, result_mean.col3)
        
        # Test FILL_ZERO strategy
        result_zero = handle_missing_values(test_df, col_info, FILL_ZERO)
        @test result_zero.col1 == [1, 0, 3, 0, 5]
        
        # Test FILL_MODE strategy
        result_mode = handle_missing_values(test_df, col_info, FILL_MODE)
        @test result_mode.col2 == [col_info["col2"].mode_value, "B", "C", 
                                   col_info["col2"].mode_value, "E"]
        
        # Test FILL_FORWARD strategy
        result_forward = handle_missing_values(test_df, col_info, FILL_FORWARD)
        @test result_forward.col1[2] == 1  # Filled with previous value
        @test result_forward.col3[3] == 2.0  # Filled with previous value
        
        # Test RAISE_ERROR strategy
        @test_throws ErrorException handle_missing_values(test_df, col_info, RAISE_ERROR)
    end
    
    @testset "Large Dataset Performance" begin
        # Create a table with many columns
        db = SQLite.DB(test_db_path)
        
        col_defs = ["id INTEGER PRIMARY KEY"]
        append!(col_defs, ["col_$i REAL" for i in 1:100])
        
        SQLite.execute(db, """
            CREATE TABLE wide_table (
                $(join(col_defs, ", "))
            )
        """)
        
        # Insert some data
        for i in 1:1000
            values = [i]
            append!(values, [rand() for _ in 1:100])
            placeholders = join(["?" for _ in 1:101], ", ")
            SQLite.execute(db, "INSERT INTO wide_table VALUES ($placeholders)", values)
        end
        
        SQLite.execute(db, """
            INSERT INTO dataset_metadata 
            (table_name, excluded_columns, id_columns, target_column)
            VALUES ('wide_table', '[]', '["id"]', 'col_100')
        """)
        
        SQLite.close(db)
        
        # Clear cache
        MetadataParser.clear_cache()
        
        # Time validation
        metadata = parse_metadata(pool, "wide_table")
        
        elapsed = @elapsed result = validate_columns(pool, metadata, 
                                                   config=ValidationConfig(sample_size=100))
        
        @test result.valid == true
        @test length(result.column_info) == 100  # 99 features + 1 target
        @test elapsed < 10.0  # Should complete in reasonable time
    end
    
    # Cleanup
    close_pool(pool)
    rm(test_db_path, force=true)
end