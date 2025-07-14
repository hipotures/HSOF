#!/usr/bin/env julia

# Database Performance Benchmarks
# Run with: julia benchmarks/database_benchmarks.jl

using Pkg
Pkg.activate(dirname(@__DIR__))

using BenchmarkTools
using SQLite
using DataFrames
using Random
using Dates
using Statistics
using Printf

# Include database module
include("../src/database/Database.jl")
using .Database

# Benchmark configuration
const BENCHMARK_DB = tempname() * "_bench.db"
const SAMPLE_SIZES = [1_000, 10_000, 100_000]
const FEATURE_COUNTS = [100, 500, 1000]
const CHUNK_SIZES = [1_000, 5_000, 10_000, 50_000]

# Results storage
benchmark_results = DataFrame()

"""
Create benchmark database with specified parameters
"""
function create_benchmark_db(n_samples::Int, n_features::Int)
    db = SQLite.DB(BENCHMARK_DB)
    
    # Create table
    columns = ["id INTEGER PRIMARY KEY", "row_id INTEGER"]
    append!(columns, ["feature_$i REAL" for i in 1:n_features])
    push!(columns, "target INTEGER")
    
    SQLite.execute(db, """
        CREATE TABLE benchmark_data (
            $(join(columns, ", "))
        )
    """)
    
    # Insert data in batches
    batch_size = min(1000, n_samples)
    for batch_start in 1:batch_size:n_samples
        batch_end = min(batch_start + batch_size - 1, n_samples)
        
        for i in batch_start:batch_end
            values = [i, i]
            append!(values, rand(n_features))
            push!(values, rand() > 0.5 ? 1 : 0)
            
            placeholders = join(["?" for _ in 1:(n_features + 3)], ", ")
            SQLite.execute(db, "INSERT INTO benchmark_data VALUES ($placeholders)", values)
        end
    end
    
    # Create metadata
    Database.create_metadata_table(db)
    Database.insert_metadata(db, Database.DatasetMetadata(
        table_name = "benchmark_data",
        excluded_columns = String[],
        id_columns = ["id", "row_id"],
        target_column = "target",
        feature_count = n_features,
        row_count = n_samples
    ))
    
    SQLite.close(db)
end

"""
Benchmark data loading performance
"""
function benchmark_data_loading()
    println("\n" * "="^60)
    println("DATA LOADING BENCHMARKS")
    println("="^60)
    
    for n_samples in SAMPLE_SIZES
        for n_features in [100, 500]  # Limit features for loading tests
            # Create test database
            rm(BENCHMARK_DB, force=true)
            create_benchmark_db(n_samples, n_features)
            
            pool = Database.create_database_connection(BENCHMARK_DB)
            
            for chunk_size in CHUNK_SIZES
                if chunk_size > n_samples
                    continue
                end
                
                # Benchmark loading
                bench = @benchmark begin
                    iterator = Database.create_chunk_iterator(
                        $pool, "benchmark_data",
                        chunk_size = $chunk_size,
                        show_progress = false
                    )
                    
                    total = 0
                    for chunk in iterator
                        total += size(chunk.data, 1)
                    end
                    
                    total
                end samples=3 seconds=10
                
                # Calculate metrics
                load_time = median(bench).time / 1e9  # Convert to seconds
                throughput_rows = n_samples / load_time
                throughput_mb = (n_samples * n_features * 8) / 1e6 / load_time
                
                # Store results
                push!(benchmark_results, (
                    test = "data_loading",
                    n_samples = n_samples,
                    n_features = n_features,
                    chunk_size = chunk_size,
                    time_seconds = load_time,
                    rows_per_second = throughput_rows,
                    mb_per_second = throughput_mb,
                    memory_mb = bench.memory / 1e6
                ))
                
                @printf("Samples: %d, Features: %d, Chunk: %d => %.1f rows/s, %.1f MB/s\n",
                       n_samples, n_features, chunk_size, throughput_rows, throughput_mb)
            end
            
            Database.close_pool(pool)
        end
    end
end

"""
Benchmark result writing performance
"""
function benchmark_result_writing()
    println("\n" * "="^60)
    println("RESULT WRITING BENCHMARKS")
    println("="^60)
    
    # Use single database for writing tests
    rm(BENCHMARK_DB, force=true)
    create_benchmark_db(1000, 100)  # Small DB is fine for writing
    
    pool = Database.create_database_connection(BENCHMARK_DB)
    
    for batch_size in [100, 500, 1000, 5000]
        for async_mode in [false, true]
            writer = Database.ResultWriter(pool,
                                         table_prefix = "bench_$(batch_size)",
                                         batch_size = batch_size,
                                         async_writing = async_mode)
            
            # Generate test results
            results = []
            for i in 1:10_000
                push!(results, Database.MCTSResult(
                    i,
                    sort(randperm(100)[1:rand(5:20)]),
                    rand(rand(5:20)),
                    rand(),
                    Dict("test" => true),
                    now()
                ))
            end
            
            # Benchmark writing
            bench = @benchmark begin
                for r in $results
                    Database.write_results!($writer, r)
                end
                close($writer)
            end samples=3 seconds=10
            
            write_time = median(bench).time / 1e9
            throughput = length(results) / write_time
            
            push!(benchmark_results, (
                test = "result_writing",
                n_samples = length(results),
                n_features = 0,
                chunk_size = batch_size,
                async = async_mode,
                time_seconds = write_time,
                results_per_second = throughput,
                memory_mb = bench.memory / 1e6
            ))
            
            @printf("Batch: %d, Async: %s => %.1f results/s\n",
                   batch_size, async_mode, throughput)
        end
    end
    
    Database.close_pool(pool)
end

"""
Benchmark checkpoint operations
"""
function benchmark_checkpoints()
    println("\n" * "="^60)
    println("CHECKPOINT BENCHMARKS")
    println("="^60)
    
    pool = Database.create_database_connection(BENCHMARK_DB)
    
    # Test different compression levels
    for compression_level in [0, 3, 6, 9]
        manager = Database.CheckpointManager(pool,
                                           table_name = "bench_comp_$compression_level",
                                           compression_level = compression_level)
        
        # Create test data of varying sizes
        for data_size_mb in [1, 10, 50]
            # Generate data
            n_elements = div(data_size_mb * 1024 * 1024, 8)  # 8 bytes per Float64
            test_data = Dict(
                "array" => rand(n_elements),
                "metadata" => Dict("size" => data_size_mb)
            )
            
            checkpoint = Database.Checkpoint(
                1,
                test_data,
                collect(1:100),
                rand(100),
                Dict("test" => true),
                now()
            )
            
            # Benchmark save
            save_bench = @benchmark begin
                Database.save_checkpoint!($manager, $checkpoint, force=true)
            end samples=3 seconds=10
            
            # Benchmark load
            load_bench = @benchmark begin
                Database.load_checkpoint($manager, 1)
            end samples=3 seconds=10
            
            # Get compression stats
            stats = Database.get_checkpoint_stats(manager)
            
            save_time = median(save_bench).time / 1e6  # ms
            load_time = median(load_bench).time / 1e6  # ms
            
            push!(benchmark_results, (
                test = "checkpoint_save",
                compression_level = compression_level,
                data_size_mb = data_size_mb,
                time_ms = save_time,
                compression_ratio = stats["avg_compression_ratio"],
                compressed_size_mb = stats["total_size_mb"]
            ))
            
            push!(benchmark_results, (
                test = "checkpoint_load",
                compression_level = compression_level,
                data_size_mb = data_size_mb,
                time_ms = load_time
            ))
            
            @printf("Compression: %d, Size: %d MB => Save: %.1f ms, Load: %.1f ms, Ratio: %.2fx\n",
                   compression_level, data_size_mb, save_time, load_time,
                   stats["avg_compression_ratio"])
        end
    end
    
    Database.close_pool(pool)
end

"""
Benchmark concurrent operations
"""
function benchmark_concurrent()
    println("\n" * "="^60)
    println("CONCURRENT OPERATIONS BENCHMARKS")
    println("="^60)
    
    # Create larger test database
    rm(BENCHMARK_DB, force=true)
    create_benchmark_db(100_000, 100)
    
    for n_threads in [1, 2, 4, 8]
        if n_threads > Threads.nthreads()
            continue
        end
        
        pool = Database.create_database_connection(BENCHMARK_DB, 
                                                 max_size = n_threads * 2)
        
        # Benchmark concurrent reads
        bench = @benchmark begin
            tasks = []
            results = Channel{Int}(n_threads)
            
            for i in 1:n_threads
                task = Threads.@spawn begin
                    iterator = Database.create_chunk_iterator(
                        pool, "benchmark_data",
                        chunk_size = 10_000,
                        show_progress = false
                    )
                    
                    count = 0
                    for chunk in iterator
                        count += size(chunk.data, 1)
                        if count >= 50_000  # Process partial data
                            break
                        end
                    end
                    
                    put!(results, count)
                    close(iterator)
                end
                push!(tasks, task)
            end
            
            # Wait for all
            for task in tasks
                wait(task)
            end
            
            close(results)
            sum(collect(results))
        end samples=3 seconds=10
        
        time_seconds = median(bench).time / 1e9
        total_rows = n_threads * 50_000
        throughput = total_rows / time_seconds
        
        push!(benchmark_results, (
            test = "concurrent_reads",
            n_threads = n_threads,
            time_seconds = time_seconds,
            total_rows = total_rows,
            rows_per_second = throughput,
            speedup = throughput / (50_000 / time_seconds)  # vs single thread
        ))
        
        @printf("Threads: %d => %.1f rows/s (%.2fx speedup)\n",
               n_threads, throughput, throughput / (50_000 / time_seconds))
        
        Database.close_pool(pool)
    end
end

"""
Generate benchmark report
"""
function generate_report()
    println("\n" * "="^80)
    println("HSOF DATABASE BENCHMARK REPORT")
    println("="^80)
    println("Date: $(now())")
    println("Julia: $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("="^80)
    
    # Group results by test type
    for test_type in unique(benchmark_results.test)
        println("\n## $test_type")
        println("-"^60)
        
        test_results = filter(r -> r.test == test_type, benchmark_results)
        
        # Display based on test type
        if test_type == "data_loading"
            sort!(test_results, [:n_samples, :chunk_size])
            println("Samples | Features | Chunk | Rows/s | MB/s | Memory")
            println("-"^60)
            
            for r in eachrow(test_results)
                @printf("%7d | %8d | %5d | %6.0f | %4.1f | %5.1f MB\n",
                       r.n_samples, r.n_features, r.chunk_size,
                       r.rows_per_second, r.mb_per_second, r.memory_mb)
            end
            
        elseif test_type == "result_writing"
            println("Batch Size | Async | Results/s | Memory")
            println("-"^60)
            
            for r in eachrow(test_results)
                @printf("%10d | %5s | %9.0f | %5.1f MB\n",
                       r.chunk_size, r.async ? "Yes" : "No",
                       r.results_per_second, r.memory_mb)
            end
            
        elseif test_type == "checkpoint_save"
            println("Compression | Size | Time | Ratio | Compressed")
            println("-"^60)
            
            for r in eachrow(test_results)
                @printf("%11d | %4d MB | %6.1f ms | %5.2fx | %7.1f MB\n",
                       r.compression_level, r.data_size_mb, r.time_ms,
                       r.compression_ratio, r.compressed_size_mb)
            end
            
        elseif test_type == "concurrent_reads"
            println("Threads | Time | Rows/s | Speedup")
            println("-"^60)
            
            for r in eachrow(test_results)
                @printf("%7d | %5.2fs | %6.0f | %6.2fx\n",
                       r.n_threads, r.time_seconds, r.rows_per_second, r.speedup)
            end
        end
    end
    
    println("\n" * "="^80)
    
    # Save results
    results_file = "benchmarks/database_benchmark_results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"
    mkpath(dirname(results_file))
    CSV.write(results_file, benchmark_results)
    println("\nResults saved to: $results_file")
end

# Main execution
function main()
    println("🚀 Starting HSOF Database Benchmarks...")
    println("This may take several minutes...")
    
    try
        benchmark_data_loading()
        benchmark_result_writing()
        benchmark_checkpoints()
        benchmark_concurrent()
        
        generate_report()
        
        println("\n✅ Benchmarks completed successfully!")
    catch e
        @error "Benchmark failed" exception=e
    finally
        # Cleanup
        rm(BENCHMARK_DB, force=true)
    end
end

# Run benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end