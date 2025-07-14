#!/usr/bin/env julia

# Database Module Test Runner
# Run all database tests

using Test

println("Running HSOF Database Module Tests...")
println("="^60)

# Test files
test_files = [
    "test_connection_pool.jl",
    "test_metadata_parser.jl",
    "test_data_loader.jl",
    "test_column_validator.jl",
    "test_progress_tracker.jl",
    "test_result_writer.jl",
    "test_checkpoint_manager.jl",
    "integration_tests.jl"
]

# Track results
passed_tests = String[]
failed_tests = String[]

# Run each test file
for test_file in test_files
    println("\n📋 Running $test_file...")
    
    try
        include(test_file)
        push!(passed_tests, test_file)
        println("✅ $test_file passed!")
    catch e
        push!(failed_tests, test_file)
        println("❌ $test_file failed!")
        println("Error: ", e)
    end
end

# Summary
println("\n" * "="^60)
println("Test Summary")
println("="^60)
println("Passed: $(length(passed_tests))/$(length(test_files))")

if !isempty(failed_tests)
    println("\nFailed tests:")
    for test in failed_tests
        println("  - $test")
    end
    exit(1)
else
    println("\n✅ All database tests passed!")
    exit(0)
end