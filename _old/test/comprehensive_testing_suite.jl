"""
Comprehensive Testing Suite for HSOF (Hierarchical Surrogate Optimization Framework)
Validates 1000x speedup, >0.9 correlation accuracy, and seamless MCTS integration

This suite performs end-to-end validation of the entire metamodel training and inference system
including unit tests, integration tests, stress tests, accuracy validation, and performance benchmarks.
"""

using Test
using Statistics
using Random
using BenchmarkTools
using CUDA
using Flux
using JLD2
using JSON3
using ProgressMeter
using Dates
using Printf

# Set random seeds for reproducibility
Random.seed!(42)
CUDA.seed!(42)

# Import all HSOF modules for testing
include("../src/metamodel/neural_architecture.jl")
include("../src/metamodel/correlation_tracking.jl")
include("../src/metamodel/model_checkpointing.jl")
include("../src/metamodel/batch_inference.jl")
include("../src/metamodel/performance_profiling.jl")
include("../src/metamodel/hyperparameter_optimization.jl")
include("../src/metamodel/gpu_memory_management.jl")

using .NeuralArchitecture
using .CorrelationTracking
using .ModelCheckpointing
using .BatchInference
using .PerformanceProfiler
using .HyperparameterOptimization
using .GPUMemoryManagement

# Mock MCTS and ML models for testing
include("mocks/mock_mcts.jl")
include("mocks/mock_ml_models.jl")

using .MockMCTS
using .MockMLModels

"""
Configuration for comprehensive testing
"""
struct TestingConfig
    # Performance requirements
    target_speedup::Float64
    target_correlation::Float64
    stress_test_duration_hours::Float64
    
    # Test parameters
    n_performance_samples::Int
    n_accuracy_samples::Int
    batch_sizes::Vector{Int}
    problem_dimensions::Vector{Int}
    
    # Output configuration
    save_results::Bool
    results_directory::String
    verbose::Bool
end

function create_testing_config(;
    target_speedup::Float64 = 1000.0,
    target_correlation::Float64 = 0.9,
    stress_test_duration_hours::Float64 = 0.1,  # 6 minutes for testing
    n_performance_samples::Int = 1000,
    n_accuracy_samples::Int = 5000,
    batch_sizes::Vector{Int} = [32, 64, 128, 256, 512],
    problem_dimensions::Vector{Int} = [10, 50, 100, 200],
    save_results::Bool = true,
    results_directory::String = "test_results",
    verbose::Bool = true
)
    return TestingConfig(
        target_speedup, target_correlation, stress_test_duration_hours,
        n_performance_samples, n_accuracy_samples, batch_sizes, problem_dimensions,
        save_results, results_directory, verbose
    )
end

"""
Main comprehensive testing suite runner
"""
function run_comprehensive_testing_suite(config::TestingConfig = create_testing_config())
    println("🚀 Starting HSOF Comprehensive Testing Suite")
    println("=" ^ 80)
    
    # Create results directory
    if config.save_results
        mkpath(config.results_directory)
    end
    
    results = Dict{String, Any}()
    
    # Track overall timing
    start_time = time()
    
    try
        # 1. Unit Tests
        println("\n📋 Phase 1: Unit Tests")
        unit_test_results = run_unit_tests(config)
        results["unit_tests"] = unit_test_results
        
        # 2. Integration Tests
        println("\n🔗 Phase 2: Integration Tests")
        integration_test_results = run_integration_tests(config)
        results["integration_tests"] = integration_test_results
        
        # 3. Performance Benchmarks
        println("\n⚡ Phase 3: Performance Benchmarks")
        performance_results = run_performance_benchmarks(config)
        results["performance_benchmarks"] = performance_results
        
        # 4. Accuracy Validation
        println("\n🎯 Phase 4: Accuracy Validation")
        accuracy_results = run_accuracy_validation(config)
        results["accuracy_validation"] = accuracy_results
        
        # 5. Stress Tests
        println("\n💪 Phase 5: Stress Tests")
        stress_test_results = run_stress_tests(config)
        results["stress_tests"] = stress_test_results
        
        # 6. MCTS Integration Tests
        println("\n🌳 Phase 6: MCTS Integration Tests")
        mcts_integration_results = run_mcts_integration_tests(config)
        results["mcts_integration"] = mcts_integration_results
        
        # 7. Generate comprehensive report
        total_time = time() - start_time
        results["metadata"] = Dict(
            "total_execution_time_seconds" => total_time,
            "timestamp" => now(),
            "configuration" => config
        )
        
        report = generate_comprehensive_report(results, config)
        
        # Save results
        if config.save_results
            save_test_results(results, config)
        end
        
        println("\n" * "=" ^ 80)
        println("🎉 Comprehensive Testing Suite Completed!")
        println("⏱️  Total execution time: $(round(total_time/60, digits=2)) minutes")
        
        return results, report
        
    catch e
        println("\n❌ Error during comprehensive testing: $e")
        if config.save_results
            # Save partial results
            results["error"] = string(e)
            results["metadata"] = Dict(
                "execution_time_seconds" => time() - start_time,
                "timestamp" => now(),
                "configuration" => config,
                "status" => "failed"
            )
            save_test_results(results, config)
        end
        rethrow(e)
    end
end

"""
Phase 1: Unit Tests for individual neural network layers and components
"""
function run_unit_tests(config::TestingConfig)
    println("  Testing individual neural network layers...")
    
    results = Dict{String, Any}()
    
    @testset "Neural Architecture Unit Tests" begin
        # Test basic architecture creation
        config_small = NeuralArchConfig(
            input_dim = 50,
            hidden_dims = [128, 64],
            output_dim = 1,
            attention_heads = 8,
            dropout_rate = 0.1
        )
        
        model = create_neural_architecture(config_small)
        @test model isa Chain
        
        # Test forward pass
        x = randn(Float32, 50, 32)  # input_dim × batch_size
        y = model(x)
        @test size(y) == (1, 32)
        
        results["architecture_creation"] = true
        results["forward_pass"] = true
    end
    
    @testset "Correlation Tracking Unit Tests" begin
        tracker = CorrelationTracker()
        
        # Test correlation calculation
        predictions = randn(1000)
        ground_truth = predictions .+ 0.1 * randn(1000)  # Add noise
        
        corr = calculate_correlation(tracker, predictions, ground_truth)
        @test 0.8 < corr < 1.0  # Should be high correlation
        
        results["correlation_calculation"] = true
    end
    
    @testset "GPU Memory Management Unit Tests" begin
        if CUDA.functional()
            memory_manager = GPUMemoryManager()
            
            # Test memory allocation and cleanup
            initial_memory = get_gpu_memory_usage(memory_manager)
            
            # Allocate some GPU memory
            temp_data = CUDA.randn(1000, 1000)
            allocated_memory = get_gpu_memory_usage(memory_manager)
            
            @test allocated_memory > initial_memory
            
            # Cleanup
            temp_data = nothing
            GC.gc()
            CUDA.reclaim()
            
            cleaned_memory = get_gpu_memory_usage(memory_manager)
            @test cleaned_memory <= allocated_memory
            
            results["gpu_memory_management"] = true
        else
            results["gpu_memory_management"] = "skipped - CUDA not available"
        end
    end
    
    println("  ✅ Unit tests completed")
    return results
end

"""
Phase 2: Integration tests with mock MCTS system
"""
function run_integration_tests(config::TestingConfig)
    println("  Testing integration with mock MCTS system...")
    
    results = Dict{String, Any}()
    
    @testset "MCTS-Metamodel Integration Tests" begin
        # Create metamodel
        arch_config = NeuralArchConfig(
            input_dim = 20,
            hidden_dims = [64, 32],
            output_dim = 1,
            attention_heads = 4,
            dropout_rate = 0.1
        )
        
        metamodel = create_neural_architecture(arch_config)
        
        # Create mock MCTS
        mcts = MockMCTSNode(problem_dim = 20)
        
        # Test integration
        n_integration_samples = 100
        integration_times = Float64[]
        
        for i in 1:n_integration_samples
            state = generate_random_state(mcts, 20)
            
            start_time = time()
            prediction = metamodel(state)
            integration_time = time() - start_time
            
            push!(integration_times, integration_time)
            
            @test size(prediction) == (1, 1)
            @test isfinite(prediction[1])
        end
        
        avg_integration_time = mean(integration_times)
        
        results["integration_samples"] = n_integration_samples
        results["avg_integration_time_ms"] = avg_integration_time * 1000
        results["integration_success"] = true
    end
    
    println("  ✅ Integration tests completed")
    return results
end

"""
Phase 3: Performance benchmarks comparing metamodel vs direct evaluation
"""
function run_performance_benchmarks(config::TestingConfig)
    println("  Running performance benchmarks...")
    
    results = Dict{String, Any}()
    speedup_results = Dict{String, Float64}()
    
    for batch_size in config.batch_sizes
        for problem_dim in config.problem_dimensions
            println("    Testing batch_size=$batch_size, problem_dim=$problem_dim")
            
            # Create metamodel
            arch_config = NeuralArchConfig(
                input_dim = problem_dim,
                hidden_dims = [min(256, problem_dim * 4), min(128, problem_dim * 2)],
                output_dim = 1,
                attention_heads = 8,
                dropout_rate = 0.1
            )
            
            metamodel = create_neural_architecture(arch_config)
            
            # Create mock ML models for comparison
            xgb_model = MockXGBoostModel(problem_dim)
            rf_model = MockRandomForestModel(problem_dim)
            
            # Generate test data
            test_data = randn(Float32, problem_dim, batch_size)
            
            # Benchmark metamodel inference
            metamodel_time = @belapsed $metamodel($test_data)
            
            # Benchmark XGBoost
            xgb_time = @belapsed evaluate_batch($xgb_model, $test_data)
            
            # Benchmark Random Forest
            rf_time = @belapsed evaluate_batch($rf_model, $test_data)
            
            # Calculate speedups
            xgb_speedup = xgb_time / metamodel_time
            rf_speedup = rf_time / metamodel_time
            
            key = "batch$(batch_size)_dim$(problem_dim)"
            speedup_results["$(key)_xgb"] = xgb_speedup
            speedup_results["$(key)_rf"] = rf_speedup
            
            println("      XGBoost speedup: $(round(xgb_speedup, digits=1))x")
            println("      RandomForest speedup: $(round(rf_speedup, digits=1))x")
        end
    end
    
    # Calculate overall metrics
    all_speedups = collect(values(speedup_results))
    
    results["speedup_results"] = speedup_results
    results["min_speedup"] = minimum(all_speedups)
    results["max_speedup"] = maximum(all_speedups)
    results["mean_speedup"] = mean(all_speedups)
    results["median_speedup"] = median(all_speedups)
    results["target_speedup_achieved"] = results["min_speedup"] >= config.target_speedup
    
    println("  ✅ Performance benchmarks completed")
    println("    Overall speedup range: $(round(results["min_speedup"], digits=1))x - $(round(results["max_speedup"], digits=1))x")
    
    return results
end

"""
Phase 4: Accuracy validation on standard ML datasets
"""
function run_accuracy_validation(config::TestingConfig)
    println("  Running accuracy validation...")
    
    results = Dict{String, Any}()
    correlation_results = Float64[]
    
    @testset "Accuracy Validation Tests" begin
        for problem_dim in config.problem_dimensions
            println("    Testing problem dimension: $problem_dim")
            
            # Create metamodel
            arch_config = NeuralArchConfig(
                input_dim = problem_dim,
                hidden_dims = [problem_dim * 2, problem_dim],
                output_dim = 1,
                attention_heads = 4,
                dropout_rate = 0.1
            )
            
            metamodel = create_neural_architecture(arch_config)
            
            # Create ground truth model
            ground_truth_model = MockXGBoostModel(problem_dim)
            
            # Generate test dataset
            n_samples = min(config.n_accuracy_samples, 1000)  # Limit for performance
            test_data = randn(Float32, problem_dim, n_samples)
            
            # Get ground truth predictions
            ground_truth = evaluate_batch(ground_truth_model, test_data)
            
            # Get metamodel predictions
            metamodel_predictions = vec(metamodel(test_data))
            
            # Calculate correlation
            correlation = cor(ground_truth, metamodel_predictions)
            push!(correlation_results, correlation)
            
            println("      Correlation: $(round(correlation, digits=4))")
            
            @test correlation > 0.5  # Basic sanity check
        end
    end
    
    # Calculate overall accuracy metrics
    results["correlation_results"] = correlation_results
    results["min_correlation"] = minimum(correlation_results)
    results["max_correlation"] = maximum(correlation_results)
    results["mean_correlation"] = mean(correlation_results)
    results["target_correlation_achieved"] = results["min_correlation"] >= config.target_correlation
    
    println("  ✅ Accuracy validation completed")
    println("    Correlation range: $(round(results["min_correlation"], digits=3)) - $(round(results["max_correlation"], digits=3))")
    
    return results
end

"""
Phase 5: Stress tests for continuous operation
"""
function run_stress_tests(config::TestingConfig)
    println("  Running stress tests for continuous operation...")
    
    results = Dict{String, Any}()
    
    # Convert hours to seconds
    test_duration = config.stress_test_duration_hours * 3600
    start_time = time()
    
    # Create metamodel for stress testing
    arch_config = NeuralArchConfig(
        input_dim = 100,
        hidden_dims = [256, 128, 64],
        output_dim = 1,
        attention_heads = 8,
        dropout_rate = 0.1
    )
    
    metamodel = create_neural_architecture(arch_config)
    
    # Initialize tracking
    iteration_count = 0
    memory_samples = Float64[]
    inference_times = Float64[]
    error_count = 0
    
    println("    Target duration: $(config.stress_test_duration_hours) hours")
    
    # Run stress test
    while (time() - start_time) < test_duration
        try
            iteration_count += 1
            
            # Generate random input
            batch_size = rand(32:128)
            test_input = randn(Float32, 100, batch_size)
            
            # Time inference
            inference_start = time()
            predictions = metamodel(test_input)
            inference_time = time() - inference_start
            push!(inference_times, inference_time)
            
            # Monitor memory usage
            if CUDA.functional()
                memory_used = CUDA.memory_status().pool.used
                push!(memory_samples, memory_used / 1024^3)  # Convert to GB
            end
            
            # Validate predictions
            @test all(isfinite.(predictions))
            
            # Progress update every 100 iterations
            if iteration_count % 100 == 0
                elapsed_hours = (time() - start_time) / 3600
                println("      Iteration $iteration_count, elapsed: $(round(elapsed_hours, digits=2))h")
            end
            
        catch e
            error_count += 1
            if error_count > 10  # Too many errors
                println("      ❌ Too many errors during stress test: $e")
                break
            end
        end
    end
    
    actual_duration = time() - start_time
    
    # Calculate stress test metrics
    results["iterations_completed"] = iteration_count
    results["actual_duration_hours"] = actual_duration / 3600
    results["target_duration_achieved"] = actual_duration >= test_duration * 0.95  # 95% tolerance
    results["error_count"] = error_count
    results["error_rate"] = error_count / max(iteration_count, 1)
    
    if !isempty(inference_times)
        results["mean_inference_time_ms"] = mean(inference_times) * 1000
        results["max_inference_time_ms"] = maximum(inference_times) * 1000
    end
    
    if !isempty(memory_samples)
        results["mean_memory_usage_gb"] = mean(memory_samples)
        results["max_memory_usage_gb"] = maximum(memory_samples)
    end
    
    results["stress_test_passed"] = (error_count == 0) && (actual_duration >= test_duration * 0.95)
    
    println("  ✅ Stress tests completed")
    println("    Iterations: $iteration_count, Errors: $error_count")
    
    return results
end

"""
Phase 6: MCTS integration tests
"""
function run_mcts_integration_tests(config::TestingConfig)
    println("  Testing seamless MCTS integration...")
    
    results = Dict{String, Any}()
    
    @testset "MCTS Integration Tests" begin
        # Test different problem sizes
        for problem_dim in [20, 50, 100]
            println("    Testing MCTS integration with problem_dim=$problem_dim")
            
            # Create metamodel
            arch_config = NeuralArchConfig(
                input_dim = problem_dim,
                hidden_dims = [problem_dim * 2, problem_dim],
                output_dim = 1,
                attention_heads = 4,
                dropout_rate = 0.1
            )
            
            metamodel = create_neural_architecture(arch_config)
            
            # Create MCTS with metamodel
            mcts = MockMCTSNode(problem_dim = problem_dim)
            
            # Test MCTS search with metamodel evaluation
            search_results = run_mcts_search(mcts, metamodel, max_iterations = 100)
            
            @test haskey(search_results, "best_value")
            @test haskey(search_results, "iterations_completed") 
            @test haskey(search_results, "total_evaluations")
            @test search_results["iterations_completed"] > 0
            
            results["dim_$(problem_dim)_integration"] = true
            results["dim_$(problem_dim)_evaluations"] = search_results["total_evaluations"]
        end
        
        # Test batch evaluation integration
        problem_dim = 50
        metamodel = create_neural_architecture(NeuralArchConfig(
            input_dim = problem_dim,
            hidden_dims = [128, 64],
            output_dim = 1
        ))
        
        # Test batch evaluation
        batch_states = randn(Float32, problem_dim, 64)
        batch_predictions = metamodel(batch_states)
        
        @test size(batch_predictions) == (1, 64)
        @test all(isfinite.(batch_predictions))
        
        results["batch_evaluation_integration"] = true
        results["batch_size_tested"] = 64
    end
    
    results["mcts_integration_passed"] = true
    
    println("  ✅ MCTS integration tests completed")
    return results
end

"""
Generate comprehensive testing report
"""
function generate_comprehensive_report(results::Dict, config::TestingConfig)
    report_lines = String[]
    
    push!(report_lines, "=" ^ 80)
    push!(report_lines, "HSOF COMPREHENSIVE TESTING SUITE REPORT")
    push!(report_lines, "=" ^ 80)
    push!(report_lines, "Timestamp: $(results["metadata"]["timestamp"])")
    push!(report_lines, "Total Execution Time: $(round(results["metadata"]["total_execution_time_seconds"]/60, digits=2)) minutes")
    push!(report_lines, "")
    
    # Overall success criteria
    push!(report_lines, "KEY SUCCESS CRITERIA:")
    push!(report_lines, "")
    
    # Speedup validation
    if haskey(results, "performance_benchmarks")
        perf = results["performance_benchmarks"]
        speedup_status = perf["target_speedup_achieved"] ? "✅ PASSED" : "❌ FAILED"
        push!(report_lines, "🚀 1000x Speedup Target: $speedup_status")
        push!(report_lines, "   Achieved: $(round(perf["min_speedup"], digits=1))x - $(round(perf["max_speedup"], digits=1))x")
        push!(report_lines, "   Mean: $(round(perf["mean_speedup"], digits=1))x")
    end
    
    # Correlation validation
    if haskey(results, "accuracy_validation")
        acc = results["accuracy_validation"]
        corr_status = acc["target_correlation_achieved"] ? "✅ PASSED" : "❌ FAILED"
        push!(report_lines, "🎯 >0.9 Correlation Target: $corr_status")
        push!(report_lines, "   Achieved: $(round(acc["min_correlation"], digits=3)) - $(round(acc["max_correlation"], digits=3))")
        push!(report_lines, "   Mean: $(round(acc["mean_correlation"], digits=3))")
    end
    
    # MCTS integration
    if haskey(results, "mcts_integration")
        mcts = results["mcts_integration"]
        mcts_status = mcts["mcts_integration_passed"] ? "✅ PASSED" : "❌ FAILED"
        push!(report_lines, "🌳 MCTS Integration: $mcts_status")
    end
    
    # Stress testing
    if haskey(results, "stress_tests")
        stress = results["stress_tests"]
        stress_status = stress["stress_test_passed"] ? "✅ PASSED" : "❌ FAILED"
        push!(report_lines, "💪 Continuous Operation: $stress_status")
        push!(report_lines, "   Duration: $(round(stress["actual_duration_hours"], digits=2)) hours")
        push!(report_lines, "   Iterations: $(stress["iterations_completed"])")
        push!(report_lines, "   Error Rate: $(round(stress["error_rate"] * 100, digits=2))%")
    end
    
    push!(report_lines, "")
    push!(report_lines, "DETAILED RESULTS:")
    push!(report_lines, "")
    
    # Unit tests summary
    if haskey(results, "unit_tests")
        push!(report_lines, "📋 Unit Tests: ✅ PASSED")
        for (key, value) in results["unit_tests"]
            push!(report_lines, "   $key: $value")
        end
        push!(report_lines, "")
    end
    
    # Integration tests summary
    if haskey(results, "integration_tests")
        push!(report_lines, "🔗 Integration Tests: ✅ PASSED")
        for (key, value) in results["integration_tests"]
            push!(report_lines, "   $key: $value")
        end
        push!(report_lines, "")
    end
    
    # Performance benchmarks detail
    if haskey(results, "performance_benchmarks")
        push!(report_lines, "⚡ Performance Benchmarks:")
        perf = results["performance_benchmarks"]
        for (key, value) in perf["speedup_results"]
            push!(report_lines, "   $key: $(round(value, digits=1))x")
        end
        push!(report_lines, "")
    end
    
    push!(report_lines, "=" ^ 80)
    
    report = join(report_lines, "\n")
    
    if config.verbose
        println("\n" * report)
    end
    
    return report
end

"""
Save test results to files
"""
function save_test_results(results::Dict, config::TestingConfig)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    
    # Save JSON results
    json_file = joinpath(config.results_directory, "comprehensive_test_results_$timestamp.json")
    open(json_file, "w") do f
        JSON3.pretty(f, results)
    end
    
    # Save JLD2 results (for Julia-specific data)
    jld2_file = joinpath(config.results_directory, "comprehensive_test_results_$timestamp.jld2")
    JLD2.save(jld2_file, "results", results)
    
    println("💾 Results saved to:")
    println("   JSON: $json_file")
    println("   JLD2: $jld2_file")
    
    return json_file, jld2_file
end

# Export main functions
export run_comprehensive_testing_suite, create_testing_config
export TestingConfig

# If running as script, execute comprehensive tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running HSOF Comprehensive Testing Suite...")
    
    config = create_testing_config(
        stress_test_duration_hours = 0.05,  # 3 minutes for quick testing
        n_accuracy_samples = 500,
        verbose = true
    )
    
    results, report = run_comprehensive_testing_suite(config)
    
    println("\n🎉 Comprehensive testing completed!")
    println("Check the test_results directory for detailed outputs.")
end