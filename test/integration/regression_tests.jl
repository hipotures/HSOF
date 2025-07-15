"""
Regression Testing Framework for Ensemble MCTS System

Ensures that changes don't break existing functionality through
comprehensive regression testing with known good baselines.
"""

using Test
using CUDA
using JSON3
using Statistics
using Dates
using Pkg

# Import ensemble components
include("../../src/gpu/mcts_gpu.jl")
include("../../src/config/ensemble_config.jl")
include("../../src/config/templates.jl")
include("test_datasets.jl")

using .MCTSGPU
using .EnsembleConfig
using .ConfigTemplates

"""
Regression test configuration
"""
struct RegressionTestConfig
    name::String
    ensemble_config::EnsembleConfiguration
    dataset_config::Dict{String, Any}
    expected_metrics::Dict{String, Any}
    tolerance::Dict{String, Float64}
    
    function RegressionTestConfig(name::String, config::EnsembleConfiguration)
        # Default expected metrics and tolerances
        expected_metrics = Dict{String, Any}(
            "memory_mb" => 50.0,
            "compression_ratio" => 2.0,
            "feature_overlap_ratio" => 0.3,
            "iterations_per_second" => 100.0
        )
        
        tolerance = Dict{String, Float64}(
            "memory_mb" => 0.2,  # 20% tolerance
            "compression_ratio" => 0.1,  # 10% tolerance
            "feature_overlap_ratio" => 0.1,  # 10% tolerance
            "iterations_per_second" => 0.15  # 15% tolerance
        )
        
        dataset_config = Dict{String, Any}(
            "num_samples" => 500,
            "num_features" => 200,
            "relevant_features" => 30,
            "iterations" => 200
        )
        
        new(name, config, dataset_config, expected_metrics, tolerance)
    end
end

"""
Regression test result
"""
struct RegressionTestResult
    test_name::String
    timestamp::DateTime
    success::Bool
    
    # Measured metrics
    measured_metrics::Dict{String, Any}
    expected_metrics::Dict{String, Any}
    
    # Regression analysis
    regressions_detected::Vector{String}
    improvements_detected::Vector{String}
    
    # Performance comparison
    performance_ratio::Dict{String, Float64}
    
    # Error information
    errors::Vector{String}
    warnings::Vector{String}
    
    # Execution info
    execution_time_seconds::Float64
    memory_usage_mb::Float64
    
    function RegressionTestResult(test_name, timestamp, success, measured, expected, 
                                regressions, improvements, ratios, errors, warnings,
                                exec_time, memory)
        new(test_name, timestamp, success, measured, expected, 
            regressions, improvements, ratios, errors, warnings, exec_time, memory)
    end
end

"""
Run single regression test
"""
function run_regression_test(config::RegressionTestConfig)
    @info "Running regression test" config.name
    
    start_time = now()
    errors = String[]
    warnings = String[]
    
    try
        # Skip if CUDA not available
        if !CUDA.functional()
            push!(warnings, "CUDA not available, using CPU fallback")
            # Could implement CPU fallback here
        end
        
        # Generate test dataset
        X, y, names, metadata = generate_synthetic_dataset(
            config.dataset_config["num_samples"],
            config.dataset_config["num_features"],
            relevant_features = config.dataset_config["relevant_features"]
        )
        
        # Create ensemble
        ensemble = MemoryEfficientTreeEnsemble(
            CUDA.device(),
            max_trees = config.ensemble_config.trees_per_gpu,
            max_nodes_per_tree = config.ensemble_config.max_nodes_per_tree
        )
        
        # Initialize ensemble
        initialize_ensemble!(ensemble, [Int[] for _ in 1:config.ensemble_config.trees_per_gpu])
        
        # Create evaluation function
        true_features = metadata["relevant_indices"]
        evaluation_function = function(feature_indices::Vector{Int})
            if isempty(feature_indices)
                return 0.5
            end
            
            overlap = length(intersect(feature_indices, true_features))
            base_score = 0.5 + 0.3 * overlap / length(feature_indices)
            noise = 0.1 * randn()
            return clamp(base_score + noise, 0.0, 1.0)
        end
        
        # Run MCTS
        exec_start = time()
        
        available_actions = UInt16.(1:min(100, config.dataset_config["num_features"]))
        prior_scores = fill(1.0f0 / length(available_actions), length(available_actions))
        
        run_ensemble_mcts!(
            ensemble, 
            config.dataset_config["iterations"], 
            evaluation_function, 
            available_actions, 
            prior_scores
        )
        
        exec_time = time() - exec_start
        
        # Collect metrics
        stats = get_ensemble_statistics(ensemble)
        selected_features = get_best_features_ensemble(ensemble, 30)
        
        # Calculate measured metrics
        measured_metrics = Dict{String, Any}(
            "memory_mb" => stats["memory_stats"]["total_mb"],
            "compression_ratio" => stats["memory_stats"]["compression_ratio"],
            "feature_overlap_ratio" => length(intersect(selected_features, true_features)) / length(selected_features),
            "iterations_per_second" => config.dataset_config["iterations"] / exec_time,
            "selected_features_count" => length(selected_features),
            "memory_efficiency" => stats["lazy_stats"]["memory_saved_mb"] / stats["memory_stats"]["total_mb"]
        )
        
        # Detect regressions and improvements
        regressions = String[]
        improvements = String[]
        performance_ratios = Dict{String, Float64}()
        
        for (metric, expected_value) in config.expected_metrics
            if haskey(measured_metrics, metric)
                measured_value = measured_metrics[metric]
                tolerance = config.tolerance[metric]
                
                ratio = measured_value / expected_value
                performance_ratios[metric] = ratio
                
                if ratio < (1.0 - tolerance)
                    push!(regressions, "$metric: measured=$measured_value, expected=$expected_value (ratio=$ratio)")
                elseif ratio > (1.0 + tolerance)
                    push!(improvements, "$metric: measured=$measured_value, expected=$expected_value (ratio=$ratio)")
                end
            end
        end
        
        # Overall success determination
        success = isempty(regressions) && isempty(errors)
        
        end_time = now()
        total_time = (end_time - start_time).value / 1000.0
        
        result = RegressionTestResult(
            config.name,
            start_time,
            success,
            measured_metrics,
            config.expected_metrics,
            regressions,
            improvements,
            performance_ratios,
            errors,
            warnings,
            total_time,
            measured_metrics["memory_mb"]
        )
        
        @info "Regression test completed" config.name success regressions=length(regressions) improvements=length(improvements)
        
        return result
        
    catch e
        push!(errors, string(e))
        @error "Regression test failed" config.name error=e
        
        end_time = now()
        total_time = (end_time - start_time).value / 1000.0
        
        # Return failed result
        return RegressionTestResult(
            config.name,
            start_time,
            false,
            Dict{String, Any}(),
            config.expected_metrics,
            ["Test execution failed: $(string(e))"],
            String[],
            Dict{String, Float64}(),
            errors,
            warnings,
            total_time,
            0.0
        )
    end
end

"""
Run comprehensive regression test suite
"""
function run_regression_test_suite(; save_results::Bool = true)
    @info "Starting comprehensive regression test suite"
    
    # Create regression test configurations
    test_configs = [
        RegressionTestConfig("development_regression", development_config()),
        RegressionTestConfig("production_regression", production_config()),
        RegressionTestConfig("benchmark_regression", benchmark_config()),
        RegressionTestConfig("single_gpu_regression", single_gpu_config()),
        RegressionTestConfig("fast_regression", fast_exploration_config())
    ]
    
    # Run all regression tests
    all_results = []
    
    for config in test_configs
        result = run_regression_test(config)
        push!(all_results, result)
    end
    
    # Analyze results
    summary = analyze_regression_results(all_results)
    
    if save_results
        save_regression_results(all_results, summary)
    end
    
    @info "Regression test suite completed" total_tests=length(all_results) successful=summary["successful_tests"]
    
    return all_results, summary
end

"""
Test specific functionality regression
"""
function test_memory_optimization_regression()
    @info "Testing memory optimization regression"
    
    # Test that memory optimizations are working
    config = benchmark_config()
    
    # Create ensemble
    ensemble = MemoryEfficientTreeEnsemble(
        CUDA.device(),
        max_trees = config.trees_per_gpu,
        max_nodes_per_tree = config.max_nodes_per_tree
    )
    
    initialize_ensemble!(ensemble, [Int[] for _ in 1:config.trees_per_gpu])
    
    # Test basic operations
    dummy_eval = features -> 0.5
    actions = UInt16.(1:50)
    priors = fill(0.02f0, 50)
    
    run_ensemble_mcts!(ensemble, 100, dummy_eval, actions, priors)
    
    # Get statistics
    stats = get_ensemble_statistics(ensemble)
    
    # Test expected optimizations
    @test stats["memory_stats"]["compression_ratio"] > 1.5
    @test stats["lazy_stats"]["memory_saved_mb"] >= 0
    @test stats["memory_stats"]["total_mb"] < 200.0
    
    # Test feature selection
    selected_features = get_best_features_ensemble(ensemble, 25)
    @test length(selected_features) <= 25
    @test all(f -> f isa Int, selected_features)
    
    @info "Memory optimization regression test passed" compression_ratio=stats["memory_stats"]["compression_ratio"]
    
    return true
end

"""
Test configuration system regression
"""
function test_configuration_regression()
    @info "Testing configuration system regression"
    
    # Test configuration loading and validation
    configs_to_test = [
        ("development", development_config()),
        ("production", production_config()),
        ("benchmark", benchmark_config())
    ]
    
    for (name, config) in configs_to_test
        # Test configuration creation
        @test config isa EnsembleConfiguration
        
        # Test basic parameters
        @test config.num_trees > 0
        @test config.trees_per_gpu > 0
        @test config.max_nodes_per_tree > 0
        @test length(config.gpu_devices) > 0
        
        # Test configuration conversion
        config_dict = config_to_dict(config)
        @test haskey(config_dict, "num_trees")
        @test haskey(config_dict, "lazy_expansion")
        
        # Test configuration reconstruction
        restored_config = dict_to_config(config_dict)
        @test restored_config.num_trees == config.num_trees
        @test restored_config.lazy_expansion == config.lazy_expansion
        
        @info "Configuration regression test passed" name
    end
    
    return true
end

"""
Test template system regression
"""
function test_template_regression()
    @info "Testing template system regression"
    
    # Test all template functions
    templates = [
        ("development", development_config),
        ("production", production_config),
        ("benchmark", benchmark_config),
        ("single-gpu", single_gpu_config),
        ("fast", fast_exploration_config),
        ("high-memory", high_memory_config)
    ]
    
    for (name, template_func) in templates
        # Test template creation
        config = template_func()
        @test config isa EnsembleConfiguration
        
        # Test template retrieval
        retrieved_config = get_template(name)
        @test retrieved_config isa EnsembleConfiguration
        
        # Test template consistency
        @test retrieved_config.num_trees == config.num_trees
        @test retrieved_config.lazy_expansion == config.lazy_expansion
        
        @info "Template regression test passed" name
    end
    
    return true
end

"""
Analyze regression test results
"""
function analyze_regression_results(results::Vector{RegressionTestResult})
    summary = Dict{String, Any}()
    
    summary["total_tests"] = length(results)
    summary["successful_tests"] = sum(r.success for r in results)
    summary["failed_tests"] = sum(!r.success for r in results)
    summary["success_rate"] = summary["successful_tests"] / summary["total_tests"]
    
    # Collect all regressions
    all_regressions = String[]
    all_improvements = String[]
    
    for result in results
        append!(all_regressions, result.regressions_detected)
        append!(all_improvements, result.improvements_detected)
    end
    
    summary["total_regressions"] = length(all_regressions)
    summary["total_improvements"] = length(all_improvements)
    summary["regressions"] = all_regressions
    summary["improvements"] = all_improvements
    
    # Performance analysis
    if !isempty(results)
        summary["avg_execution_time"] = mean(r.execution_time_seconds for r in results)
        summary["avg_memory_usage"] = mean(r.memory_usage_mb for r in results)
        summary["max_memory_usage"] = maximum(r.memory_usage_mb for r in results)
    end
    
    # Test-specific analysis
    test_summary = Dict{String, Any}()
    for result in results
        test_summary[result.test_name] = Dict(
            "success" => result.success,
            "regressions" => length(result.regressions_detected),
            "improvements" => length(result.improvements_detected),
            "execution_time" => result.execution_time_seconds,
            "memory_usage" => result.memory_usage_mb
        )
    end
    summary["test_details"] = test_summary
    
    summary["analysis_timestamp"] = now()
    
    return summary
end

"""
Save regression test results
"""
function save_regression_results(results::Vector{RegressionTestResult}, summary::Dict)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    output_dir = "test/regression_results/$timestamp"
    mkpath(output_dir)
    
    # Save detailed results
    open(joinpath(output_dir, "regression_results.json"), "w") do io
        JSON3.pretty(io, results)
    end
    
    # Save summary
    open(joinpath(output_dir, "regression_summary.json"), "w") do io
        JSON3.pretty(io, summary)
    end
    
    # Create regression report
    report = generate_regression_report(results, summary)
    open(joinpath(output_dir, "regression_report.md"), "w") do io
        write(io, report)
    end
    
    @info "Regression test results saved to $output_dir"
end

"""
Generate regression report
"""
function generate_regression_report(results::Vector{RegressionTestResult}, summary::Dict)
    report = """
    # Regression Test Report
    
    Generated: $(now())
    
    ## Summary
    
    - **Total Tests**: $(summary["total_tests"])
    - **Successful Tests**: $(summary["successful_tests"])
    - **Failed Tests**: $(summary["failed_tests"])
    - **Success Rate**: $(round(summary["success_rate"] * 100, digits=1))%
    - **Total Regressions**: $(summary["total_regressions"])
    - **Total Improvements**: $(summary["total_improvements"])
    
    ## Test Results
    
    """
    
    for result in results
        status = result.success ? "✅ PASS" : "❌ FAIL"
        report *= """
        ### $(result.test_name) $status
        
        - **Execution Time**: $(round(result.execution_time_seconds, digits=2))s
        - **Memory Usage**: $(round(result.memory_usage_mb, digits=2)) MB
        - **Regressions**: $(length(result.regressions_detected))
        - **Improvements**: $(length(result.improvements_detected))
        
        """
        
        if !isempty(result.regressions_detected)
            report *= "**Regressions Detected:**\n"
            for regression in result.regressions_detected
                report *= "- $regression\n"
            end
            report *= "\n"
        end
        
        if !isempty(result.improvements_detected)
            report *= "**Improvements Detected:**\n"
            for improvement in result.improvements_detected
                report *= "- $improvement\n"
            end
            report *= "\n"
        end
    end
    
    if summary["total_regressions"] > 0
        report *= """
        ## ⚠️ Regression Alert
        
        $(summary["total_regressions"]) regressions detected across all tests.
        Please review the failing tests and address any performance degradations.
        
        """
    end
    
    if summary["total_improvements"] > 0
        report *= """
        ## 🎉 Performance Improvements
        
        $(summary["total_improvements"]) performance improvements detected.
        
        """
    end
    
    return report
end

"""
Run basic regression tests
"""
function run_basic_regression_tests()
    @info "Running basic regression tests"
    
    @testset "Basic Regression Tests" begin
        @testset "Memory Optimization" begin
            @test test_memory_optimization_regression()
        end
        
        @testset "Configuration System" begin
            @test test_configuration_regression()
        end
        
        @testset "Template System" begin
            @test test_template_regression()
        end
    end
    
    @info "Basic regression tests completed"
end

export RegressionTestConfig, RegressionTestResult
export run_regression_test, run_regression_test_suite, run_basic_regression_tests
export test_memory_optimization_regression, test_configuration_regression, test_template_regression
export analyze_regression_results, save_regression_results