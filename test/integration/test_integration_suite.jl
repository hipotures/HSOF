using Test
using Random
using Statistics

# Include modules
include("data/dataset_loaders.jl")
include("fixtures/expected_outputs.jl")
include("pipeline_runner.jl")
include("result_validation.jl")

using .DatasetLoaders
using .ExpectedOutputs
using .PipelineRunner
using .ResultValidation

@testset "HSOF Integration Test Suite" begin
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Prepare datasets if not already done
    println("\nPreparing test datasets...")
    datasets = load_all_reference_datasets()
    
    @testset "Dataset Loading Tests" begin
        @test haskey(datasets, "titanic")
        @test haskey(datasets, "mnist")
        @test haskey(datasets, "synthetic")
        
        # Verify dataset dimensions
        @test datasets["titanic"].n_samples == 891
        @test datasets["titanic"].n_features == 12
        
        @test datasets["mnist"].n_samples == 10000  # Subset
        @test datasets["mnist"].n_features == 784
        
        @test datasets["synthetic"].n_samples == 10000
        @test datasets["synthetic"].n_features == 5000
    end
    
    @testset "Pipeline End-to-End Tests" begin
        results = Dict{String, PipelineTestResult}()
        
        @testset "Titanic Dataset Pipeline" begin
            result = run_pipeline_test(datasets["titanic"], verbose=false)
            results["titanic"] = result
            
            @test result.passed
            @test result.total_runtime < 60.0  # Should complete in under 1 minute
            @test result.feature_reduction == [12, 12, 8, 5]  # Expected reduction
            @test all(q >= 0.8 for q in result.quality_scores)  # Good quality
        end
        
        @testset "MNIST Dataset Pipeline" begin
            result = run_pipeline_test(datasets["mnist"], verbose=false)
            results["mnist"] = result
            
            @test result.passed
            @test result.total_runtime < 300.0  # 5 minutes max
            @test result.feature_reduction[1] == 784
            @test result.feature_reduction[end] <= 20  # Final features
            @test all(q >= 0.7 for q in result.quality_scores)
        end
        
        @testset "Synthetic Dataset Pipeline" begin
            result = run_pipeline_test(datasets["synthetic"], verbose=false)
            results["synthetic"] = result
            
            @test result.passed
            @test result.total_runtime < 600.0  # 10 minutes max
            @test result.feature_reduction == [5000, 500, 50, 15]  # Expected reduction
            @test all(q >= 0.6 for q in result.quality_scores)
        end
        
        # Store results for further validation
        global pipeline_results = results
    end
    
    @testset "Quality Validation Tests" begin
        for (name, dataset) in datasets
            if haskey(pipeline_results, name) && pipeline_results[name].passed
                result = pipeline_results[name]
                
                @testset "$name Quality Metrics" begin
                    # Get final features
                    if haskey(result.stage_results, 3)
                        final_features = result.stage_results[3]["selected_features"]
                        X, y, _ = prepare_dataset_for_pipeline(dataset)
                        
                        metrics = calculate_feature_quality(X, y, final_features)
                        
                        @test metrics.relevance_score > 0.0
                        @test metrics.redundancy_score > 0.5  # Low redundancy
                        @test metrics.coverage_score > 0.3    # Reasonable coverage
                        @test metrics.overall_score > 0.5     # Good overall quality
                    end
                end
            end
        end
    end
    
    @testset "Performance Constraint Tests" begin
        for (name, result) in pipeline_results
            @testset "$name Performance" begin
                # Stage-wise performance checks
                for stage in 1:3
                    if haskey(result.stage_results, stage)
                        expected = get_expected_output(name, stage)
                        actual = result.stage_results[stage]
                        
                        @test actual["runtime"] <= expected.max_runtime_seconds
                        @test actual["memory_mb"] <= expected.max_memory_mb
                        @test actual["quality_score"] >= expected.min_quality_score
                    end
                end
            end
        end
    end
    
    @testset "Parametrized Dataset Size Tests" begin
        # Test with different dataset sizes
        sizes = [(100, 50), (1000, 100), (5000, 500)]
        
        @testset "Size: $n_samples × $n_features" for (n_samples, n_features) in sizes
            # Generate small synthetic dataset
            small_dataset = generate_synthetic_dataset(
                n_samples = n_samples,
                n_features = n_features,
                n_informative = min(20, n_features ÷ 5),
                n_redundant = min(10, n_features ÷ 10),
                n_classes = 2
            )
            
            # Run pipeline with adjusted expectations
            config = get_pipeline_config("synthetic")
            config["stage1"]["max_features"] = min(100, n_features ÷ 2)
            config["stage2"]["num_trees"] = min(10, n_features ÷ 50)
            config["stage2"]["max_iterations"] = 1000
            config["stage2"]["gpu_enabled"] = false  # Disable GPU for small datasets
            config["stage3"]["ensemble_size"] = 3
            
            # Quick test - just verify it runs
            result = run_pipeline_test(small_dataset, verbose=false)
            
            @test result.passed || length(result.errors) == 0  # Either pass or no critical errors
            @test result.feature_reduction[1] == n_features
            @test result.feature_reduction[end] < n_features ÷ 2  # At least 50% reduction
        end
    end
    
    @testset "Feature Stability Tests" begin
        # Run multiple times on small dataset to test stability
        n_runs = 3
        run_features = Vector{Vector{Int}}()
        
        small_dataset = datasets["titanic"]  # Use small dataset for speed
        
        for run in 1:n_runs
            result = run_pipeline_test(small_dataset, verbose=false)
            if result.passed && haskey(result.stage_results, 3)
                push!(run_features, result.stage_results[3]["selected_features"])
            end
        end
        
        if length(run_features) >= 2
            stability = validate_feature_stability(run_features, min_stability=0.6)
            @test stability["stable"] || stability["average_similarity"] > 0.5
        end
    end
    
    @testset "Baseline Comparison Tests" begin
        # Compare final results with baseline methods
        for (name, dataset) in datasets
            if haskey(pipeline_results, name) && pipeline_results[name].passed
                result = pipeline_results[name]
                
                if haskey(result.stage_results, 3)
                    final_features = result.stage_results[3]["selected_features"]
                    X, y, _ = prepare_dataset_for_pipeline(dataset)
                    
                    comparison = compare_with_baseline(final_features, X, y)
                    
                    # We should be at least competitive with baselines
                    @test comparison["summary"]["average_improvement"] > -0.1  # Max 10% worse
                    @test comparison["summary"]["better_than_baselines"] >= 1  # Better than at least one
                end
            end
        end
    end
    
    @testset "Integration Report Generation" begin
        # Generate comprehensive reports
        reports = Dict{String, Any}()
        
        for (name, dataset) in datasets
            if haskey(pipeline_results, name)
                result = pipeline_results[name]
                X, y, _ = prepare_dataset_for_pipeline(dataset)
                
                report = generate_validation_report(result, X, y)
                reports[name] = report
                
                # Basic report validation
                @test haskey(report, "dataset")
                @test haskey(report, "passed")
                @test haskey(report, "feature_reduction")
                @test haskey(report, "final_quality")
            end
        end
        
        # Save summary report
        summary_file = "test/integration/test_summary_$(Dates.format(now(), "yyyymmdd_HHMMSS")).txt"
        open(summary_file, "w") do io
            println(io, "HSOF Integration Test Summary")
            println(io, "=" ^ 60)
            println(io, "Generated: $(now())")
            println(io)
            
            for (name, report) in reports
                println(io, "\n$name Dataset:")
                println(io, "  Status: $(report["passed"] ? "PASSED" : "FAILED")")
                println(io, "  Runtime: $(round(report["runtime"], digits=2))s")
                println(io, "  Memory: $(round(report["memory_mb"], digits=2)) MB")
                println(io, "  Feature reduction: $(report["feature_reduction"]["reduction_ratio"])")
                
                if haskey(report, "final_quality")
                    println(io, "  Final quality: $(round(report["final_quality"]["overall"], digits=3))")
                end
                
                if report["issues"]["total_issues"] > 0
                    println(io, "  Issues: $(report["issues"]["total_issues"])")
                end
            end
        end
        
        @test isfile(summary_file)
    end
end

# Print test summary
println("\n" * "=" * 60)
println("Integration Test Suite Complete")
println("=" * 60)