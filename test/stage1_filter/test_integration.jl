module Stage1IntegrationTests

using Test
using CUDA
using Statistics
using Random
using DataFrames
using CSV
using Dates

# Include all Stage 1 modules
include("../../src/stage1_filter/gpu_memory_layout.jl")
include("../../src/stage1_filter/mutual_information.jl")
include("../../src/stage1_filter/correlation_matrix.jl")
include("../../src/stage1_filter/variance_calculation.jl")
include("../../src/stage1_filter/feature_ranking.jl")
include("../../src/stage1_filter/kernel_fusion.jl")
include("../../src/stage1_filter/fused_pipeline.jl")

using .GPUMemoryLayout
using .MutualInformation
using .CorrelationMatrix
using .VarianceCalculation
using .FeatureRanking
using .KernelFusion
using .FusedPipeline

export run_integration_tests, run_dataset_size_tests, run_edge_case_tests

"""
Test configuration for integration tests
"""
struct IntegrationTestConfig
    dataset_sizes::Vector{Tuple{Int,Int}}  # (n_samples, n_features)
    n_features_select::Int
    test_iterations::Int
    save_results::Bool
    results_dir::String
end

function IntegrationTestConfig(;
    dataset_sizes = [(1000, 5000), (10000, 5000), (100000, 5000), (1000000, 5000)],
    n_features_select = 500,
    test_iterations = 3,
    save_results = true,
    results_dir = "test_results"
)
    return IntegrationTestConfig(
        dataset_sizes,
        n_features_select,
        test_iterations,
        save_results,
        results_dir
    )
end

"""
Generate synthetic dataset with known properties
"""
function generate_test_dataset(n_samples::Int, n_features::Int;
    n_informative::Int = 100,
    n_redundant::Int = 50,
    n_constant::Int = 10,
    noise_level::Float32 = 0.1f0,
    seed::Int = 42
)
    Random.seed!(seed)
    
    X = zeros(Float32, n_features, n_samples)
    y = zeros(Int32, n_samples)
    
    # Informative features
    for i in 1:n_informative
        X[i, :] = randn(Float32, n_samples)
        weight = randn(Float32)
        y .+= Int32.(round.(X[i, :] * weight))
    end
    
    # Redundant features (correlated with informative)
    for i in 1:n_redundant
        source_idx = rand(1:n_informative)
        X[n_informative + i, :] = X[source_idx, :] + noise_level * randn(Float32, n_samples)
    end
    
    # Random noise features
    n_noise = n_features - n_informative - n_redundant - n_constant
    for i in 1:n_noise
        X[n_informative + n_redundant + i, :] = randn(Float32, n_samples)
    end
    
    # Constant features
    for i in 1:n_constant
        X[n_features - n_constant + i, :] .= Float32(i)
    end
    
    # Convert y to classes
    y = Int32.((y .> median(y)) .+ 1)
    
    return X, y, (n_informative, n_redundant, n_constant)
end

"""
Run integration tests for different dataset sizes
"""
function run_dataset_size_tests(config::IntegrationTestConfig)
    println("\n" * "="^80)
    println("STAGE 1 INTEGRATION TESTS - DATASET SIZE SCALING")
    println("="^80)
    
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return nothing
    end
    
    results = DataFrame(
        n_samples = Int[],
        n_features = Int[],
        time_total = Float64[],
        time_variance = Float64[],
        time_mi = Float64[],
        time_correlation = Float64[],
        time_ranking = Float64[],
        memory_used_mb = Float64[],
        throughput_features_sec = Float64[],
        n_selected = Int[],
        n_informative_found = Int[]
    )
    
    for (n_samples, n_features) in config.dataset_sizes
        println("\n--- Testing $(n_samples) samples × $(n_features) features ---")
        
        # Skip if insufficient memory
        required_memory = n_samples * n_features * sizeof(Float32) * 4 / 1024^3
        available_memory = CUDA.available_memory() / 1024^3
        
        if required_memory > available_memory * 0.8
            println("Skipping - insufficient GPU memory (need $(round(required_memory, digits=1))GB)")
            continue
        end
        
        # Generate dataset
        println("Generating dataset...")
        X, y, (n_info, n_red, n_const) = generate_test_dataset(
            n_samples, n_features,
            n_informative = min(100, n_features ÷ 10),
            n_redundant = min(50, n_features ÷ 20)
        )
        
        # Transfer to GPU
        X_gpu = CuArray(X)
        y_gpu = CuArray(y)
        CUDA.synchronize()
        
        # Record initial memory
        CUDA.reclaim()
        initial_memory = CUDA.memory_status().used
        
        # Time each component
        times = Dict{String, Float64}()
        
        # 1. Variance filtering
        t_var = @elapsed begin
            var_config = VarianceConfig()
            variances = compute_variance(X_gpu)
            valid_mask = filter_by_variance(variances, var_config.threshold)
            CUDA.synchronize()
        end
        times["variance"] = t_var
        
        # 2. Mutual information
        t_mi = @elapsed begin
            mi_config = MutualInformationConfig()
            mi_scores = compute_mutual_information(X_gpu, y_gpu, mi_config)
            CUDA.synchronize()
        end
        times["mi"] = t_mi
        
        # 3. Correlation matrix
        t_corr = @elapsed begin
            corr_config = CorrelationConfig()
            corr_matrix = compute_correlation_matrix(X_gpu, corr_config)
            redundant_pairs = find_redundant_features(corr_matrix, corr_config.threshold)
            CUDA.synchronize()
        end
        times["correlation"] = t_corr
        
        # 4. Feature ranking
        t_rank = @elapsed begin
            rank_config = RankingConfig(n_features_to_select = config.n_features_select)
            selected_features = select_features(X_gpu, y_gpu, rank_config)
            CUDA.synchronize()
        end
        times["ranking"] = t_rank
        
        # Total time
        t_total = sum(values(times))
        
        # Memory usage
        peak_memory = CUDA.memory_status().used
        memory_used = (peak_memory - initial_memory) / 1024^2
        
        # Calculate metrics
        selected_cpu = Array(selected_features)
        valid_selected = selected_cpu[selected_cpu .> 0]
        n_selected = length(valid_selected)
        n_informative_found = sum(valid_selected .<= n_info)
        throughput = n_features / t_total
        
        # Store results
        push!(results, (
            n_samples, n_features, t_total,
            times["variance"], times["mi"], times["correlation"], times["ranking"],
            memory_used, throughput, n_selected, n_informative_found
        ))
        
        # Print summary
        println("✓ Total time: $(round(t_total, digits=2))s")
        println("  - Variance: $(round(times["variance"]*1000, digits=1))ms")
        println("  - MI: $(round(times["mi"]*1000, digits=1))ms")
        println("  - Correlation: $(round(times["correlation"]*1000, digits=1))ms")
        println("  - Ranking: $(round(times["ranking"]*1000, digits=1))ms")
        println("✓ Memory used: $(round(memory_used, digits=1))MB")
        println("✓ Throughput: $(round(throughput, digits=0)) features/sec")
        println("✓ Selected $n_selected features ($n_informative_found informative)")
        
        # Test assertions
        @test t_total < 60.0  # Should complete within 1 minute
        @test n_selected > 0  # Should select some features
        @test n_informative_found > n_selected * 0.5  # Should find majority informative
    end
    
    # Save results if requested
    if config.save_results && !isempty(results)
        mkpath(config.results_dir)
        filename = joinpath(config.results_dir, "integration_test_results_$(Dates.now()).csv")
        CSV.write(filename, results)
        println("\nResults saved to: $filename")
    end
    
    return results
end

"""
Run edge case tests
"""
function run_edge_case_tests()
    println("\n" * "="^80)
    println("STAGE 1 INTEGRATION TESTS - EDGE CASES")
    println("="^80)
    
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return
    end
    
    # Test 1: Empty dataset
    println("\n--- Test 1: Empty Dataset ---")
    X_empty = CUDA.zeros(Float32, 100, 0)
    y_empty = CuArray(Int32[])
    
    @test_throws Exception select_features(
        X_empty, y_empty, 
        RankingConfig(n_features_to_select = 10)
    )
    println("✓ Empty dataset handled correctly")
    
    # Test 2: Single sample
    println("\n--- Test 2: Single Sample ---")
    X_single = CUDA.randn(Float32, 100, 1)
    y_single = CuArray([Int32(1)])
    
    selected_single = select_features(
        X_single, y_single,
        RankingConfig(n_features_to_select = 10)
    )
    @test length(Array(selected_single)) == 10
    println("✓ Single sample handled correctly")
    
    # Test 3: All constant features
    println("\n--- Test 3: All Constant Features ---")
    X_const = CUDA.ones(Float32, 100, 1000)
    y_const = CuArray(Int32.(rand(1:2, 1000)))
    
    selected_const = select_features(
        X_const, y_const,
        RankingConfig(n_features_to_select = 10)
    )
    valid_const = Array(selected_const)[Array(selected_const) .> 0]
    @test length(valid_const) == 0  # No features should be selected
    println("✓ All constant features filtered out")
    
    # Test 4: All identical features
    println("\n--- Test 4: All Identical Features ---")
    base_feature = randn(Float32, 1000)
    X_identical = CUDA.zeros(Float32, 50, 1000)
    for i in 1:50
        X_identical[i, :] = base_feature
    end
    y_identical = CuArray(Int32.(rand(1:2, 1000)))
    
    selected_identical = select_features(
        X_identical, y_identical,
        RankingConfig(
            n_features_to_select = 10,
            correlation_threshold = 0.99f0
        )
    )
    valid_identical = Array(selected_identical)[Array(selected_identical) .> 0]
    @test length(valid_identical) <= 2  # Should select very few
    println("✓ Identical features correctly deduplicated")
    
    # Test 5: Extreme values
    println("\n--- Test 5: Extreme Values ---")
    X_extreme = CUDA.zeros(Float32, 100, 500)
    X_extreme[1:20, :] = 1e-30 * CUDA.randn(Float32, 20, 500)  # Very small
    X_extreme[21:40, :] = 1e10 * CUDA.randn(Float32, 20, 500)  # Very large
    X_extreme[41:60, :] = CUDA.randn(Float32, 20, 500)         # Normal
    X_extreme[61:80, :] .= Inf32                                # Infinity
    X_extreme[81:100, :] .= NaN32                               # NaN
    
    y_extreme = CuArray(Int32.(rand(1:2, 500)))
    
    # Should handle without crashing
    selected_extreme = select_features(
        X_extreme, y_extreme,
        RankingConfig(n_features_to_select = 10)
    )
    valid_extreme = Array(selected_extreme)[Array(selected_extreme) .> 0]
    @test length(valid_extreme) > 0  # Should select some valid features
    @test all(valid_extreme .<= 60)  # Should not select Inf/NaN features
    println("✓ Extreme values handled correctly")
    
    # Test 6: Memory stress test
    println("\n--- Test 6: Memory Stress Test ---")
    available_gb = CUDA.available_memory() / 1024^3
    
    # Use 70% of available memory
    element_size = sizeof(Float32)
    n_elements = Int(floor(available_gb * 0.7 * 1024^3 / element_size))
    n_features_stress = 5000
    n_samples_stress = n_elements ÷ n_features_stress
    
    println("Creating $(round(n_elements * element_size / 1024^3, digits=1))GB dataset...")
    
    X_stress = CUDA.randn(Float32, n_features_stress, n_samples_stress)
    y_stress = CuArray(Int32.(rand(1:3, n_samples_stress)))
    
    # Should complete without OOM
    t_stress = @elapsed begin
        selected_stress = select_features(
            X_stress, y_stress,
            RankingConfig(n_features_to_select = 100)
        )
        CUDA.synchronize()
    end
    
    @test t_stress < 300.0  # Should complete within 5 minutes
    println("✓ Memory stress test passed ($(round(t_stress, digits=1))s)")
    
    # Cleanup
    CUDA.reclaim()
    
    println("\n✓ All edge case tests passed!")
end

"""
Run performance benchmarks
"""
function run_performance_benchmarks(config::IntegrationTestConfig)
    println("\n" * "="^80)
    println("STAGE 1 PERFORMANCE BENCHMARKS")
    println("="^80)
    
    if !CUDA.functional()
        @warn "CUDA not functional, skipping benchmarks"
        return nothing
    end
    
    # Standard benchmark dataset
    n_samples = 10000
    n_features = 5000
    
    println("Benchmark dataset: $n_samples samples × $n_features features")
    
    # Generate dataset
    X, y, _ = generate_test_dataset(n_samples, n_features)
    X_gpu = CuArray(X)
    y_gpu = CuArray(y)
    
    # Warm up
    select_features(X_gpu, y_gpu, RankingConfig(n_features_to_select = 100))
    CUDA.synchronize()
    
    # Benchmark different configurations
    configs = [
        ("Default", RankingConfig()),
        ("High Selectivity", RankingConfig(n_features_to_select = 100)),
        ("Low Selectivity", RankingConfig(n_features_to_select = 1000)),
        ("No Correlation", RankingConfig(correlation_threshold = 1.0f0)),
        ("Strict Correlation", RankingConfig(correlation_threshold = 0.8f0)),
        ("High MI Threshold", RankingConfig(mi_threshold = 0.1f0))
    ]
    
    results = DataFrame(
        config_name = String[],
        time_ms = Float64[],
        memory_mb = Float64[],
        features_selected = Int[],
        throughput_feat_sec = Float64[]
    )
    
    for (name, cfg) in configs
        println("\n--- Benchmarking: $name ---")
        
        # Record memory before
        CUDA.reclaim()
        mem_before = CUDA.memory_status().used
        
        # Run multiple iterations
        times = Float64[]
        for i in 1:config.test_iterations
            t = @elapsed begin
                selected = select_features(X_gpu, y_gpu, cfg)
                CUDA.synchronize()
            end
            push!(times, t)
        end
        
        # Get memory usage
        mem_after = CUDA.memory_status().used
        memory_used_mb = (mem_after - mem_before) / 1024^2
        
        # Calculate metrics
        avg_time = mean(times) * 1000  # Convert to ms
        std_time = std(times) * 1000
        min_time = minimum(times) * 1000
        selected = select_features(X_gpu, y_gpu, cfg)
        n_selected = sum(Array(selected) .> 0)
        throughput = n_features / (avg_time / 1000)
        
        # Store results
        push!(results, (name, avg_time, memory_used_mb, n_selected, throughput))
        
        # Print results
        println("  Time: $(round(avg_time, digits=1))ms ± $(round(std_time, digits=1))ms")
        println("  Min time: $(round(min_time, digits=1))ms")
        println("  Memory: $(round(memory_used_mb, digits=1))MB")
        println("  Selected: $n_selected features")
        println("  Throughput: $(round(throughput, digits=0)) features/sec")
    end
    
    # Compare with fused pipeline
    println("\n--- Benchmarking: Fused Pipeline ---")
    
    fused_config = FusedPipelineConfig(n_features_to_select = Int32(500))
    selected_fused = CUDA.fill(Int32(-1), 500)
    
    times_fused = Float64[]
    for i in 1:config.test_iterations
        t = @elapsed begin
            fused_feature_selection_pipeline!(selected_fused, X_gpu, y_gpu, fused_config)
            CUDA.synchronize()
        end
        push!(times_fused, t)
    end
    
    avg_time_fused = mean(times_fused) * 1000
    improvement = (results[1, :time_ms] - avg_time_fused) / results[1, :time_ms] * 100
    
    println("  Time: $(round(avg_time_fused, digits=1))ms")
    println("  Improvement over default: $(round(improvement, digits=1))%")
    
    # Save benchmark results
    if config.save_results
        mkpath(config.results_dir)
        filename = joinpath(config.results_dir, "benchmark_results_$(Dates.now()).csv")
        CSV.write(filename, results)
        println("\nBenchmark results saved to: $filename")
    end
    
    return results
end

"""
Run automated performance regression detection
"""
function run_performance_regression_test(baseline_file::String = "")
    println("\n" * "="^80)
    println("PERFORMANCE REGRESSION TEST")
    println("="^80)
    
    if !CUDA.functional()
        @warn "CUDA not functional, skipping regression test"
        return
    end
    
    # Standard regression test configuration
    n_samples = 10000
    n_features = 5000
    n_runs = 5
    tolerance = 0.05  # 5% tolerance
    
    # Generate dataset
    X, y, _ = generate_test_dataset(n_samples, n_features)
    X_gpu = CuArray(X)
    y_gpu = CuArray(y)
    
    # Warm up
    select_features(X_gpu, y_gpu, RankingConfig())
    CUDA.synchronize()
    
    # Run current performance test
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed begin
            selected = select_features(X_gpu, y_gpu, RankingConfig())
            CUDA.synchronize()
        end
        push!(times, t)
    end
    
    current_time = median(times)
    current_throughput = n_features / current_time
    
    println("Current performance:")
    println("  Time: $(round(current_time*1000, digits=1))ms")
    println("  Throughput: $(round(current_throughput, digits=0)) features/sec")
    
    # Compare with baseline if provided
    if isfile(baseline_file)
        baseline = CSV.read(baseline_file, DataFrame)
        if !isempty(baseline)
            baseline_time = baseline[1, :time_ms] / 1000
            baseline_throughput = baseline[1, :throughput]
            
            regression = (current_time - baseline_time) / baseline_time
            
            println("\nBaseline performance:")
            println("  Time: $(round(baseline_time*1000, digits=1))ms")
            println("  Throughput: $(round(baseline_throughput, digits=0)) features/sec")
            
            if regression > tolerance
                @warn "Performance regression detected!" regression=round(regression*100, digits=1)
                @test false  # Fail the test
            else
                println("\n✓ No performance regression ($(round(regression*100, digits=1))% change)")
                @test true
            end
        end
    else
        println("\nNo baseline file provided - saving current as baseline")
        
        # Save current as baseline
        baseline_data = DataFrame(
            dataset = ["$(n_samples)x$(n_features)"],
            time_ms = [current_time * 1000],
            throughput = [current_throughput],
            date = [Dates.now()]
        )
        
        mkpath("test_results")
        CSV.write("test_results/performance_baseline.csv", baseline_data)
    end
end

"""
Main integration test runner
"""
function run_integration_tests(; 
    test_sizes = true,
    test_edge_cases = true,
    test_benchmarks = true,
    test_regression = true,
    config = IntegrationTestConfig()
)
    println("\n" * "="^80)
    println("RUNNING STAGE 1 FAST FILTERING INTEGRATION TESTS")
    println("="^80)
    println("Date: $(Dates.now())")
    println("GPU: $(CUDA.functional() ? CUDA.name(CUDA.device()) : "Not available")")
    
    test_results = Dict{String, Any}()
    
    # Run dataset size tests
    if test_sizes
        test_results["size_scaling"] = run_dataset_size_tests(config)
    end
    
    # Run edge case tests
    if test_edge_cases
        run_edge_case_tests()
        test_results["edge_cases"] = "Passed"
    end
    
    # Run performance benchmarks
    if test_benchmarks
        test_results["benchmarks"] = run_performance_benchmarks(config)
    end
    
    # Run regression test
    if test_regression
        baseline_file = joinpath(config.results_dir, "performance_baseline.csv")
        run_performance_regression_test(baseline_file)
        test_results["regression"] = "Checked"
    end
    
    println("\n" * "="^80)
    println("ALL INTEGRATION TESTS COMPLETED")
    println("="^80)
    
    return test_results
end

end # module

# Run tests if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    using .Stage1IntegrationTests
    
    # Configure tests
    config = IntegrationTestConfig(
        dataset_sizes = [(1000, 5000), (10000, 5000), (100000, 5000)],
        n_features_select = 500,
        test_iterations = 3,
        save_results = true,
        results_dir = "test_results"
    )
    
    # Run all tests
    results = run_integration_tests(config = config)
end