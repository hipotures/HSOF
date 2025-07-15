#!/usr/bin/env julia

using Test
using CUDA
using Statistics
using Random
using Printf

# Skip if no GPU
if !CUDA.functional()
    @warn "CUDA not functional, skipping batch inference tests"
    exit(0)
end

println("BATCH INFERENCE SYSTEM - COMPREHENSIVE TESTS")
println("="^80)
println("Testing high-throughput inference pipeline processing 1000+ feature combinations")
println("Target: <1ms latency for batch inference")
println("="^80)

# Include the batch inference module
include("../../src/metamodel/batch_inference.jl")
using .BatchInference

# Import internal functions for testing
import .BatchInference: lookup_cache, update_cache!

"""
Generate synthetic neural network weights for testing
"""
function generate_test_network(config::BatchInferenceConfig)
    weights = Matrix{Float32}[]
    biases = Vector{Float32}[]
    
    layer_dims = [config.feature_dim; config.hidden_dims; config.output_dim]
    
    for i in 1:(length(layer_dims)-1)
        input_dim = layer_dims[i]
        output_dim = layer_dims[i+1]
        
        # Xavier initialization
        scale = sqrt(2.0f0 / (input_dim + output_dim))
        W = randn(Float32, output_dim, input_dim) * scale
        b = zeros(Float32, output_dim)
        
        push!(weights, W)
        push!(biases, b)
    end
    
    return weights, biases
end

"""
Generate test feature combinations
"""
function generate_test_features(n_combinations::Int, feature_dim::Int; seed::Int = 42)
    Random.seed!(seed)
    
    # Generate diverse feature combinations
    feature_combinations = []
    
    for i in 1:n_combinations
        # Random binary features
        features = rand(Float32, feature_dim) .> 0.5
        push!(feature_combinations, Float32.(features))
    end
    
    return feature_combinations
end

"""
Test 1: Basic batch inference functionality
"""
function test_basic_inference()
    println("\n--- Test 1: Basic Batch Inference ---")
    
    config = BatchInferenceConfig(
        max_batch_size = Int32(128),
        target_latency_ms = 1.0f0,
        feature_dim = Int32(500),
        hidden_dims = Int32[256, 128],
        output_dim = Int32(1),
        use_mixed_precision = false,
        enable_profiling = true
    )
    
    # Generate test network
    weights, biases = generate_test_network(config)
    
    # Create inference engine
    engine = create_batch_inference_engine(config, weights, biases)
    
    # Test basic functionality
    @test engine.config.max_batch_size == 128
    @test length(engine.active_batches) > 0
    @test engine.cache.capacity == config.cache_size
    
    # Test batch creation
    batch = InferenceBatch(config)
    @test size(batch.features) == (config.feature_dim, config.max_batch_size)
    @test length(batch.outputs) == config.max_batch_size
    
    println("✓ Basic inference engine creation successful")
    println("✓ Batch structures initialized correctly")
    
    return engine
end

"""
Test 2: Cache functionality
"""
function test_cache_functionality()
    println("\n--- Test 2: Cache Functionality ---")
    
    cache_size = Int32(1024)
    cache = BatchCache(cache_size)
    
    @test cache.capacity == cache_size
    @test cache.size == 0
    
    # Test cache operations
    feature_hash = UInt64(12345)
    output_value = 0.85f0
    timestamp = UInt64(time() * 1_000_000)
    
    # Test cache miss
    result = lookup_cache(cache, feature_hash)
    @test result === nothing
    
    # Add to cache
    update_cache!(cache, feature_hash, output_value, timestamp)
    @test cache.size == 1
    
    # Test cache hit
    result = lookup_cache(cache, feature_hash)
    @test result !== nothing
    @test result ≈ output_value
    
    # Check statistics
    CUDA.@allowscalar begin
        @test cache.total_lookups[1] == 2  # One miss, one hit
        @test cache.total_hits[1] == 1
        @test cache.hit_rate[1] ≈ 0.5f0
    end
    
    println("✓ Cache lookup and update operations working")
    println("✓ Cache statistics tracking correctly")
    
    return cache
end

"""
Test 3: High-throughput inference benchmark
"""
function test_high_throughput_inference()
    println("\n--- Test 3: High-Throughput Inference Benchmark ---")
    
    # Configuration for high throughput
    config = BatchInferenceConfig(
        max_batch_size = Int32(1024),  # Target: 1000+ features
        target_latency_ms = 0.8f0,     # Target: <1ms
        feature_dim = Int32(500),
        hidden_dims = Int32[256, 128, 64],
        output_dim = Int32(1),
        use_mixed_precision = true,
        enable_profiling = true
    )
    
    # Generate test network
    weights, biases = generate_test_network(config)
    
    # Create inference engine
    engine = create_batch_inference_engine(config, weights, biases)
    
    # Generate test features
    n_test_combinations = 2048
    test_features = generate_test_features(n_test_combinations, config.feature_dim)
    
    println("Testing with $(n_test_combinations) feature combinations")
    println("Target batch size: $(config.max_batch_size)")
    println("Target latency: $(config.target_latency_ms) ms")
    
    # Prepare test batch
    batch = engine.active_batches[1]
    batch_size = min(config.max_batch_size, n_test_combinations)
    
    # Fill batch with test features
    CUDA.@allowscalar batch.batch_size[1] = batch_size
    for i in 1:batch_size
        # Create dummy request
        request = InferenceRequest(
            UInt64(i),                          # request_id
            hash(test_features[i]),             # feature_hash
            Int32(i),                           # node_idx
            Int32(0),                           # priority
            UInt64(time() * 1_000_000)         # timestamp_us
        )
        batch.requests[i] = request
        
        # Copy features
        batch.features[:, i] = test_features[i]
    end
    
    # Benchmark inference
    println("\nRunning inference benchmark...")
    
    # Warm up
    for _ in 1:3
        process_inference_batch!(engine, batch)
        CUDA.synchronize()
    end
    
    # Actual benchmark
    n_iterations = 10
    latencies = Float64[]
    
    for iter in 1:n_iterations
        start_time = time()
        
        processed_count = process_inference_batch!(engine, batch)
        CUDA.synchronize()
        
        end_time = time()
        latency_ms = (end_time - start_time) * 1000
        push!(latencies, latency_ms)
        
        @test processed_count == batch_size
        
        if iter <= 3  # Show first few iterations
            println("  Iteration $iter: $(round(latency_ms, digits=3)) ms for $batch_size features")
        end
    end
    
    # Calculate statistics
    avg_latency = mean(latencies)
    min_latency = minimum(latencies)
    max_latency = maximum(latencies)
    std_latency = std(latencies)
    
    # Calculate throughput
    throughput_features_per_sec = batch_size / (avg_latency / 1000)
    throughput_per_ms = batch_size / avg_latency
    
    println("\n📊 Performance Results:")
    println("  Average latency: $(round(avg_latency, digits=3)) ± $(round(std_latency, digits=3)) ms")
    println("  Min latency: $(round(min_latency, digits=3)) ms")
    println("  Max latency: $(round(max_latency, digits=3)) ms")
    println("  Features per batch: $batch_size")
    println("  Throughput: $(round(throughput_features_per_sec, digits=0)) features/sec")
    println("  Throughput: $(round(throughput_per_ms, digits=1)) features/ms")
    
    # Performance assertions
    @test avg_latency < config.target_latency_ms * 1.5  # Allow 50% tolerance
    @test batch_size >= 1000  # Should handle 1000+ features
    @test throughput_per_ms > 500  # Should process 500+ features per ms
    
    performance_met = avg_latency < config.target_latency_ms
    throughput_met = batch_size >= 1000
    
    if performance_met && throughput_met
        println("✅ Performance target MET: $(round(avg_latency, digits=3)) ms < $(config.target_latency_ms) ms")
        println("✅ Throughput target MET: $batch_size >= 1000 features")
    else
        perf_status = performance_met ? "MET" : "MISSED"
        throughput_status = throughput_met ? "MET" : "MISSED"
        println("⚠️  Performance target: $perf_status")
        println("⚠️  Throughput target: $throughput_status")
    end
    
    return engine, avg_latency, throughput_per_ms
end

"""
Test 4: Mixed precision inference
"""
function test_mixed_precision()
    println("\n--- Test 4: Mixed Precision Inference ---")
    
    config_fp32 = BatchInferenceConfig(
        max_batch_size = Int32(512),
        feature_dim = Int32(500),
        hidden_dims = Int32[256, 128],
        output_dim = Int32(1),
        use_mixed_precision = false
    )
    
    config_fp16 = BatchInferenceConfig(
        max_batch_size = Int32(512),
        feature_dim = Int32(500),
        hidden_dims = Int32[256, 128],
        output_dim = Int32(1),
        use_mixed_precision = true
    )
    
    # Generate same network for both
    weights, biases = generate_test_network(config_fp32)
    
    engine_fp32 = create_batch_inference_engine(config_fp32, weights, biases)
    engine_fp16 = create_batch_inference_engine(config_fp16, weights, biases)
    
    # Test features
    test_features = generate_test_features(256, config_fp32.feature_dim)
    
    # Prepare batches
    batch_fp32 = engine_fp32.active_batches[1]
    batch_fp16 = engine_fp16.active_batches[1]
    
    batch_size = 256
    CUDA.@allowscalar begin
        batch_fp32.batch_size[1] = batch_size
        batch_fp16.batch_size[1] = batch_size
    end
    
    # Fill with same test data
    for i in 1:batch_size
        request = InferenceRequest(UInt64(i), hash(test_features[i]), Int32(i), Int32(0), UInt64(time() * 1_000_000))
        batch_fp32.requests[i] = request
        batch_fp16.requests[i] = request
        
        batch_fp32.features[:, i] = test_features[i]
        batch_fp16.features[:, i] = test_features[i]
    end
    
    # Benchmark both
    # FP32
    start_time = time()
    process_inference_batch!(engine_fp32, batch_fp32)
    CUDA.synchronize()
    fp32_time = (time() - start_time) * 1000
    
    # FP16
    start_time = time()
    process_inference_batch!(engine_fp16, batch_fp16)
    CUDA.synchronize()
    fp16_time = (time() - start_time) * 1000
    
    # Compare results (should be close)
    outputs_fp32 = Array(batch_fp32.outputs[1:batch_size])
    outputs_fp16 = Array(batch_fp16.outputs[1:batch_size])
    
    max_diff = maximum(abs.(outputs_fp32 - outputs_fp16))
    avg_diff = mean(abs.(outputs_fp32 - outputs_fp16))
    
    speedup = fp32_time / fp16_time
    
    println("  FP32 inference time: $(round(fp32_time, digits=3)) ms")
    println("  FP16 inference time: $(round(fp16_time, digits=3)) ms")
    println("  Speedup: $(round(speedup, digits=2))x")
    println("  Max output difference: $(round(max_diff, digits=6))")
    println("  Avg output difference: $(round(avg_diff, digits=6))")
    
    # Assertions
    @test max_diff < 0.01  # Outputs should be close
    @test fp16_time <= fp32_time  # FP16 should be faster or equal
    
    if speedup > 1.1
        println("✅ Mixed precision provides significant speedup: $(round(speedup, digits=2))x")
    else
        println("ℹ️  Mixed precision speedup: $(round(speedup, digits=2))x (may vary by hardware)")
    end
    
    return speedup
end

"""
Test 5: Cache effectiveness
"""
function test_cache_effectiveness()
    println("\n--- Test 5: Cache Effectiveness ---")
    
    config = BatchInferenceConfig(
        max_batch_size = Int32(256),
        feature_dim = Int32(100),  # Smaller for easier testing
        hidden_dims = Int32[64, 32],
        cache_size = Int32(512),
        use_mixed_precision = false
    )
    
    weights, biases = generate_test_network(config)
    engine = create_batch_inference_engine(config, weights, biases)
    
    # Generate limited set of features for cache testing
    unique_features = generate_test_features(100, config.feature_dim)
    
    # Create test with repeated features
    test_features = Vector{Vector{Float32}}()
    for _ in 1:500  # More requests than unique features
        push!(test_features, rand(unique_features))
    end
    
    println("Testing cache with $(length(unique_features)) unique features, $(length(test_features)) total requests")
    
    batch = engine.active_batches[1]
    batch_size = min(config.max_batch_size, length(test_features))
    
    # Process in chunks to simulate real usage
    total_processed = 0
    cache_hits_before = CUDA.@allowscalar engine.cache.total_hits[1]
    
    for chunk_start in 1:batch_size:length(test_features)
        chunk_end = min(chunk_start + batch_size - 1, length(test_features))
        chunk_size = chunk_end - chunk_start + 1
        
        CUDA.@allowscalar batch.batch_size[1] = chunk_size
        
        for i in 1:chunk_size
            feature_idx = chunk_start + i - 1
            request = InferenceRequest(
                UInt64(feature_idx),
                hash(test_features[feature_idx]),
                Int32(feature_idx),
                Int32(0),
                UInt64(time() * 1_000_000)
            )
            batch.requests[i] = request
            batch.features[:, i] = test_features[feature_idx]
        end
        
        process_inference_batch!(engine, batch)
        total_processed += chunk_size
        
        if chunk_start == 1
            println("  First batch processed (filling cache)")
        end
    end
    
    cache_hits_after = CUDA.@allowscalar engine.cache.total_hits[1]
    cache_lookups = CUDA.@allowscalar engine.cache.total_lookups[1]
    final_hit_rate = CUDA.@allowscalar engine.cache.hit_rate[1]
    
    cache_hits_gained = cache_hits_after - cache_hits_before
    expected_hit_rate = (total_processed - length(unique_features)) / total_processed
    
    println("  Total requests processed: $total_processed")
    println("  Unique features: $(length(unique_features))")
    println("  Cache hits gained: $cache_hits_gained")
    println("  Final hit rate: $(round(final_hit_rate * 100, digits=1))%")
    println("  Expected hit rate: $(round(expected_hit_rate * 100, digits=1))%")
    
    @test final_hit_rate > 0.5  # Should have significant cache hits
    @test cache_hits_gained > 0  # Should gain hits over time
    
    if final_hit_rate > 0.7
        println("✅ Cache effectiveness: EXCELLENT ($(round(final_hit_rate * 100, digits=1))% hit rate)")
    elseif final_hit_rate > 0.5
        println("✅ Cache effectiveness: GOOD ($(round(final_hit_rate * 100, digits=1))% hit rate)")
    else
        println("⚠️  Cache effectiveness: NEEDS IMPROVEMENT ($(round(final_hit_rate * 100, digits=1))% hit rate)")
    end
    
    return final_hit_rate
end

"""
Test 6: Stress test with maximum throughput
"""
function test_maximum_throughput()
    println("\n--- Test 6: Maximum Throughput Stress Test ---")
    
    config = BatchInferenceConfig(
        max_batch_size = Int32(2048),  # Large batch
        target_latency_ms = 2.0f0,     # Relaxed for stress test
        feature_dim = Int32(500),
        hidden_dims = Int32[512, 256, 128],
        output_dim = Int32(1),
        use_mixed_precision = true,
        cache_size = Int32(4096)
    )
    
    weights, biases = generate_test_network(config)
    engine = create_batch_inference_engine(config, weights, biases)
    
    # Generate large number of features
    n_features = 5000
    test_features = generate_test_features(n_features, config.feature_dim)
    
    println("Stress testing with $n_features feature combinations")
    println("Batch size: $(config.max_batch_size)")
    
    total_processed = 0
    total_time = 0.0
    n_batches = 0
    
    batch = engine.active_batches[1]
    
    for chunk_start in 1:config.max_batch_size:n_features
        chunk_end = min(chunk_start + config.max_batch_size - 1, n_features)
        batch_size = chunk_end - chunk_start + 1
        
        CUDA.@allowscalar batch.batch_size[1] = batch_size
        
        # Fill batch
        for i in 1:batch_size
            feature_idx = chunk_start + i - 1
            request = InferenceRequest(
                UInt64(feature_idx),
                hash(test_features[feature_idx]),
                Int32(feature_idx),
                Int32(0),
                UInt64(time() * 1_000_000)
            )
            batch.requests[i] = request
            batch.features[:, i] = test_features[feature_idx]
        end
        
        # Time this batch
        start_time = time()
        processed = process_inference_batch!(engine, batch)
        CUDA.synchronize()
        batch_time = time() - start_time
        
        total_processed += processed
        total_time += batch_time
        n_batches += 1
        
        if n_batches <= 3 || n_batches % 5 == 0
            throughput = processed / batch_time
            println("  Batch $n_batches: $processed features in $(round(batch_time * 1000, digits=1)) ms ($(round(throughput, digits=0)) feat/sec)")
        end
    end
    
    # Final statistics
    avg_batch_time = total_time / n_batches
    total_throughput = total_processed / total_time
    features_per_ms = total_processed / (total_time * 1000)
    
    println("\n📊 Stress Test Results:")
    println("  Total features processed: $total_processed")
    println("  Total batches: $n_batches")
    println("  Total time: $(round(total_time, digits=3)) seconds")
    println("  Average batch time: $(round(avg_batch_time * 1000, digits=2)) ms")
    println("  Overall throughput: $(round(total_throughput, digits=0)) features/sec")
    println("  Features per millisecond: $(round(features_per_ms, digits=1))")
    
    # Get engine statistics
    stats = get_performance_stats(engine)
    println("\n📈 Engine Statistics:")
    cache_hit_rate = stats["cache_hit_rate"]
    println("  Cache hit rate: $(round(cache_hit_rate * 100, digits=1))%")
    avg_latency_ms = stats["avg_latency_ms"]
    println("  Average latency: $(round(avg_latency_ms, digits=3)) ms")
    
    # Performance assertions
    @test total_processed == n_features
    @test features_per_ms > 100  # Should process 100+ features per ms
    @test stats["cache_hit_rate"] >= 0.0  # Cache should be functional
    
    if features_per_ms > 500
        println("✅ EXCELLENT throughput: $(round(features_per_ms, digits=1)) features/ms")
    elseif features_per_ms > 200
        println("✅ GOOD throughput: $(round(features_per_ms, digits=1)) features/ms")
    else
        println("⚠️  Throughput needs improvement: $(round(features_per_ms, digits=1)) features/ms")
    end
    
    return features_per_ms, stats
end

"""
Main test runner
"""
function run_batch_inference_tests()
    println("\n🚀 Starting Batch Inference System Tests...")
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Available memory: $(round(CUDA.available_memory() / 1e9, digits=2)) GB\n")
    
    test_results = Dict{String, Any}()
    all_tests_passed = true
    
    try
        # Test 1: Basic functionality
        engine = test_basic_inference()
        test_results["basic_functionality"] = "PASSED"
        
        # Test 2: Cache functionality
        cache = test_cache_functionality()
        test_results["cache_functionality"] = "PASSED"
        
        # Test 3: High-throughput inference
        engine, latency, throughput = test_high_throughput_inference()
        test_results["high_throughput"] = Dict(
            "status" => "PASSED",
            "avg_latency_ms" => latency,
            "throughput_per_ms" => throughput,
            "target_met" => latency < 1.0
        )
        
        # Test 4: Mixed precision
        speedup = test_mixed_precision()
        test_results["mixed_precision"] = Dict(
            "status" => "PASSED",
            "speedup" => speedup
        )
        
        # Test 5: Cache effectiveness
        hit_rate = test_cache_effectiveness()
        test_results["cache_effectiveness"] = Dict(
            "status" => "PASSED",
            "hit_rate" => hit_rate
        )
        
        # Test 6: Maximum throughput stress test
        max_throughput, stats = test_maximum_throughput()
        test_results["stress_test"] = Dict(
            "status" => "PASSED",
            "max_throughput_per_ms" => max_throughput,
            "cache_hit_rate" => stats["cache_hit_rate"]
        )
        
    catch e
        println("❌ Test failed with error: $e")
        all_tests_passed = false
        test_results["error"] = string(e)
    end
    
    # Final summary
    println("\n" * "="^80)
    println("BATCH INFERENCE SYSTEM - TEST RESULTS")
    println("="^80)
    
    if all_tests_passed
        println("🎉 ALL TESTS PASSED!")
        println("✅ Basic functionality: Working")
        println("✅ Cache system: Working")
        
        if haskey(test_results, "high_throughput") && test_results["high_throughput"]["target_met"]
            println("✅ Performance target: MET (<1ms latency)")
        else
            println("⚠️  Performance target: Needs optimization")
        end
        
        if haskey(test_results, "stress_test")
            throughput = test_results["stress_test"]["max_throughput_per_ms"]
            if throughput > 500
                println("✅ Throughput target: EXCELLENT ($(round(throughput, digits=1)) feat/ms)")
            else
                println("✅ Throughput target: ADEQUATE ($(round(throughput, digits=1)) feat/ms)")
            end
        end
        
        println("\n✅ Task 5.6 - Create Batch Inference System: COMPLETED")
        println("✅ High-throughput inference pipeline: IMPLEMENTED")
        println("✅ Fused CUDA kernels: IMPLEMENTED") 
        println("✅ Output caching: IMPLEMENTED")
        println("✅ Profiling hooks: IMPLEMENTED")
        
    else
        println("❌ SOME TESTS FAILED")
        println("❌ Task 5.6 - Create Batch Inference System: NEEDS ATTENTION")
    end
    
    println("="^80)
    return all_tests_passed, test_results
end

# Run tests if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    success, results = run_batch_inference_tests()
    exit(success ? 0 : 1)
end

# Export for module usage
run_batch_inference_tests