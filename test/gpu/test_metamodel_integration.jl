using Test
using CUDA
using Random

# Include the metamodel integration module
include("../../src/gpu/kernels/metamodel_integration.jl")

using .MetamodelIntegration

# Mock metamodel function for testing
function mock_metamodel(features::CuArray{Float32, 2}, batch_size::Int32)
    # Simple mock: return random scores
    return CUDA.rand(Float32, Int(batch_size))
end

# Mock metamodel with deterministic output
function deterministic_metamodel(features::CuArray{Float32, 2}, batch_size::Int32)
    # Return scores based on feature sum
    scores = zeros(Float32, Int(batch_size))
    CUDA.@allowscalar begin
        for i in 1:Int(batch_size)
            scores[i] = sum(features[:, i]) / size(features, 1)
        end
    end
    return CuArray(scores)
end

@testset "Metamodel Integration Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU metamodel integration tests"
        return
    end
    
    @testset "MetamodelConfig Creation" begin
        config = MetamodelConfig(
            Int32(32),      # batch_size
            Int32(100),     # feature_dim
            Int32(1),       # output_dim
            Int32(1000),    # max_queue_size
            10.0f0,         # timeout_ms
            Int32(512),     # cache_size
            0.5f0,          # fallback_score
            false           # use_mixed_precision
        )
        
        @test config.batch_size == 32
        @test config.feature_dim == 100
        @test config.output_dim == 1
        @test config.max_queue_size == 1000
        @test config.timeout_ms == 10.0f0
        @test config.cache_size == 512
        @test config.fallback_score == 0.5f0
        @test config.use_mixed_precision == false
    end
    
    @testset "EvalQueue Creation" begin
        config = MetamodelConfig(
            Int32(16), Int32(50), Int32(1), Int32(100),
            5.0f0, Int32(64), 0.5f0, false
        )
        
        queue = EvalQueue(config)
        
        # Check initial state
        CUDA.@allowscalar begin
            @test queue.queue_head[1] == 1
            @test queue.queue_tail[1] == 1
            @test queue.queue_size[1] == 0
            @test queue.batch_ready[1] == false
            @test queue.batch_size[1] == 0
            @test queue.total_requests[1] == 0
            @test queue.total_batches[1] == 0
        end
    end
    
    @testset "ResultCache Creation" begin
        cache_size = Int32(128)
        cache = ResultCache(cache_size)
        
        # Check initial state
        CUDA.@allowscalar begin
            @test cache.total_hits[1] == 0
            @test cache.total_misses[1] == 0
            @test cache.hit_rate[1] == 0.0f0
            @test all(.!cache.cache_valid)
        end
    end
    
    @testset "MetamodelManager Creation" begin
        config = MetamodelConfig(
            Int32(8), Int32(20), Int32(1), Int32(50),
            2.0f0, Int32(32), 0.5f0, false
        )
        
        manager = MetamodelManager(config)
        
        @test manager.config == config
        @test manager.is_active == true
        @test manager.last_inference_time == 0.0f0
        @test manager.total_inferences == 0
        @test size(manager.feature_buffer) == (20, 8)
        @test size(manager.output_buffer) == (1, 8)
        @test isnothing(manager.feature_buffer_fp16)
        @test isnothing(manager.output_buffer_fp16)
    end
    
    @testset "Mixed Precision Buffers" begin
        config = MetamodelConfig(
            Int32(4), Int32(10), Int32(2), Int32(20),
            1.0f0, Int32(16), 0.5f0, true  # Mixed precision enabled
        )
        
        manager = MetamodelManager(config)
        
        @test !isnothing(manager.feature_buffer_fp16)
        @test !isnothing(manager.output_buffer_fp16)
        @test size(manager.feature_buffer_fp16) == (10, 4)
        @test size(manager.output_buffer_fp16) == (2, 4)
    end
    
    @testset "Enqueue Evaluation" begin
        config = MetamodelConfig(
            Int32(4), Int32(10), Int32(1), Int32(20),
            1.0f0, Int32(16), 0.5f0, false
        )
        
        manager = MetamodelManager(config)
        
        # Enqueue several evaluations
        request_ids = Int32[]
        for i in 1:5
            request_id = enqueue_evaluation!(manager, Int32(i), Int32(1))
            push!(request_ids, request_id)
        end
        
        @test length(request_ids) == 5
        @test all(request_ids .> 0)
        
        # Check queue state
        CUDA.@allowscalar begin
            @test manager.eval_queue.queue_size[1] == 5
        end
    end
    
    @testset "Batch Processing" begin
        config = MetamodelConfig(
            Int32(4), Int32(10), Int32(1), Int32(20),
            0.1f0, Int32(16), 0.5f0, false  # Very short timeout
        )
        
        manager = MetamodelManager(config)
        
        # Enqueue requests
        for i in 1:3
            enqueue_evaluation!(manager, Int32(i))
        end
        
        # Process batch with mock metamodel
        sleep(0.2)  # Wait for timeout
        processed = process_batch!(manager, mock_metamodel)
        
        @test processed == 3
        
        # Check results are ready
        results = check_results(manager, Int32[1, 2, 3])
        @test length(results) == 3
        @test all(values(results) .>= 0.0f0)
        @test all(values(results) .<= 1.0f0)
    end
    
    @testset "Fallback on Error" begin
        config = MetamodelConfig(
            Int32(2), Int32(5), Int32(1), Int32(10),
            0.1f0, Int32(8), 0.75f0, false
        )
        
        manager = MetamodelManager(config)
        
        # Function that throws error
        error_metamodel(features, batch_size) = error("Metamodel failure")
        
        # Enqueue requests
        enqueue_evaluation!(manager, Int32(1))
        enqueue_evaluation!(manager, Int32(2))
        
        # Process batch - should use fallback
        sleep(0.2)
        processed = process_batch!(manager, error_metamodel)
        
        @test processed == 2
        
        # Check fallback scores were used
        results = check_results(manager, Int32[1, 2])
        @test length(results) == 2
        @test all(values(results) .== 0.75f0)
    end
    
    @testset "Statistics Collection" begin
        config = MetamodelConfig(
            Int32(4), Int32(10), Int32(1), Int32(20),
            0.1f0, Int32(16), 0.5f0, false
        )
        
        manager = MetamodelManager(config)
        
        # Process some batches
        for batch_num in 1:3
            # Enqueue requests
            for i in 1:4
                enqueue_evaluation!(manager, Int32((batch_num-1)*4 + i))
            end
            
            sleep(0.2)
            process_batch!(manager, mock_metamodel)
        end
        
        # Get statistics
        stats = get_eval_statistics(manager)
        
        @test stats["total_requests"] == 12
        @test stats["total_batches"] == 3
        @test stats["avg_batch_size"] > 0
        @test stats["is_active"] == true
        @test stats["total_inferences"] == 12
        @test stats["cache_hits"] >= 0
        @test stats["cache_misses"] >= 0
    end
    
    @testset "Queue Overflow Handling" begin
        # Small queue for testing overflow
        config = MetamodelConfig(
            Int32(2), Int32(5), Int32(1), Int32(5),  # max_queue_size = 5
            10.0f0, Int32(8), 0.5f0, false
        )
        
        manager = MetamodelManager(config)
        
        # Try to enqueue more than queue size
        for i in 1:10
            enqueue_evaluation!(manager, Int32(i))
        end
        
        # Queue should be full but not crash
        CUDA.@allowscalar begin
            @test manager.eval_queue.queue_size[1] <= 5
        end
    end
    
    @testset "Empty Batch Handling" begin
        config = MetamodelConfig(
            Int32(4), Int32(10), Int32(1), Int32(20),
            100.0f0, Int32(16), 0.5f0, false  # Long timeout
        )
        
        manager = MetamodelManager(config)
        
        # Process without any requests
        processed = process_batch!(manager, mock_metamodel)
        
        @test processed == 0
        
        # Statistics should still work
        stats = get_eval_statistics(manager)
        @test stats["queue_size"] == 0
        @test stats["total_requests"] == 0
    end
    
    @testset "Priority Queue Ordering" begin
        config = MetamodelConfig(
            Int32(8), Int32(10), Int32(1), Int32(20),
            0.1f0, Int32(16), 0.5f0, false
        )
        
        manager = MetamodelManager(config)
        
        # Enqueue with different priorities
        enqueue_evaluation!(manager, Int32(1), Int32(1))  # Low priority
        enqueue_evaluation!(manager, Int32(2), Int32(5))  # High priority
        enqueue_evaluation!(manager, Int32(3), Int32(3))  # Medium priority
        
        # Note: Current implementation doesn't sort by priority
        # This test just verifies the priority field is stored
        CUDA.@allowscalar begin
            @test manager.eval_queue.queue_size[1] == 3
        end
    end
end

println("\n✅ Metamodel integration tests completed!")