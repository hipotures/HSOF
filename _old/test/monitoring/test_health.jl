using Test
using HSOF.Health
using HTTP
using JSON3
using CUDA

@testset "Health Monitoring Tests" begin
    
    @testset "Health Status Enum" begin
        @test Int(HEALTHY) == 1
        @test Int(WARNING) == 2
        @test Int(CRITICAL) == 3
        @test Int(UNKNOWN) == 4
        
        # Test ordering for max() operations
        @test max(HEALTHY, WARNING) == WARNING
        @test max(WARNING, CRITICAL) == CRITICAL
        @test max(HEALTHY, CRITICAL) == CRITICAL
    end
    
    @testset "HealthCheck Structure" begin
        # Test basic construction
        check = HealthCheck(HEALTHY)
        @test check.status == HEALTHY
        @test check.message == ""
        @test isempty(check.details)
        @test check.timestamp isa DateTime
        
        # Test with message and details
        details = Dict{String, Any}("test" => true, "value" => 42)
        check = HealthCheck(WARNING, "Test warning", details)
        @test check.status == WARNING
        @test check.message == "Test warning"
        @test check.details["test"] == true
        @test check.details["value"] == 42
    end
    
    @testset "GPU Health Checks" begin
        gpu_health = check_gpu_health()
        @test gpu_health isa HealthCheck
        @test haskey(gpu_health.details, "gpu_count")
        
        if CUDA.functional()
            # When CUDA is available
            @test gpu_health.details["gpu_count"] >= 0
            
            if gpu_health.details["gpu_count"] > 0
                @test haskey(gpu_health.details, "gpus")
                @test gpu_health.details["gpus"] isa Vector
                
                # Check GPU details structure
                for gpu in gpu_health.details["gpus"]
                    @test haskey(gpu, "device_id")
                    @test haskey(gpu, "available")
                    @test haskey(gpu, "memory_used_gb")
                    @test haskey(gpu, "memory_total_gb")
                    @test haskey(gpu, "memory_percentage")
                    @test haskey(gpu, "temperature_celsius")
                    @test haskey(gpu, "utilization_percent")
                    @test haskey(gpu, "power_watts")
                end
            end
        else
            # When CUDA is not available
            @test gpu_health.status == CRITICAL
            @test gpu_health.details["cuda_available"] == false
        end
    end
    
    @testset "Model Health Checks" begin
        # Test initial state
        model_health = check_model_health()
        @test model_health isa HealthCheck
        @test haskey(model_health.details, "metamodel_loaded")
        @test haskey(model_health.details, "inference_ready")
        @test haskey(model_health.details, "average_inference_ms")
        @test haskey(model_health.details, "error_rate")
        
        # Test health updates
        Health.update_model_health(metamodel_loaded=true, inference_ready=true)
        model_health = check_model_health()
        @test model_health.details["metamodel_loaded"] == true
        @test model_health.details["inference_ready"] == true
        
        # Test inference time updates
        Health.update_model_health(inference_time_ms=50.0)
        Health.update_model_health(inference_time_ms=60.0)
        model_health = check_model_health()
        @test model_health.details["average_inference_ms"] > 0
        
        # Test error rate updates
        Health.update_model_health(error_occurred=true)
        model_health = check_model_health()
        @test model_health.details["error_rate"] > 0
    end
    
    @testset "Pipeline Health Checks" begin
        pipeline_health = check_pipeline_health()
        @test pipeline_health isa HealthCheck
        @test haskey(pipeline_health.details, "stage1_operational")
        @test haskey(pipeline_health.details, "stage2_operational")
        @test haskey(pipeline_health.details, "stage3_operational")
        @test haskey(pipeline_health.details, "database_connected")
        @test haskey(pipeline_health.details, "redis_connected")
        @test haskey(pipeline_health.details, "filesystem_accessible")
        
        # Test pipeline updates
        Health.update_pipeline_health(
            stage1=true,
            stage2=true,
            stage3=true,
            database=true,
            redis=true,
            filesystem=true
        )
        
        pipeline_health = check_pipeline_health()
        @test pipeline_health.status == HEALTHY
        @test all(values(pipeline_health.details))  # All should be true
        
        # Test partial failure
        Health.update_pipeline_health(stage2=false)
        pipeline_health = check_pipeline_health()
        @test pipeline_health.status == CRITICAL
        @test occursin("Stage 2", pipeline_health.message)
    end
    
    @testset "Health Aggregation" begin
        overall_health = aggregate_health()
        @test overall_health isa HealthCheck
        @test haskey(overall_health.details, "overall_status")
        @test haskey(overall_health.details, "components")
        @test haskey(overall_health.details, "timestamp")
        
        components = overall_health.details["components"]
        @test haskey(components, "gpu")
        @test haskey(components, "model")
        @test haskey(components, "pipeline")
        
        # Each component should have status, message, and details
        for component in values(components)
            @test haskey(component, "status")
            @test haskey(component, "message")
            @test haskey(component, "details")
        end
    end
    
    @testset "Health Configuration" begin
        # Test that configuration values are properly set
        @test Health.HEALTH_CONFIG["gpu_memory_warning_threshold"] == 80.0
        @test Health.HEALTH_CONFIG["gpu_memory_critical_threshold"] == 95.0
        @test Health.HEALTH_CONFIG["gpu_temp_warning_threshold"] == 80.0
        @test Health.HEALTH_CONFIG["gpu_temp_critical_threshold"] == 90.0
        @test Health.HEALTH_CONFIG["model_error_rate_warning"] == 0.05
        @test Health.HEALTH_CONFIG["model_error_rate_critical"] == 0.10
        @test Health.HEALTH_CONFIG["inference_latency_warning_ms"] == 100.0
        @test Health.HEALTH_CONFIG["inference_latency_critical_ms"] == 500.0
    end
    
    @testset "Recovery Mechanisms" begin
        # Test GPU recovery (mock test without actual GPU reset)
        if CUDA.functional() && length(CUDA.devices()) > 0
            # We can't actually test GPU reset in unit tests
            # Just verify the function exists and returns a boolean
            @test Health.recover_gpu_error(0) isa Bool
        end
        
        # Test model recovery
        @test Health.recover_model_error() isa Bool
        
        # Test checkpoint recovery with non-existent file
        @test Health.recover_from_checkpoint("/tmp/nonexistent.jld2") == false
    end
    
    @testset "HTTP Health Endpoints" begin
        # Test health handler responses
        function test_endpoint(path, expected_keys)
            req = HTTP.Request("GET", path)
            response = Health.health_handler(req)
            @test response.status ∈ [200, 503, 500]
            
            if response.status != 404
                body = JSON3.read(String(response.body))
                for key in expected_keys
                    @test haskey(body, key)
                end
            end
            
            return response
        end
        
        # Test main health endpoint
        resp = test_endpoint("/health", ["overall_status", "components", "timestamp"])
        
        # Test GPU health endpoint
        resp = test_endpoint("/health/gpu", ["gpu_count"])
        
        # Test model health endpoint
        resp = test_endpoint("/health/model", ["metamodel_loaded", "inference_ready"])
        
        # Test pipeline health endpoint
        resp = test_endpoint("/health/pipeline", ["stage1_operational", "stage2_operational"])
        
        # Test 404 for unknown endpoint
        req = HTTP.Request("GET", "/health/unknown")
        resp = Health.health_handler(req)
        @test resp.status == 404
    end
end