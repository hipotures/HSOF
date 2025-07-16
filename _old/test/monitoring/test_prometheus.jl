using Test
using HTTP
using JSON3
using CUDA

# Include the modules to test
include("../../src/monitoring/prometheus.jl")
include("../../src/monitoring/prometheus_integration.jl")

using .Prometheus
using .PrometheusIntegration

@testset "Prometheus Metrics Tests" begin
    
    @testset "Basic Metric Operations" begin
        # Clear metrics registry for clean tests
        empty!(Prometheus.METRICS_REGISTRY)
        
        # Test counter operations
        Prometheus.increment_counter!("test_counter", 5.0)
        Prometheus.increment_counter!("test_counter", 3.0)
        
        @test haskey(Prometheus.METRICS_REGISTRY, "test_counter")
        counter = Prometheus.METRICS_REGISTRY["test_counter"]
        @test counter.value == 8.0
        
        # Test gauge operations
        Prometheus.set_gauge!("test_gauge", 42.5)
        Prometheus.set_gauge!("test_gauge", 37.5)
        
        @test haskey(Prometheus.METRICS_REGISTRY, "test_gauge")
        gauge = Prometheus.METRICS_REGISTRY["test_gauge"]
        @test gauge.value == 37.5
        
        # Test histogram operations
        Prometheus.observe_histogram!("test_histogram", 0.1)
        Prometheus.observe_histogram!("test_histogram", 0.5)
        Prometheus.observe_histogram!("test_histogram", 2.0)
        
        @test haskey(Prometheus.METRICS_REGISTRY, "test_histogram")
        histogram = Prometheus.METRICS_REGISTRY["test_histogram"]
        @test histogram.count == 3
        @test histogram.sum == 2.6
    end
    
    @testset "Labeled Metrics" begin
        empty!(Prometheus.METRICS_REGISTRY)
        
        # Test metrics with labels
        labels1 = Dict("gpu" => "0", "type" => "memory")
        labels2 = Dict("gpu" => "1", "type" => "memory")
        
        Prometheus.set_gauge!("gpu_metric", 1024.0, labels1)
        Prometheus.set_gauge!("gpu_metric", 2048.0, labels2)
        
        # Should create separate metrics for each label combination
        key1 = Prometheus.labels_key("gpu_metric", labels1)
        key2 = Prometheus.labels_key("gpu_metric", labels2)
        
        @test haskey(Prometheus.METRICS_REGISTRY, key1)
        @test haskey(Prometheus.METRICS_REGISTRY, key2)
        @test Prometheus.METRICS_REGISTRY[key1].value == 1024.0
        @test Prometheus.METRICS_REGISTRY[key2].value == 2048.0
    end
    
    @testset "HSOF Metrics Initialization" begin
        empty!(Prometheus.METRICS_REGISTRY)
        
        Prometheus.initialize_hsof_metrics()
        
        # Check that key metrics are initialized
        expected_metrics = [
            "hsof_gpu_utilization",
            "hsof_stage_duration_seconds", 
            "hsof_model_inference_duration_seconds",
            "hsof_mcts_nodes_explored_total",
            "hsof_features_selected",
            "hsof_database_queries_total"
        ]
        
        for metric_name in expected_metrics
            found = any(startswith(key, metric_name) for key in keys(Prometheus.METRICS_REGISTRY))
            @test found
        end
        
        @test length(Prometheus.METRICS_REGISTRY) > 20  # Should have many metrics
    end
    
    @testset "Prometheus Export Format" begin
        empty!(Prometheus.METRICS_REGISTRY)
        
        # Add some test metrics
        Prometheus.increment_counter!("http_requests_total", 100.0, Dict("status" => "200"))
        Prometheus.increment_counter!("http_requests_total", 5.0, Dict("status" => "404"))
        Prometheus.set_gauge!("memory_usage_bytes", 1073741824.0)
        
        output = Prometheus.export_metrics()
        
        # Check format compliance
        @test contains(output, "# HELP")
        @test contains(output, "# TYPE")
        @test contains(output, "http_requests_total")
        @test contains(output, "memory_usage_bytes")
        @test contains(output, "status=\"200\"")
        @test contains(output, "100")
        @test contains(output, "1.073741824e9")
    end
    
    @testset "GPU Metrics Update" begin
        if CUDA.functional()
            empty!(Prometheus.METRICS_REGISTRY)
            Prometheus.initialize_hsof_metrics()
            
            # Update GPU metrics
            Prometheus.update_gpu_metrics()
            
            # Check that GPU metrics were created
            gpu_metrics_found = any(contains(key, "hsof_gpu") for key in keys(Prometheus.METRICS_REGISTRY))
            @test gpu_metrics_found
            
            # Verify specific GPU metrics exist
            gpu_util_found = any(startswith(key, "hsof_gpu_utilization") for key in keys(Prometheus.METRICS_REGISTRY))
            @test gpu_util_found
        else
            @info "Skipping GPU tests - CUDA not functional"
        end
    end
    
    @testset "Convenience Recording Functions" begin
        empty!(Prometheus.METRICS_REGISTRY)
        Prometheus.initialize_hsof_metrics()
        
        # Test stage operation recording
        Prometheus.record_stage_operation(1, 5.5, 1000, false)
        
        # Check that stage metrics were updated
        stage_ops = any(contains(key, "hsof_stage_operations_total") && contains(key, "stage=\"1\"") 
                       for key in keys(Prometheus.METRICS_REGISTRY))
        @test stage_ops
        
        # Test model inference recording
        Prometheus.record_model_inference(0.045, 0.92, false)
        
        # Check model metrics
        model_inferences = any(startswith(key, "hsof_model_inference_total") 
                              for key in keys(Prometheus.METRICS_REGISTRY))
        @test model_inferences
        
        # Test MCTS recording
        Prometheus.record_mcts_metrics(500, 10, 0.85, 0.002)
        
        # Check MCTS metrics
        mcts_nodes = any(startswith(key, "hsof_mcts_nodes_explored_total") 
                        for key in keys(Prometheus.METRICS_REGISTRY))
        @test mcts_nodes
    end
    
    @testset "Integration Module" begin
        # Test monitoring state management
        @test !PrometheusIntegration.MONITORING_STATE.active
        
        # Start monitoring (without automatic updates to avoid background tasks in tests)
        PrometheusIntegration.start_monitoring(update_interval=1.0, auto_update_gpu=false)
        @test PrometheusIntegration.MONITORING_STATE.active
        
        # Test hook functions
        finish_func = PrometheusIntegration.report_stage_metrics(2, "test")
        @test isa(finish_func, Function)
        
        # Call finish function
        finish_func(500, false)
        
        # Stop monitoring
        PrometheusIntegration.stop_monitoring()
        @test !PrometheusIntegration.MONITORING_STATE.active
    end
    
    @testset "HTTP Metrics Endpoint" begin
        empty!(Prometheus.METRICS_REGISTRY)
        Prometheus.initialize_hsof_metrics()
        
        # Add some test data
        Prometheus.increment_counter!("test_requests", 42.0)
        Prometheus.set_gauge!("test_memory", 1024.0)
        
        # Create a mock HTTP request
        req = HTTP.Request("GET", "/metrics")
        
        # Test metrics handler
        response = Prometheus.metrics_handler(req)
        
        @test response.status == 200
        @test response.headers[1][2] == "text/plain; version=0.0.4; charset=utf-8"
        
        body = String(response.body)
        @test contains(body, "test_requests")
        @test contains(body, "test_memory")
        @test contains(body, "42")
        @test contains(body, "1024")
    end
end

@testset "Integration Hooks Tests" begin
    using .PrometheusIntegration.Hooks
    
    empty!(Prometheus.METRICS_REGISTRY)
    Prometheus.initialize_hsof_metrics()
    
    @testset "Stage Hooks" begin
        # Test stage start hook
        finish_func = on_stage_start(1)
        @test isa(finish_func, Function)
        
        # Test stage completion hook
        on_stage_complete(1, 2.5, 100)
        
        # Verify metrics were recorded
        stage_ops = any(contains(key, "hsof_stage_operations_total") && contains(key, "stage=\"1\"") 
                       for key in keys(Prometheus.METRICS_REGISTRY))
        @test stage_ops
        
        # Test stage error hook
        on_stage_error(1, ErrorException("test error"))
        
        # Verify error was recorded
        stage_errors = any(contains(key, "hsof_stage_errors_total") && contains(key, "stage=\"1\"") 
                          for key in keys(Prometheus.METRICS_REGISTRY))
        @test stage_errors
    end
    
    @testset "Model Hooks" begin
        # Test model inference hook
        on_model_inference(45.2, 0.89)
        
        # Test model loaded hook  
        on_model_loaded("v1.2.3")
        
        # Verify metrics were recorded
        model_loaded = any(startswith(key, "hsof_model_loaded") 
                          for key in keys(Prometheus.METRICS_REGISTRY))
        @test model_loaded
    end
    
    @testset "MCTS Hooks" begin
        # Test MCTS iteration hook
        on_mcts_iteration(250, 0.75)
        
        # Test feature evaluation hook
        on_feature_evaluation(50, 1000, 0.82)
        
        # Verify metrics were recorded
        mcts_nodes = any(startswith(key, "hsof_mcts_nodes_explored_total") 
                        for key in keys(Prometheus.METRICS_REGISTRY))
        @test mcts_nodes
        
        feature_selected = any(startswith(key, "hsof_features_selected") 
                              for key in keys(Prometheus.METRICS_REGISTRY))
        @test feature_selected
    end
    
    @testset "Database Hooks" begin
        # Test database query hook
        on_database_query(15.7, "SELECT")
        
        # Verify metrics were recorded
        db_queries = any(contains(key, "hsof_database_queries_total") 
                        for key in keys(Prometheus.METRICS_REGISTRY))
        @test db_queries
    end
end

println("All Prometheus metrics tests completed successfully!")