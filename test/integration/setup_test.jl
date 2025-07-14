# HSOF Integration Test Suite
# End-to-end tests for environment setup and component interactions

using Test
using HSOF
using CUDA
using DataFrames
using SQLite
using TOML
using Logging

# Set up test logging
test_logger = ConsoleLogger(stderr, Logging.Info)
global_logger(test_logger)

# Test results tracking
test_results = Dict{String, Bool}()

@testset "HSOF Integration Tests" begin
    
    @testset "Environment Setup" begin
        @info "Testing environment setup..."
        
        @testset "Julia Environment" begin
            @test VERSION >= v"1.9.0"
            test_results["julia_version"] = true
            
            # Test threading
            @test Threads.nthreads() > 1
            @info "Julia threads: $(Threads.nthreads())"
        end
        
        @testset "Package Loading" begin
            # Test all required packages can be loaded
            required_packages = [
                :CUDA, :BenchmarkTools, :DataFrames, :Flux,
                :JSON, :MLJ, :Test, :TOML, :Documenter
            ]
            
            for pkg in required_packages
                try
                    @eval using $pkg
                    @test true
                    test_results["package_$pkg"] = true
                catch e
                    @test false
                    test_results["package_$pkg"] = false
                    @error "Failed to load package $pkg" exception=e
                end
            end
        end
    end
    
    @testset "GPU Environment" begin
        @info "Testing GPU environment..."
        
        cuda_functional = CUDA.functional()
        @test cuda_functional skip=!cuda_functional
        test_results["cuda_functional"] = cuda_functional
        
        if cuda_functional
            @testset "GPU Detection" begin
                gpu_count = length(CUDA.devices())
                @test gpu_count >= 1
                @info "Found $gpu_count GPU(s)"
                test_results["gpu_count"] = gpu_count >= 1
                
                # Test each GPU
                for (i, dev) in enumerate(CUDA.devices())
                    CUDA.device!(dev)
                    
                    # Check memory
                    total_mem = CUDA.totalmem(dev) / 2^30
                    @test total_mem >= 8.0  # At least 8GB
                    @info "GPU $i: $(CUDA.name(dev)) - $(round(total_mem, digits=1)) GB"
                    
                    # Test basic operations
                    a = CUDA.rand(1000)
                    b = CUDA.rand(1000)
                    c = a .+ b
                    @test length(c) == 1000
                    @test eltype(c) == Float32
                    
                    # Test memory allocation/deallocation
                    initial_free = CUDA.available_memory()
                    large_array = CUDA.zeros(10_000, 10_000)
                    allocated = initial_free - CUDA.available_memory()
                    @test allocated > 0
                    
                    # Free memory
                    large_array = nothing
                    GC.gc()
                    CUDA.reclaim()
                    
                    final_free = CUDA.available_memory()
                    @test final_free > initial_free - 100_000_000  # Most memory reclaimed
                end
                
                # Test multi-GPU communication if available
                if gpu_count > 1
                    @testset "Multi-GPU Communication" begin
                        # Test peer access
                        CUDA.device!(0)
                        can_access = CUDA.can_access_peer(CUDA.CuDevice(1))
                        @info "Peer access GPU0→GPU1: $can_access"
                        
                        # Test data transfer
                        CUDA.device!(0)
                        a_gpu0 = CUDA.rand(1000)
                        
                        CUDA.device!(1)
                        a_gpu1 = CuArray(Array(a_gpu0))  # Transfer through host
                        
                        @test Array(a_gpu0) ≈ Array(a_gpu1)
                        test_results["multi_gpu"] = true
                    end
                else
                    test_results["multi_gpu"] = false
                end
            end
            
            @testset "CUDA Kernel Execution" begin
                # Test custom kernel
                function test_kernel(a, b, c)
                    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
                    if i <= length(c)
                        c[i] = a[i] * 2.0f0 + b[i]
                    end
                    return
                end
                
                n = 10000
                a = CUDA.rand(n)
                b = CUDA.rand(n)
                c = CUDA.zeros(n)
                
                @cuda threads=256 blocks=cld(n,256) test_kernel(a, b, c)
                synchronize()
                
                # Verify results
                c_expected = Array(a) .* 2.0f0 .+ Array(b)
                @test Array(c) ≈ c_expected rtol=1e-5
                
                test_results["cuda_kernels"] = true
            end
        else
            @warn "CUDA not functional, skipping GPU tests"
            test_results["cuda_functional"] = false
        end
    end
    
    @testset "Configuration System" begin
        @info "Testing configuration system..."
        
        @testset "Configuration Loading" begin
            # Test loading GPU config
            gpu_config_path = joinpath(@__DIR__, "../../configs/gpu_config.toml")
            if isfile(gpu_config_path)
                gpu_config = TOML.parsefile(gpu_config_path)
                
                @test haskey(gpu_config, "gpu")
                @test haskey(gpu_config["gpu"], "cuda")
                @test gpu_config["gpu"]["device_ids"] isa Vector
                
                test_results["gpu_config"] = true
            else
                @test_skip "GPU config file not found"
                test_results["gpu_config"] = false
            end
            
            # Test loading algorithm config
            algo_config_path = joinpath(@__DIR__, "../../configs/algorithm_config.toml")
            if isfile(algo_config_path)
                algo_config = TOML.parsefile(algo_config_path)
                
                @test haskey(algo_config, "algorithms")
                @test haskey(algo_config["algorithms"], "filtering")
                @test haskey(algo_config["algorithms"], "mcts")
                @test haskey(algo_config["algorithms"], "ensemble")
                
                test_results["algo_config"] = true
            else
                @test_skip "Algorithm config file not found"
                test_results["algo_config"] = false
            end
        end
        
        @testset "Configuration Validation" begin
            # Test config loader
            include("../../configs/config_loader.jl")
            
            # Load configs
            ConfigLoader.load_configs("dev")
            config = ConfigLoader.get_config()
            
            @test !isnothing(config)
            @test haskey(config, :gpu_config)
            @test haskey(config, :algorithm_config)
            
            # Validate GPU config
            @test config.gpu_config["device_ids"] isa Vector
            @test config.gpu_config["cuda"]["memory_limit_gb"] > 0
            
            # Validate algorithm config
            @test config.algorithm_config["filtering"]["variance_threshold"] > 0
            @test config.algorithm_config["mcts"]["n_iterations"] > 0
            
            test_results["config_validation"] = true
        end
    end
    
    @testset "Database Integration" begin
        @info "Testing database integration..."
        
        @testset "SQLite Connection" begin
            # Create test database
            test_db = "test_integration.db"
            db = SQLite.DB(test_db)
            
            # Create test table
            SQLite.execute(db, """
                CREATE TABLE IF NOT EXISTS feature_results (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    n_features INTEGER,
                    selected_features TEXT,
                    performance REAL
                )
            """)
            
            # Insert test data
            stmt = SQLite.Stmt(db, """
                INSERT INTO feature_results (timestamp, n_features, selected_features, performance)
                VALUES (?, ?, ?, ?)
            """)
            
            SQLite.execute(stmt, [string(now()), 100, "[1,2,3,4,5]", 0.95])
            
            # Query data
            results = DBInterface.execute(db, "SELECT COUNT(*) as count FROM feature_results") |> DataFrame
            @test results.count[1] >= 1
            
            # Clean up
            SQLite.close(db)
            rm(test_db, force=true)
            
            test_results["database"] = true
        end
    end
    
    @testset "Component Integration" begin
        @info "Testing component integration..."
        
        # Load GPU manager
        include("../../src/gpu/device_manager.jl")
        
        @testset "GPU Manager Integration" begin
            if CUDA.functional()
                # Initialize device manager
                gpu_manager = DeviceManager.initialize_devices()
                @test !isnothing(gpu_manager)
                
                # Get device info
                device_info = DeviceManager.get_device_info()
                @test device_info["device_count"] >= 1
                @test haskey(device_info, "devices")
                
                # Validate environment
                validation_results, issues = DeviceManager.validate_gpu_environment()
                @test validation_results["cuda_functional"]
                
                if !validation_results["all_checks_passed"]
                    @info "Validation issues: " issues
                end
                
                # Clean up
                DeviceManager.cleanup()
                
                test_results["gpu_manager"] = true
            else
                @test_skip "GPU manager requires CUDA"
                test_results["gpu_manager"] = false
            end
        end
        
        @testset "Memory Manager Integration" begin
            if CUDA.functional()
                # Test memory allocation tracking
                GPUManager.set_device!(0)
                
                # Allocate memory
                test_array = MemoryManager.allocate(Float32, 1000, 1000)
                @test size(test_array) == (1000, 1000)
                
                # Check memory stats
                stats = MemoryManager.get_memory_stats(0)
                @test haskey(stats, "used_gb")
                @test stats["used_gb"] > 0
                
                # Free memory
                MemoryManager.free(test_array)
                
                test_results["memory_manager"] = true
            else
                @test_skip "Memory manager requires CUDA"
                test_results["memory_manager"] = false
            end
        end
    end
    
    @testset "Pipeline Smoke Test" begin
        @info "Running pipeline smoke test..."
        
        @testset "Minimal Feature Selection" begin
            # Generate small test data
            n_samples = 100
            n_features = 50
            
            X = randn(n_samples, n_features)
            y = rand([0, 1], n_samples)
            
            # Create mock feature selection
            function mock_select_features(X, y)
                n_features = size(X, 2)
                n_selected = max(5, n_features ÷ 10)
                
                # Simulate selection
                selected_indices = sort(randperm(n_features)[1:n_selected])
                feature_scores = rand(length(selected_indices))
                
                return (
                    selected_indices = selected_indices,
                    feature_scores = feature_scores,
                    computation_time = rand() * 10
                )
            end
            
            # Run mock pipeline
            results = mock_select_features(X, y)
            
            @test length(results.selected_indices) >= 5
            @test length(results.selected_indices) <= n_features
            @test all(1 .<= results.selected_indices .<= n_features)
            @test length(results.feature_scores) == length(results.selected_indices)
            @test results.computation_time > 0
            
            @info "Selected $(length(results.selected_indices)) features from $n_features"
            
            test_results["pipeline_smoke"] = true
        end
    end
    
    # Summary
    @testset "Test Summary" begin
        @info "Integration test summary:"
        
        passed = count(values(test_results))
        total = length(test_results)
        
        println("\nTest Results:")
        println("="^40)
        for (test, result) in sort(collect(test_results))
            status = result ? "✓" : "✗"
            color = result ? :green : :red
            printstyled("  $test: $status\n"; color=color)
        end
        println("="^40)
        println("Passed: $passed/$total")
        
        @test passed == total
    end
end

# Return exit code based on test results
all_passed = all(values(test_results))
exit(all_passed ? 0 : 1)