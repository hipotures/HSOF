# Test Configuration Loader

push!(LOAD_PATH, joinpath(@__DIR__, ".."))
include("../configs/config_loader.jl")
using Test

@testset "Configuration Loader Tests" begin
    # Test loading configurations
    @testset "Load Configurations" begin
        # Load development config
        config = ConfigLoader.load_configs("dev")
        @test config !== nothing
        @test config.environment == "dev"
        @test haskey(config.gpu_config, "cuda")
        @test haskey(config.algorithm_config, "mcts")
        @test haskey(config.data_config, "database")
    end
    
    # Test configuration access
    @testset "Configuration Access" begin
        gpu_config = ConfigLoader.get_gpu_config()
        @test haskey(gpu_config, "cuda")
        @test gpu_config["cuda"]["memory_limit_gb"] == 10  # Dev override
        
        algo_config = ConfigLoader.get_algorithm_config()
        @test haskey(algo_config, "mcts")
        @test algo_config["mcts"]["max_iterations"] == 10000
        
        data_config = ConfigLoader.get_data_config()
        @test haskey(data_config, "database")
        @test data_config["database"]["connection_pool_size"] == 4
    end
    
    # Test path-based access
    @testset "Path-based Access" begin
        # Valid paths
        @test ConfigLoader.get_config_value("gpu.cuda.memory_limit_gb") == 10
        @test ConfigLoader.get_config_value("algorithm.mcts.exploration_constant") ≈ 1.41421356
        @test ConfigLoader.get_config_value("data.database.read_only") == true
        
        # Invalid paths with defaults
        @test ConfigLoader.get_config_value("invalid.path", "default") == "default"
        @test ConfigLoader.get_config_value("gpu.invalid.key", 42) == 42
    end
    
    # Test configuration validation
    @testset "Configuration Validation" begin
        @test ConfigLoader.validate_config() == true
    end
    
    # Test configuration override
    @testset "Configuration Override" begin
        original_value = ConfigLoader.get_config_value("algorithm.mcts.max_iterations")
        
        # Override the value
        ConfigLoader.override_config!("algorithm.mcts.max_iterations", 5000)
        @test ConfigLoader.get_config_value("algorithm.mcts.max_iterations") == 5000
        
        # Restore original
        ConfigLoader.override_config!("algorithm.mcts.max_iterations", original_value)
        @test ConfigLoader.get_config_value("algorithm.mcts.max_iterations") == original_value
    end
    
    # Test production config
    @testset "Production Configuration" begin
        # Load production config
        prod_config = ConfigLoader.load_configs("prod")
        @test prod_config.environment == "prod"
        
        # Check production overrides
        @test prod_config.gpu_config["cuda"]["memory_limit_gb"] == 22
        @test prod_config.gpu_config["cuda"]["enable_profiling"] == false
        @test prod_config.gpu_config["cuda"]["math_mode"] == "FAST_MATH"
    end
    
    # Test runtime config save
    @testset "Save Runtime Config" begin
        temp_file = tempname() * ".toml"
        ConfigLoader.save_runtime_config(temp_file)
        
        @test isfile(temp_file)
        
        # Load and verify saved config
        using TOML
        saved_config = TOML.parsefile(temp_file)
        @test haskey(saved_config, "environment")
        @test haskey(saved_config, "timestamp")
        @test haskey(saved_config, "gpu")
        @test haskey(saved_config, "algorithm")
        @test haskey(saved_config, "data")
        
        # Cleanup
        rm(temp_file)
    end
end

# Print configuration summary
println("\n" * "=" ^ 60)
println("Configuration Summary")
println("=" ^ 60)

config = ConfigLoader.get_config()
println("Environment: ", config.environment)
println("\nKey Configuration Values:")
println("  GPU Memory Limit: ", ConfigLoader.get_config_value("gpu.cuda.memory_limit_gb"), " GB")
println("  MCTS Max Iterations: ", ConfigLoader.get_config_value("algorithm.mcts.max_iterations"))
println("  Ensemble Size: ", ConfigLoader.get_config_value("algorithm.mcts.ensemble_size"))
println("  Target Features: ", ConfigLoader.get_config_value("algorithm.filtering.target_features"))
println("  Metamodel Architecture: ", ConfigLoader.get_config_value("algorithm.metamodel.architecture"))
println("  Database Pool Size: ", ConfigLoader.get_config_value("data.database.connection_pool_size"))
println("  Chunk Size: ", ConfigLoader.get_config_value("data.data_loading.chunk_size"))
println("=" ^ 60)