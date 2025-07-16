using Test
using CUDA
using Statistics
using JSON3

# Include the threshold management module
include("../../src/stage1_filter/threshold_management.jl")

using .ThresholdManagement
using .ThresholdManagement.GPUMemoryLayout: ThresholdConfig

# Create temporary directory for config files
test_config_dir = mktempdir()

@testset "Threshold Management Tests" begin
    
    @testset "Default Configuration Creation" begin
        config = create_default_config()
        
        @test config.base_config.mi_threshold == Float32(0.01)
        @test config.base_config.correlation_threshold == Float32(0.95)
        @test config.base_config.variance_threshold == Float32(1e-6)
        @test config.base_config.target_features == 500
        
        @test config.adaptive_mi_threshold == config.base_config.mi_threshold
        @test config.adaptive_corr_threshold == config.base_config.correlation_threshold
        @test config.adaptive_var_threshold == config.base_config.variance_threshold
        
        @test config.mi_threshold_range == (Float32(0.0), Float32(1.0))
        @test config.corr_threshold_range == (Float32(0.5), Float32(0.999))
        @test config.var_threshold_range == (Float32(1e-10), Float32(0.1))
        
        @test config.adaptation_rate == Float32(0.1)
        @test config.min_features_buffer == 50
        @test config.max_iterations == 20
        @test config.validate_feature_count == true
        @test config.strict_mode == false
    end
    
    @testset "Configuration Validation" begin
        config = create_default_config()
        
        # Valid configuration should pass
        @test validate_config!(config) == true
        
        # Test clamping of thresholds
        config.adaptive_mi_threshold = Float32(-0.5)
        validate_config!(config)
        @test config.adaptive_mi_threshold == Float32(0.0)
        
        config.adaptive_mi_threshold = Float32(1.5)
        validate_config!(config)
        @test config.adaptive_mi_threshold == Float32(1.0)
        
        # Test invalid configurations
        config.base_config = ThresholdConfig(
            Float32(0.01), Float32(0.95), Float32(1e-6), Int32(6000)
        )
        @test_throws ErrorException validate_config!(config)
        
        config.base_config = ThresholdConfig(
            Float32(0.01), Float32(0.95), Float32(1e-6), Int32(0)
        )
        @test_throws ErrorException validate_config!(config)
        
        # Test adaptation rate validation
        config = create_default_config()
        config.adaptation_rate = Float32(0.0)
        @test_throws ErrorException validate_config!(config)
        
        config.adaptation_rate = Float32(1.5)
        @test_throws ErrorException validate_config!(config)
    end
    
    @testset "Configuration Save/Load" begin
        config = create_default_config()
        
        # Modify some values
        config.adaptive_mi_threshold = Float32(0.02)
        config.adaptive_corr_threshold = Float32(0.9)
        config.strict_mode = true
        
        # Save configuration
        config_file = joinpath(test_config_dir, "test_config.json")
        save_config(config, config_file)
        
        @test isfile(config_file)
        
        # Load configuration
        loaded_config = load_config(config_file)
        
        @test loaded_config.base_config.mi_threshold == config.base_config.mi_threshold
        @test loaded_config.base_config.correlation_threshold == config.base_config.correlation_threshold
        @test loaded_config.base_config.variance_threshold == config.base_config.variance_threshold
        @test loaded_config.base_config.target_features == config.base_config.target_features
        @test loaded_config.strict_mode == config.strict_mode
        
        # Test loading non-existent file
        non_existent_config = load_config(joinpath(test_config_dir, "non_existent.json"))
        @test non_existent_config.base_config.mi_threshold == Float32(0.01)  # Should use defaults
    end
    
    @testset "Runtime Configuration Updates" begin
        config = create_default_config()
        
        # Update MI threshold
        update_runtime_config!(config, mi_threshold=Float32(0.05))
        @test config.base_config.mi_threshold == Float32(0.05)
        @test config.adaptive_mi_threshold == Float32(0.05)
        
        # Update correlation threshold
        update_runtime_config!(config, correlation_threshold=Float32(0.8))
        @test config.base_config.correlation_threshold == Float32(0.8)
        @test config.adaptive_corr_threshold == Float32(0.8)
        
        # Update variance threshold
        update_runtime_config!(config, variance_threshold=Float32(1e-5))
        @test config.base_config.variance_threshold == Float32(1e-5)
        @test config.adaptive_var_threshold == Float32(1e-5)
        
        # Update target features
        update_runtime_config!(config, target_features=Int32(400))
        @test config.base_config.target_features == 400
        
        # Update adaptation parameters
        update_runtime_config!(config, adaptation_rate=Float32(0.2), strict_mode=true)
        @test config.adaptation_rate == Float32(0.2)
        @test config.strict_mode == true
        
        # Test multiple updates at once
        update_runtime_config!(
            config,
            mi_threshold=Float32(0.03),
            correlation_threshold=Float32(0.85),
            target_features=Int32(450)
        )
        @test config.base_config.mi_threshold == Float32(0.03)
        @test config.base_config.correlation_threshold == Float32(0.85)
        @test config.base_config.target_features == 450
    end
    
    if CUDA.functional()
        @testset "Threshold Statistics Calculation" begin
            n_features = 1000
            
            # Create test data
            mi_scores = CUDA.rand(Float32, n_features)
            correlations = CUDA.rand(Float32, n_features) * 0.5f0 .+ 0.5f0  # [0.5, 1.0]
            variances = CUDA.rand(Float32, n_features) * 0.1f0  # [0, 0.1]
            
            stats = calculate_threshold_stats(mi_scores, correlations, variances)
            
            @test length(stats.mi_percentiles) == 9
            @test length(stats.corr_percentiles) == 9
            @test length(stats.var_percentiles) == 9
            
            # Check percentiles are ordered
            for i in 2:9
                @test stats.mi_percentiles[i] >= stats.mi_percentiles[i-1]
                @test stats.corr_percentiles[i] >= stats.corr_percentiles[i-1]
                @test stats.var_percentiles[i] >= stats.var_percentiles[i-1]
            end
            
            # Check ranges
            @test all(p >= 0.0f0 && p <= 1.0f0 for p in stats.mi_percentiles)
            @test all(p >= 0.5f0 && p <= 1.0f0 for p in stats.corr_percentiles)
            @test all(p >= 0.0f0 && p <= 0.1f0 for p in stats.var_percentiles)
        end
        
        @testset "Feature Counting" begin
            n_features = 1000
            config = create_default_config()
            
            # Create test data with known distribution
            mi_scores = CUDA.zeros(Float32, n_features)
            variances = CUDA.ones(Float32, n_features) * 0.01f0
            
            # Set half the features to pass MI threshold
            CUDA.@allowscalar begin
                for i in 1:500
                    mi_scores[i] = 0.02f0  # Above default threshold of 0.01
                end
            end
            
            count = count_passing_features(mi_scores, variances, config)
            @test count == 500
            
            # Test with different thresholds
            config.adaptive_mi_threshold = Float32(0.015)
            count = count_passing_features(mi_scores, variances, config)
            @test count == 500  # Still 500 features above 0.015
            
            config.adaptive_mi_threshold = Float32(0.025)
            count = count_passing_features(mi_scores, variances, config)
            @test count == 0  # No features above 0.025
            
            # Test variance threshold
            config.adaptive_mi_threshold = Float32(0.01)
            config.adaptive_var_threshold = Float32(0.02)
            count = count_passing_features(mi_scores, variances, config)
            @test count == 0  # No features pass variance threshold
        end
        
        @testset "Adaptive Threshold Adjustment" begin
            n_features = 1000
            config = create_default_config()
            config.base_config = ThresholdConfig(
                config.base_config.mi_threshold,
                config.base_config.correlation_threshold,
                config.base_config.variance_threshold,
                Int32(300)  # Target 300 features
            )
            
            # Create synthetic stats
            mi_scores = CUDA.rand(Float32, n_features)
            correlations = CUDA.rand(Float32, n_features)
            variances = CUDA.rand(Float32, n_features) * 0.01f0
            
            stats = calculate_threshold_stats(mi_scores, correlations, variances)
            
            # Test adjustment when we have too many features
            current_count = Int32(500)
            converged = adjust_thresholds_for_count!(config, current_count, stats)
            @test converged == false  # Should need adjustment
            @test config.adaptive_mi_threshold > config.base_config.mi_threshold  # Should increase
            
            # Test adjustment when we have too few features
            current_count = Int32(100)
            old_mi_threshold = config.adaptive_mi_threshold
            converged = adjust_thresholds_for_count!(config, current_count, stats)
            @test converged == false  # Should need adjustment
            @test config.adaptive_mi_threshold < old_mi_threshold  # Should decrease
            
            # Test exact match
            current_count = Int32(300)
            converged = adjust_thresholds_for_count!(config, current_count, stats)
            @test converged == true  # Should be done
            
            # Test max iterations
            config.max_iterations = Int32(2)
            config.strict_mode = true
            current_count = Int32(100)
            @test_throws ErrorException begin
                for i in 1:3
                    adjust_thresholds_for_count!(config, current_count, stats, Int32(i))
                end
            end
            
            # Test non-strict mode
            config.strict_mode = false
            converged = adjust_thresholds_for_count!(config, current_count, stats, Int32(3))
            @test converged == false  # Should return false but not error
        end
    else
        @warn "CUDA not functional, skipping GPU tests"
    end
    
    @testset "Constant Memory Configuration" begin
        config = create_default_config()
        config.adaptive_mi_threshold = Float32(0.02)
        config.adaptive_corr_threshold = Float32(0.9)
        config.adaptive_var_threshold = Float32(1e-5)
        
        const_config = create_constant_memory_config(config)
        
        @test const_config[1] == Float32(0.02)
        @test const_config[2] == Float32(0.9)
        @test const_config[3] == Float32(1e-5)
        @test const_config[4] == Int32(500)
    end
    
    @testset "Configuration Logging" begin
        config = create_default_config()
        
        # Just test that logging doesn't error
        log_config(config)
        log_config(config, "  ")  # With prefix
        
        # Modify and log again
        config.adaptive_mi_threshold = Float32(0.05)
        config.strict_mode = true
        log_config(config)
        
        @test true  # If we get here, logging worked
    end
end

# Cleanup
rm(test_config_dir, recursive=true)

println("\n✅ Threshold management tests completed!")