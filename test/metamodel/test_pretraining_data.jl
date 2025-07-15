using Test
using Random
using Statistics
using HDF5

# Include the pre-training data module
include("../../src/metamodel/pretraining_data.jl")

using .PretrainingData

@testset "Pre-training Data Generation Tests" begin
    
    @testset "PretrainingConfig Creation" begin
        # Test default configuration
        config = create_pretraining_config()
        
        @test config.n_samples == 10000
        @test config.n_features == 500
        @test config.min_features == 10
        @test config.max_features == 100
        @test config.n_cv_folds == 5
        @test config.n_parallel_workers == Sys.CPU_THREADS
        @test config.use_xgboost == true
        @test config.use_random_forest == true
        @test config.augmentation_factor == 1.5
        
        # Test custom configuration
        custom_config = create_pretraining_config(
            n_samples = 5000,
            n_features = 200,
            min_features = 5,
            max_features = 50,
            n_parallel_workers = 2
        )
        
        @test custom_config.n_samples == 5000
        @test custom_config.n_features == 200
        @test custom_config.min_features == 5
        @test custom_config.max_features == 50
        @test custom_config.n_parallel_workers == 2
    end
    
    @testset "Synthetic Data Generation" begin
        n_samples = 1000
        n_features = 50
        
        X, y = generate_synthetic_data(n_samples, n_features)
        
        @test size(X) == (n_samples, n_features)
        @test length(y) == n_samples
        @test eltype(X) == Float32
        @test eltype(y) == Int
        @test all(y .∈ Ref([0, 1]))  # Binary classification
        
        # Check feature statistics
        @test all(isfinite.(X))
        
        # Check class balance (should be roughly balanced)
        class_balance = mean(y)
        @test 0.3 < class_balance < 0.7
    end
    
    @testset "Diverse Combinations Generation" begin
        config = create_pretraining_config(
            n_samples = 100,
            n_features = 50,
            min_features = 5,
            max_features = 20
        )
        
        combinations = generate_diverse_combinations(config, 100)
        
        @test length(combinations) == 100
        
        # Check feature counts are in range
        feature_counts = [length(combo) for combo in combinations]
        @test all(config.min_features .<= feature_counts .<= config.max_features)
        
        # Check indices are valid
        for combo in combinations
            @test all(1 .<= combo .<= config.n_features)
            @test issorted(combo)
            @test allunique(combo)
        end
        
        # Check diversity (different sizes should be represented)
        unique_sizes = unique(feature_counts)
        @test length(unique_sizes) > 5
    end
    
    @testset "Data Augmentation" begin
        config = create_pretraining_config(
            n_features = 50,
            min_features = 5,
            max_features = 20,
            augmentation_factor = 2.0
        )
        
        # Create initial combinations
        original_combos = [Int32.([1, 2, 3, 4, 5]),
                          Int32.([10, 11, 12, 13, 14, 15]),
                          Int32.([20, 21, 22, 23, 24, 25, 26, 27])]
        
        augmented = augment_combinations(original_combos, config)
        
        # Should have approximately double the original
        @test length(augmented) ≈ length(original_combos) * config.augmentation_factor rtol=0.2
        
        # Check all augmented combinations are valid
        for combo in augmented
            @test all(1 .<= combo .<= config.n_features)
            @test config.min_features <= length(combo) <= config.max_features
            @test issorted(combo)
            @test allunique(combo)
        end
        
        # Original combinations should still be present
        for orig in original_combos
            @test orig in augmented
        end
    end
    
    @testset "Feature Combination Structure" begin
        indices = Int32.([1, 5, 10, 15])
        xgb_score = 0.85f0
        rf_score = 0.82f0
        avg_score = 0.835f0
        n_features = Int32(4)
        gen_time = 1.5f0
        
        combo = FeatureCombination(
            indices,
            xgb_score,
            rf_score,
            avg_score,
            n_features,
            gen_time
        )
        
        @test combo.indices == indices
        @test combo.xgboost_score == xgb_score
        @test combo.rf_score == rf_score
        @test combo.avg_score == avg_score
        @test combo.n_features == n_features
        @test combo.generation_time == gen_time
    end
    
    @testset "HDF5 Save and Load" begin
        # Create test data
        results = [
            FeatureCombination(
                Int32.([1, 2, 3]),
                0.8f0, 0.75f0, 0.775f0,
                Int32(3), 0.1f0
            ),
            FeatureCombination(
                Int32.([5, 10, 15, 20, 25]),
                0.9f0, 0.85f0, 0.875f0,
                Int32(5), 0.2f0
            ),
            FeatureCombination(
                Int32.([2, 4, 6, 8]),
                0.7f0, 0.72f0, 0.71f0,
                Int32(4), 0.15f0
            )
        ]
        
        config = create_pretraining_config(
            n_samples = 3,
            n_features = 30,
            output_path = tempname() * ".h5"
        )
        
        # Save to HDF5
        save_to_hdf5(results, config)
        
        @test isfile(config.output_path)
        
        # Check file contents
        h5open(config.output_path, "r") do file
            @test haskey(attributes(file), "n_samples")
            @test read(attributes(file)["n_samples"]) == 3
            @test read(attributes(file)["n_features"]) == 30
            
            @test haskey(file, "feature_indices")
            @test haskey(file, "xgboost_scores")
            @test haskey(file, "rf_scores")
            @test haskey(file, "avg_scores")
            @test haskey(file, "n_features")
            @test haskey(file, "generation_times")
            
            # Check dimensions
            indices_data = read(file, "feature_indices")
            @test size(indices_data, 1) == 3
            
            scores = read(file, "avg_scores")
            @test length(scores) == 3
            @test scores ≈ [0.775f0, 0.875f0, 0.71f0]
        end
        
        # Load and verify
        loaded_results = load_pretraining_data(config.output_path)
        
        @test length(loaded_results) == length(results)
        
        for (orig, loaded) in zip(results, loaded_results)
            @test orig.indices == loaded.indices
            @test orig.xgboost_score ≈ loaded.xgboost_score
            @test orig.rf_score ≈ loaded.rf_score
            @test orig.avg_score ≈ loaded.avg_score
            @test orig.n_features == loaded.n_features
            @test orig.generation_time ≈ loaded.generation_time
        end
        
        # Clean up
        rm(config.output_path)
    end
    
    @testset "Small Scale Generation Test" begin
        # Test with very small dataset for speed
        config = create_pretraining_config(
            n_samples = 10,
            n_features = 20,
            min_features = 3,
            max_features = 8,
            n_cv_folds = 2,
            n_parallel_workers = 1,  # Sequential for testing
            use_xgboost = false,     # Only use RF for speed
            use_random_forest = true,
            augmentation_factor = 1.2,
            output_path = tempname() * ".h5"
        )
        
        # Generate small synthetic dataset
        X, y = generate_synthetic_data(100, config.n_features)
        
        # Generate pre-training data
        results = generate_pretraining_data(config; X=X, y=y)
        
        @test length(results) == config.n_samples
        @test isfile(config.output_path)
        
        # Check results
        for result in results
            @test length(result.indices) >= config.min_features
            @test length(result.indices) <= config.max_features
            @test 0 <= result.rf_score <= 1
            @test result.xgboost_score == 0.5f0  # Should be default since XGBoost disabled
            @test result.generation_time > 0
        end
        
        # Check score distribution
        avg_scores = [r.avg_score for r in results]
        @test 0.3 < mean(avg_scores) < 0.9  # Reasonable accuracy range
        @test std(avg_scores) > 0.01  # Some variation
        
        # Clean up
        rm(config.output_path)
    end
end

println("\n✅ Pre-training data generation tests completed!")