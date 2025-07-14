using Test
using CUDA
using Random

# Include the categorical features module
include("../../src/stage1_filter/categorical_features.jl")

using .CategoricalFeatures
using .CategoricalFeatures.GPUMemoryLayout

@testset "Categorical Features Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU categorical feature tests"
        return
    end
    
    @testset "CategoricalConfig Creation" begin
        config = create_categorical_config(100, 1000)
        
        @test config.n_features == 100
        @test config.n_samples == 1000
        @test config.n_categorical == 0
        @test config.max_categories == 100
        @test config.encoding_type == :onehot
        @test config.sparse_threshold == Float32(0.1)
        @test config.use_gpu_encoding == true
        
        # Test with custom parameters
        config2 = create_categorical_config(50, 500,
                                          n_categorical=10,
                                          encoding_type=:ordinal,
                                          max_categories=20)
        
        @test config2.n_categorical == 10
        @test config2.encoding_type == :ordinal
        @test config2.max_categories == 20
    end
    
    @testset "Categorical Feature Detection" begin
        n_samples = 1000
        n_features = 20
        
        # Create test data with mixed numerical and categorical features
        data = CUDA.zeros(Float32, n_samples, n_features)
        
        # Features 1-5: continuous numerical
        data[:, 1:5] = CUDA.randn(Float32, n_samples, 5)
        
        # Features 6-10: categorical with few categories
        for j in 6:10
            n_cats = rand(3:10)
            data[:, j] = Float32.(rand(0:(n_cats-1), n_samples))
        end
        
        # Features 11-15: categorical with many categories (high cardinality)
        for j in 11:15
            n_cats = rand(40:60)
            data[:, j] = Float32.(rand(0:(n_cats-1), n_samples))
        end
        
        # Features 16-20: more continuous numerical
        data[:, 16:20] = CUDA.randn(Float32, n_samples, 5) .* 100
        
        # Detect categorical features
        config = create_categorical_config(n_features, n_samples)
        cat_indices, cat_counts = detect_categorical_features(data, config, categorical_threshold=50)
        
        @test length(cat_indices) >= 5  # Should detect at least features 6-10
        @test all(6 .<= cat_indices .<= 15)  # Should be in the categorical range
        @test all(cat_counts .>= 3)  # Should have at least 3 categories each
    end
    
    @testset "One-Hot Encoding" begin
        n_samples = 100
        n_features = 5
        
        # Create simple categorical data
        cat_data = CUDA.zeros(Float32, n_samples, n_features)
        
        # Feature 1: binary (0, 1)
        cat_data[:, 1] = Float32.(rand(0:1, n_samples))
        
        # Feature 2: 3 categories (0, 1, 2)
        cat_data[:, 2] = Float32.(rand(0:2, n_samples))
        
        # Feature 3: 5 categories (0-4)
        cat_data[:, 3] = Float32.(rand(0:4, n_samples))
        
        # Features 4-5: continuous (should not be detected as categorical)
        cat_data[:, 4:5] = CUDA.randn(Float32, n_samples, 2)
        
        # Detect and encode
        config = create_categorical_config(n_features, n_samples, encoding_type=:onehot)
        cat_indices, cat_counts = detect_categorical_features(cat_data, config, categorical_threshold=10)
        
        # Should detect first 3 features
        @test length(cat_indices) == 3
        @test cat_indices == Int32[1, 2, 3]
        @test cat_counts == Int32[2, 3, 5]  # Binary, 3-cat, 5-cat
        
        # Create metadata
        metadata = create_categorical_metadata(cat_indices, cat_counts, :onehot)
        
        @test metadata.total_encoded_features == 2 + 3 + 5  # Total one-hot columns
        @test Array(metadata.encoding_offsets) == Int32[2, 5, 10]
        
        # Encode features
        feature_matrix = create_feature_matrix(n_samples, n_features)
        copyto!(feature_matrix.data[1:n_samples, 1:n_features], cat_data)
        
        encoded_matrix = create_feature_matrix(n_samples, metadata.total_encoded_features)
        encode_categorical_features!(encoded_matrix, feature_matrix, metadata, config)
        
        # Test encoding correctness
        encoded_cpu = Array(encoded_matrix.data[1:n_samples, 1:metadata.total_encoded_features])
        
        # Check that each row has exactly one 1 per original categorical feature
        for i in 1:n_samples
            # Feature 1 (binary): columns 1-2
            @test sum(encoded_cpu[i, 1:2]) ≈ 1.0
            
            # Feature 2 (3-cat): columns 3-5
            @test sum(encoded_cpu[i, 3:5]) ≈ 1.0
            
            # Feature 3 (5-cat): columns 6-10
            @test sum(encoded_cpu[i, 6:10]) ≈ 1.0
        end
    end
    
    @testset "Ordinal Encoding" begin
        n_samples = 100
        n_features = 3
        
        # Create ordinal data
        cat_data = CUDA.zeros(Float32, n_samples, n_features)
        
        # Feature 1: Low (0), Medium (1), High (2)
        cat_data[:, 1] = Float32.(rand(0:2, n_samples))
        
        # Feature 2: 5 ordinal levels (0-4)
        cat_data[:, 2] = Float32.(rand(0:4, n_samples))
        
        # Feature 3: 10 ordinal levels (0-9)
        cat_data[:, 3] = Float32.(rand(0:9, n_samples))
        
        # Setup for ordinal encoding
        config = create_categorical_config(n_features, n_samples, 
                                         n_categorical=n_features,
                                         encoding_type=:ordinal)
        
        cat_indices = Int32[1, 2, 3]
        cat_counts = Int32[3, 5, 10]
        
        metadata = create_categorical_metadata(cat_indices, cat_counts, :ordinal)
        
        @test metadata.total_encoded_features == 3  # Ordinal keeps same dimensionality
        
        # Encode features
        feature_matrix = create_feature_matrix(n_samples, n_features)
        copyto!(feature_matrix.data[1:n_samples, 1:n_features], cat_data)
        
        encoded_matrix = create_feature_matrix(n_samples, n_features)
        encode_categorical_features!(encoded_matrix, feature_matrix, metadata, config)
        
        # Test encoding correctness
        encoded_cpu = Array(encoded_matrix.data[1:n_samples, 1:n_features])
        cat_data_cpu = Array(cat_data)
        
        for i in 1:n_samples
            # Feature 1: should map 0->0, 1->0.5, 2->1.0
            expected1 = cat_data_cpu[i, 1] / 2.0
            @test encoded_cpu[i, 1] ≈ expected1
            
            # Feature 2: should map 0->0, ..., 4->1.0
            expected2 = cat_data_cpu[i, 2] / 4.0
            @test encoded_cpu[i, 2] ≈ expected2
            
            # Feature 3: should map 0->0, ..., 9->1.0
            expected3 = cat_data_cpu[i, 3] / 9.0
            @test encoded_cpu[i, 3] ≈ expected3
        end
    end
    
    @testset "Mixed Feature Matrix" begin
        n_samples = 200
        n_numerical = 10
        n_categorical = 5
        
        # Create numerical features
        numerical_data = CUDA.randn(Float32, n_samples, n_numerical)
        
        # Create categorical features
        categorical_data = CUDA.zeros(Float32, n_samples, n_categorical)
        for j in 1:n_categorical
            n_cats = rand(2:5)
            categorical_data[:, j] = Float32.(rand(0:(n_cats-1), n_samples))
        end
        
        # Create metadata for categorical features
        cat_indices = collect(Int32, 1:n_categorical)
        cat_counts = Int32[2, 3, 4, 5, 3]  # Different category counts
        
        metadata = create_categorical_metadata(cat_indices, cat_counts, :onehot)
        
        # Create mixed feature matrix
        config = create_categorical_config(n_categorical, n_samples, 
                                         n_categorical=n_categorical,
                                         encoding_type=:onehot)
        
        mixed_matrix = create_mixed_feature_matrix(numerical_data, categorical_data, metadata, config)
        
        # Check dimensions
        total_features = n_numerical + metadata.total_encoded_features
        @test mixed_matrix.n_features == total_features
        @test mixed_matrix.n_samples == n_samples
        
        # Check that numerical features are preserved
        mixed_data_cpu = Array(mixed_matrix.data[1:n_samples, 1:total_features])
        numerical_cpu = Array(numerical_data)
        
        @test mixed_data_cpu[:, 1:n_numerical] ≈ numerical_cpu
        
        # Check that categorical features are encoded
        encoded_start = n_numerical + 1
        encoded_end = n_numerical + metadata.total_encoded_features
        encoded_part = mixed_data_cpu[:, encoded_start:encoded_end]
        
        # Each row should have exactly n_categorical ones
        for i in 1:n_samples
            @test sum(encoded_part[i, :]) ≈ Float32(n_categorical)
        end
    end
    
    @testset "Sparse One-Hot Representation" begin
        n_samples = 1000
        n_categorical = 5
        
        # Create categorical data
        categorical_data = CUDA.zeros(Float32, n_samples, n_categorical)
        for j in 1:n_categorical
            categorical_data[:, j] = Float32.(rand(0:4, n_samples))  # 5 categories each
        end
        
        # Create metadata
        cat_indices = collect(Int32, 1:n_categorical)
        cat_counts = fill(Int32(5), n_categorical)
        metadata = create_categorical_metadata(cat_indices, cat_counts, :onehot)
        
        config = create_categorical_config(n_categorical, n_samples,
                                         n_categorical=n_categorical,
                                         encoding_type=:onehot)
        
        # Build sparse representation
        sparse_matrix = build_sparse_onehot(categorical_data, metadata, config)
        
        @test sparse_matrix.n_rows == n_samples
        @test sparse_matrix.n_cols == metadata.total_encoded_features
        @test sparse_matrix.nnz == n_samples * n_categorical
        
        # Verify CSR structure
        row_ptr_cpu = Array(sparse_matrix.row_ptr)
        @test row_ptr_cpu[1] == 0
        @test row_ptr_cpu[end] == sparse_matrix.nnz
        
        # Each row should have n_categorical non-zeros
        for i in 1:n_samples
            nnz_in_row = row_ptr_cpu[i+1] - row_ptr_cpu[i]
            @test nnz_in_row == n_categorical
        end
    end
    
    @testset "Feature Selection with Mixed Data" begin
        n_samples = 100
        n_numerical = 20
        n_categorical = 5
        
        # Create test data
        numerical_data = CUDA.randn(Float32, n_samples, n_numerical)
        categorical_data = CUDA.zeros(Float32, n_samples, n_categorical)
        
        for j in 1:n_categorical
            categorical_data[:, j] = Float32.(rand(0:2, n_samples))  # 3 categories each
        end
        
        # Create metadata
        cat_indices = collect(Int32, 1:n_categorical)
        cat_counts = fill(Int32(3), n_categorical)
        metadata = create_categorical_metadata(cat_indices, cat_counts, :onehot)
        
        config = create_categorical_config(n_categorical, n_samples,
                                         n_categorical=n_categorical,
                                         encoding_type=:onehot)
        
        # Create mixed matrix
        mixed_matrix = create_mixed_feature_matrix(numerical_data, categorical_data, metadata, config)
        
        # Simulate feature selection
        n_selected = 10
        total_features = n_numerical + metadata.total_encoded_features
        
        # Select some numerical and some encoded categorical features
        selected_indices = CuArray(Int32[1, 5, 10, 15, 20,  # Numerical
                                        21, 22, 25, 30, 35])  # Encoded categorical
        
        # Apply selection
        numerical_selected, categorical_selected = apply_selection_mixed_features(
            mixed_matrix, selected_indices, n_selected, n_numerical, metadata
        )
        
        @test length(numerical_selected) == 5
        @test all(numerical_selected .<= n_numerical)
        
        # Categorical selections should map back to original indices
        @test length(categorical_selected) >= 1
        @test all(categorical_selected .<= n_categorical)
    end
end

println("\n✅ Categorical features tests completed!")