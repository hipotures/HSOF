using Test
using CUDA

# Include the GPU memory layout module
include("../../src/stage1_filter/gpu_memory_layout.jl")

using .GPUMemoryLayout

@testset "GPU Memory Layout Tests" begin
    
    # Skip tests if no GPU available
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU memory layout tests"
        return
    end
    
    @testset "Padding Calculations" begin
        # Test WARP_SIZE alignment
        @test calculate_padding(Int32(30), WARP_SIZE) == 32
        @test calculate_padding(Int32(32), WARP_SIZE) == 32
        @test calculate_padding(Int32(33), WARP_SIZE) == 64
        
        # Test 128-byte alignment
        @test calculate_padding(Int32(100), ALIGNMENT_BYTES) == 128
        @test calculate_padding(Int32(128), ALIGNMENT_BYTES) == 128
        @test calculate_padding(Int32(129), ALIGNMENT_BYTES) == 256
        
        # Test float4 alignment
        @test calculate_padding(Int32(3), Int32(4)) == 4
        @test calculate_padding(Int32(4), Int32(4)) == 4
        @test calculate_padding(Int32(5), Int32(4)) == 8
    end
    
    @testset "Feature Matrix Creation" begin
        n_samples = 1000
        n_features = 100
        
        fm = create_feature_matrix(n_samples, n_features)
        
        @test fm.n_samples == n_samples
        @test fm.n_features == n_features
        @test fm.padded_samples >= n_samples
        @test fm.padded_samples % WARP_SIZE == 0
        @test fm.padded_features >= n_features
        @test fm.padded_features % 4 == 0
        
        @test size(fm.data) == (fm.padded_samples, fm.padded_features)
        @test length(fm.feature_types) == n_features
        @test length(fm.feature_names) == n_features
        
        # Test with mixed feature types
        feature_types = [i % 3 == 0 ? CATEGORICAL : NUMERIC for i in 1:n_features]
        fm_mixed = create_feature_matrix(n_samples, n_features, feature_types)
        
        CUDA.@allowscalar begin
            for i in 1:n_features
                expected_type = i % 3 == 0 ? Int32(CATEGORICAL) : Int32(NUMERIC)
                @test fm_mixed.feature_types[i] == expected_type
            end
        end
    end
    
    @testset "Histogram Buffers Creation" begin
        n_features = 50
        
        hb = create_histogram_buffers(n_features)
        
        @test size(hb.feature_hist) == (HISTOGRAM_BINS, n_features)
        @test size(hb.target_hist) == (HISTOGRAM_BINS,)
        @test size(hb.joint_hist) == (HISTOGRAM_BINS, HISTOGRAM_BINS, n_features)
        
        @test size(hb.bin_edges) == (HISTOGRAM_BINS + 1, n_features)
        @test size(hb.target_bin_edges) == (HISTOGRAM_BINS + 1,)
        
        @test size(hb.feature_probs) == (HISTOGRAM_BINS, n_features)
        @test size(hb.target_probs) == (HISTOGRAM_BINS,)
        @test size(hb.joint_probs) == (HISTOGRAM_BINS, HISTOGRAM_BINS, n_features)
    end
    
    @testset "Correlation Matrix Creation" begin
        n_features = 100
        n_samples = 1000
        
        cm = create_correlation_matrix(n_features, n_samples)
        
        @test cm.n_features == n_features
        
        # Check upper triangle storage size
        expected_size = div(n_features * (n_features - 1), 2)
        @test length(cm.data) == expected_size
        
        @test length(cm.feature_means) == n_features
        @test length(cm.feature_stds) == n_features
        
        # Check standardized data has padded dimensions
        @test size(cm.standardized_data, 1) >= n_samples
        @test size(cm.standardized_data, 1) % 16 == 0
        @test size(cm.standardized_data, 2) >= n_features
        @test size(cm.standardized_data, 2) % 4 == 0
    end
    
    @testset "Variance Buffers Creation" begin
        n_features = 200
        n_blocks = 128
        
        vb = create_variance_buffers(n_features, n_blocks)
        
        @test length(vb.variances) == n_features
        @test length(vb.means) == n_features
        @test length(vb.m2_values) == n_features
        @test length(vb.counts) == n_features
        
        @test size(vb.block_sums) == (n_blocks, n_features)
        @test size(vb.block_counts) == (n_blocks, n_features)
    end
    
    @testset "Ranking Buffers Creation" begin
        n_features = 5000
        
        rb = create_ranking_buffers(n_features)
        
        @test length(rb.mi_scores) == n_features
        @test length(rb.feature_indices) == n_features
        @test length(rb.selected_mask) == n_features
        
        # Check feature indices are initialized correctly
        CUDA.@allowscalar begin
            for i in 1:min(10, n_features)
                @test rb.feature_indices[i] == i
            end
        end
        
        # Check correlation pairs storage
        max_pairs = div(n_features * (n_features - 1), 2)
        @test size(rb.correlation_pairs) == (2, max_pairs)
        @test length(rb.n_correlation_pairs) == 1
        
        @test length(rb.selected_indices) == TARGET_FEATURES
        @test length(rb.n_selected) == 1
    end
    
    @testset "Shared Memory Configuration" begin
        config = calculate_shared_memory_config()
        
        @test config.tile_rows == TILE_SIZE
        @test config.tile_cols == TILE_SIZE
        @test config.shared_bins <= HISTOGRAM_BINS
        @test config.reduction_size == TILE_SIZE * 2
        
        # Check total bytes calculation
        expected_tile = TILE_SIZE * TILE_SIZE * sizeof(Float32)
        expected_hist = config.shared_bins * sizeof(Int32)
        expected_reduction = config.reduction_size * sizeof(Float32)
        expected_total = expected_tile + expected_hist + expected_reduction
        
        @test config.total_bytes == expected_total
        
        # Typical shared memory limit is 48KB per block
        @test config.total_bytes <= 49152
    end
    
    @testset "Upper Triangle Indexing" begin
        n = Int32(5)
        
        # Test symmetric access
        @test get_upper_triangle_index(Int32(0), Int32(1), n) == 
              get_upper_triangle_index(Int32(1), Int32(0), n)
        
        # Test specific indices
        @test get_upper_triangle_index(Int32(0), Int32(1), n) == 0
        @test get_upper_triangle_index(Int32(0), Int32(2), n) == 1
        @test get_upper_triangle_index(Int32(0), Int32(3), n) == 2
        @test get_upper_triangle_index(Int32(0), Int32(4), n) == 3
        @test get_upper_triangle_index(Int32(1), Int32(2), n) == 4
        @test get_upper_triangle_index(Int32(1), Int32(3), n) == 5
        @test get_upper_triangle_index(Int32(1), Int32(4), n) == 6
        @test get_upper_triangle_index(Int32(2), Int32(3), n) == 7
        @test get_upper_triangle_index(Int32(2), Int32(4), n) == 8
        @test get_upper_triangle_index(Int32(3), Int32(4), n) == 9
        
        # Test larger matrix
        n_large = Int32(100)
        idx1 = get_upper_triangle_index(Int32(10), Int32(20), n_large)
        idx2 = get_upper_triangle_index(Int32(20), Int32(10), n_large)
        @test idx1 == idx2
        @test idx1 < div(n_large * (n_large - 1), 2)
    end
    
    @testset "Memory Usage Calculation" begin
        # Test with small dataset
        println("\nSmall dataset test:")
        print_memory_usage(10000, 1000)
        
        # Test with medium dataset
        println("\nMedium dataset test:")
        print_memory_usage(100000, 5000)
        
        # Test with large dataset (should trigger warning)
        println("\nLarge dataset test:")
        print_memory_usage(1000000, 5000)
    end
    
    @testset "Batch Processing Limits" begin
        # Test that batch size is reasonable
        @test MAX_SAMPLES_PER_BATCH == 100000
        
        # Calculate memory for one batch
        fm_batch = create_feature_matrix(MAX_SAMPLES_PER_BATCH, MAX_FEATURES)
        batch_bytes = sizeof(fm_batch.data)
        batch_gb = batch_bytes / (1024^3)
        
        println("\nBatch memory usage: $(round(batch_gb, digits=2)) GB")
        @test batch_gb < 8.0  # Should fit in 8GB
    end
    
    @testset "ThresholdConfig Default Values" begin
        config = ThresholdConfig(
            Float32(0.01),    # MI threshold
            Float32(0.95),    # Correlation threshold  
            Float32(1e-6),    # Variance threshold
            Int32(500)        # Target features
        )
        
        @test config.mi_threshold == Float32(0.01)
        @test config.correlation_threshold == Float32(0.95)
        @test config.variance_threshold == Float32(1e-6)
        @test config.target_features == 500
    end
end

println("\n✅ GPU memory layout tests completed!")