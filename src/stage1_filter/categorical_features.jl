module CategoricalFeatures

using CUDA
using CUDA.CUSPARSE
using Statistics

# Include the GPUMemoryLayout module
include("gpu_memory_layout.jl")
using .GPUMemoryLayout: FeatureMatrix, WARP_SIZE, TILE_SIZE, MAX_FEATURES, create_feature_matrix

"""
Configuration for categorical feature handling
"""
struct CategoricalConfig
    n_features::Int32              # Total number of features (including encoded)
    n_samples::Int32               # Number of samples
    n_categorical::Int32           # Number of categorical features
    max_categories::Int32          # Maximum categories per feature
    encoding_type::Symbol          # :onehot, :ordinal, :target
    sparse_threshold::Float32      # Threshold for sparse representation
    use_gpu_encoding::Bool         # Use GPU for encoding (vs CPU preprocessing)
end

"""
Create default categorical configuration
"""
function create_categorical_config(n_features::Integer, n_samples::Integer;
                                 n_categorical::Integer = 0,
                                 max_categories::Integer = 100,
                                 encoding_type::Symbol = :onehot,
                                 sparse_threshold::Float32 = Float32(0.1),
                                 use_gpu_encoding::Bool = true)
    return CategoricalConfig(
        Int32(n_features),
        Int32(n_samples),
        Int32(n_categorical),
        Int32(max_categories),
        encoding_type,
        sparse_threshold,
        use_gpu_encoding
    )
end

"""
Structure to hold categorical feature metadata
"""
struct CategoricalMetadata
    feature_indices::CuArray{Int32, 1}    # Indices of categorical features
    category_counts::CuArray{Int32, 1}    # Number of categories per feature
    category_offsets::CuArray{Int32, 1}   # Offsets in category mapping
    category_mapping::CuArray{Int32, 1}   # Flattened category ID mapping
    encoding_offsets::CuArray{Int32, 1}   # Offsets in encoded feature space
    total_encoded_features::Int32          # Total features after encoding
end

"""
Structure for sparse one-hot encoding on GPU
"""
struct SparseOneHotMatrix
    row_ptr::CuArray{Int32, 1}      # CSR row pointers
    col_idx::CuArray{Int32, 1}      # Column indices
    values::CuArray{Float32, 1}     # Non-zero values (all 1.0 for one-hot)
    n_rows::Int32
    n_cols::Int32
    nnz::Int32                       # Number of non-zeros
end

"""
GPU kernel to identify categorical features and count categories
"""
function count_categories_kernel!(
    category_counts::CuDeviceArray{Int32, 1},
    is_categorical::CuDeviceArray{Bool, 1},
    feature_data::CuDeviceArray{Float32, 2},
    n_samples::Int32,
    n_features::Int32,
    categorical_threshold::Int32
)
    feature_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if feature_idx <= n_features
        # Count unique values for this feature
        unique_count = Int32(0)
        
        # Simple approach: check if feature has integer values and count uniques
        # In practice, would use more sophisticated detection
        all_integers = true
        max_val = -Inf32
        
        for sample_idx in 1:n_samples
            val = feature_data[sample_idx, feature_idx]
            if val != floor(val)
                all_integers = false
                break
            end
            max_val = max(max_val, val)
        end
        
        if all_integers && max_val < Float32(categorical_threshold)
            # Count unique integer values
            # Simplified: assume values are 0 to max_val
            unique_count = Int32(ceil(max_val)) + 1
            is_categorical[feature_idx] = true
            category_counts[feature_idx] = unique_count
        else
            is_categorical[feature_idx] = false
            category_counts[feature_idx] = 0
        end
    end
    
    return nothing
end

"""
GPU kernel for one-hot encoding of categorical features
"""
function encode_onehot_kernel!(
    encoded_data::CuDeviceArray{Float32, 2},
    categorical_data::CuDeviceArray{Float32, 2},
    feature_indices::CuDeviceArray{Int32, 1},
    category_counts::CuDeviceArray{Int32, 1},
    encoding_offsets::CuDeviceArray{Int32, 1},
    n_samples::Int32,
    n_categorical::Int32
)
    sample_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    cat_feature_idx = blockIdx().y
    
    if sample_idx <= n_samples && cat_feature_idx <= n_categorical
        # Get original feature index
        feature_idx = feature_indices[cat_feature_idx]
        
        # Get category value
        cat_value = Int32(categorical_data[sample_idx, feature_idx])
        
        # Get encoding offset for this feature
        encoding_offset = cat_feature_idx == 1 ? Int32(0) : encoding_offsets[cat_feature_idx - 1]
        n_categories = category_counts[cat_feature_idx]
        
        # One-hot encode
        for cat_id in 1:n_categories
            encoded_idx = encoding_offset + cat_id
            if cat_id == cat_value + 1  # 0-indexed to 1-indexed
                encoded_data[sample_idx, encoded_idx] = 1.0f0
            else
                encoded_data[sample_idx, encoded_idx] = 0.0f0
            end
        end
    end
    
    return nothing
end

"""
GPU kernel for ordinal encoding
"""
function encode_ordinal_kernel!(
    encoded_data::CuDeviceArray{Float32, 2},
    categorical_data::CuDeviceArray{Float32, 2},
    feature_indices::CuDeviceArray{Int32, 1},
    category_counts::CuDeviceArray{Int32, 1},
    ordinal_mapping::CuDeviceArray{Float32, 2},  # [max_categories × n_categorical]
    n_samples::Int32,
    n_categorical::Int32
)
    sample_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    cat_feature_idx = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    
    if sample_idx <= n_samples && cat_feature_idx <= n_categorical
        feature_idx = feature_indices[cat_feature_idx]
        cat_value = Int32(categorical_data[sample_idx, feature_idx])
        
        # Look up ordinal value
        if cat_value >= 0 && cat_value < category_counts[cat_feature_idx]
            ordinal_value = ordinal_mapping[cat_value + 1, cat_feature_idx]
            encoded_data[sample_idx, cat_feature_idx] = ordinal_value
        else
            encoded_data[sample_idx, cat_feature_idx] = 0.0f0  # Default for unknown
        end
    end
    
    return nothing
end

"""
Build sparse CSR representation for one-hot encoded features
"""
function build_sparse_onehot(
    categorical_data::CuArray{Float32, 2},
    metadata::CategoricalMetadata,
    config::CategoricalConfig
)
    n_samples = config.n_samples
    n_categorical = config.n_categorical
    total_encoded = metadata.total_encoded_features
    
    # Count non-zeros (one per sample per categorical feature)
    nnz = n_samples * n_categorical
    
    # Allocate CSR arrays
    row_ptr = CUDA.zeros(Int32, n_samples + 1)
    col_idx = CUDA.zeros(Int32, nnz)
    values = CUDA.ones(Float32, nnz)
    
    # Build CSR structure on GPU
    threads = 256
    blocks = cld(n_samples, threads)
    
    function build_csr_kernel!(
        row_ptr::CuDeviceArray{Int32, 1},
        col_idx::CuDeviceArray{Int32, 1},
        categorical_data::CuDeviceArray{Float32, 2},
        feature_indices::CuDeviceArray{Int32, 1},
        encoding_offsets::CuDeviceArray{Int32, 1},
        n_samples::Int32,
        n_categorical::Int32
    )
        sample_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        
        if sample_idx <= n_samples
            # Each sample has n_categorical non-zeros
            row_ptr[sample_idx + 1] = sample_idx * n_categorical
            
            # Fill column indices
            nnz_idx = (sample_idx - 1) * n_categorical + 1
            
            for cat_idx in 1:n_categorical
                feature_idx = feature_indices[cat_idx]
                cat_value = Int32(categorical_data[sample_idx, feature_idx])
                
                # Calculate encoded column index
                encoding_offset = cat_idx == 1 ? Int32(0) : encoding_offsets[cat_idx - 1]
                col_idx[nnz_idx] = encoding_offset + cat_value + 1
                
                nnz_idx += 1
            end
        end
        
        return nothing
    end
    
    @cuda threads=threads blocks=blocks build_csr_kernel!(
        row_ptr, col_idx, categorical_data, 
        metadata.feature_indices, metadata.encoding_offsets,
        n_samples, n_categorical
    )
    
    # Set first row pointer
    CUDA.@allowscalar row_ptr[1] = 0
    
    return SparseOneHotMatrix(row_ptr, col_idx, values, n_samples, Int32(total_encoded), nnz)
end

"""
GPU kernel for high-cardinality feature hashing
"""
function hash_encode_kernel!(
    encoded_data::CuDeviceArray{Float32, 2},
    categorical_data::CuDeviceArray{Float32, 2},
    feature_indices::CuDeviceArray{Int32, 1},
    hash_dim::Int32,
    n_samples::Int32,
    n_categorical::Int32
)
    sample_idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    cat_feature_idx = blockIdx().y
    
    if sample_idx <= n_samples && cat_feature_idx <= n_categorical
        feature_idx = feature_indices[cat_feature_idx]
        cat_value = Int32(categorical_data[sample_idx, feature_idx])
        
        # Simple hash function (in practice, use more sophisticated hash)
        hash_idx = (cat_value * 2654435761) % hash_dim + 1
        
        # Set hashed feature
        encoded_data[sample_idx, (cat_feature_idx - 1) * hash_dim + hash_idx] = 1.0f0
    end
    
    return nothing
end

"""
Detect categorical features in the dataset
"""
function detect_categorical_features(
    feature_data::CuArray{Float32, 2},
    config::CategoricalConfig;
    categorical_threshold::Integer = 50
)
    n_features = size(feature_data, 2)
    n_samples = size(feature_data, 1)
    
    # Allocate detection arrays
    category_counts = CUDA.zeros(Int32, n_features)
    is_categorical = CUDA.zeros(Bool, n_features)
    
    # Run detection kernel
    threads = 256
    blocks = cld(n_features, threads)
    
    @cuda threads=threads blocks=blocks count_categories_kernel!(
        category_counts, is_categorical, feature_data,
        Int32(n_samples), Int32(n_features), Int32(categorical_threshold)
    )
    
    CUDA.synchronize()
    
    # Extract categorical feature indices
    is_cat_cpu = Array(is_categorical)
    cat_indices = Int32[]
    cat_counts = Int32[]
    
    for i in 1:n_features
        if is_cat_cpu[i]
            push!(cat_indices, i)
            push!(cat_counts, Array(category_counts)[i])
        end
    end
    
    return cat_indices, cat_counts
end

"""
Create categorical metadata from detected features
"""
function create_categorical_metadata(
    cat_indices::Vector{Int32},
    cat_counts::Vector{Int32},
    encoding_type::Symbol
)
    n_categorical = length(cat_indices)
    
    if n_categorical == 0
        return nothing
    end
    
    # Calculate encoding offsets based on type
    encoding_offsets = Int32[]
    total_encoded = Int32(0)
    
    if encoding_type == :onehot
        # Each categorical expands to n_categories features
        for count in cat_counts
            total_encoded += count
            push!(encoding_offsets, total_encoded)
        end
    elseif encoding_type == :ordinal
        # Each categorical remains one feature
        total_encoded = n_categorical
        encoding_offsets = collect(Int32, 1:n_categorical)
    else
        error("Unsupported encoding type: $encoding_type")
    end
    
    # Create category mapping (simplified - in practice would map actual values)
    category_offsets = Int32[0]
    total_categories = Int32(0)
    for count in cat_counts
        total_categories += count
        push!(category_offsets, total_categories)
    end
    
    category_mapping = collect(Int32, 0:(total_categories-1))
    
    return CategoricalMetadata(
        CuArray(cat_indices),
        CuArray(cat_counts),
        CuArray(category_offsets[1:end-1]),
        CuArray(category_mapping),
        CuArray(encoding_offsets),
        total_encoded
    )
end

"""
Encode categorical features using specified method
"""
function encode_categorical_features!(
    encoded_matrix::FeatureMatrix,
    feature_matrix::FeatureMatrix,
    metadata::CategoricalMetadata,
    config::CategoricalConfig
)
    n_categorical = Int32(length(metadata.feature_indices))
    
    if config.encoding_type == :onehot
        # Dense one-hot encoding
        threads = (256, 1)
        blocks = (cld(config.n_samples, threads[1]), Int(n_categorical))
        
        @cuda threads=threads blocks=blocks encode_onehot_kernel!(
            encoded_matrix.data,
            feature_matrix.data,
            metadata.feature_indices,
            metadata.category_counts,
            metadata.encoding_offsets,
            config.n_samples,
            n_categorical
        )
        
    elseif config.encoding_type == :ordinal
        # Create default ordinal mapping (0, 1, 2, ...)
        max_cats = maximum(Array(metadata.category_counts))
        ordinal_mapping = CUDA.zeros(Float32, max_cats, n_categorical)
        
        # Fill with simple ordinal values
        for i in 1:n_categorical
            n_cats = Array(metadata.category_counts)[i]
            ordinal_mapping[1:n_cats, i] = collect(Float32, 0:(n_cats-1)) ./ (n_cats - 1)
        end
        
        threads = (16, 16)
        blocks = (cld(config.n_samples, threads[1]), cld(n_categorical, threads[2]))
        
        @cuda threads=threads blocks=blocks encode_ordinal_kernel!(
            encoded_matrix.data,
            feature_matrix.data,
            metadata.feature_indices,
            metadata.category_counts,
            ordinal_mapping,
            config.n_samples,
            n_categorical
        )
    end
    
    CUDA.synchronize()
end

"""
Create mixed feature matrix with both numerical and encoded categorical features
"""
function create_mixed_feature_matrix(
    numerical_data::CuArray{Float32, 2},
    categorical_data::CuArray{Float32, 2},
    metadata::CategoricalMetadata,
    config::CategoricalConfig
)
    n_samples = size(numerical_data, 1)
    n_numerical = size(numerical_data, 2)
    n_encoded = metadata.total_encoded_features
    
    # Create combined matrix
    mixed_matrix = create_feature_matrix(n_samples, n_numerical + n_encoded)
    
    # Copy numerical features
    mixed_matrix.data[1:n_samples, 1:n_numerical] = numerical_data
    
    # Encode categorical features into remaining columns
    encoded_view = view(mixed_matrix.data, 1:n_samples, (n_numerical+1):(n_numerical+n_encoded))
    
    # Create temporary encoded matrix
    encoded_matrix = create_feature_matrix(n_samples, n_encoded)
    
    # Create temporary feature matrix for categorical data
    cat_feature_matrix = create_feature_matrix(n_samples, size(categorical_data, 2))
    copyto!(cat_feature_matrix.data[1:n_samples, 1:size(categorical_data, 2)], categorical_data)
    
    encode_categorical_features!(encoded_matrix, cat_feature_matrix, metadata, config)
    
    # Copy encoded features
    copyto!(encoded_view, encoded_matrix.data[1:n_samples, 1:n_encoded])
    
    return mixed_matrix
end

"""
Apply feature selection to mixed numerical/categorical data
"""
function apply_selection_mixed_features(
    mixed_matrix::FeatureMatrix,
    selected_indices::CuArray{Int32, 1},
    n_selected::Integer,
    n_numerical::Integer,
    metadata::CategoricalMetadata
)
    # Map selected indices back to original features
    # Handle both numerical and categorical features appropriately
    
    selected_indices_cpu = Array(selected_indices)[1:n_selected]
    
    # Separate numerical and categorical selections
    numerical_selected = Int32[]
    categorical_selected = Int32[]
    
    for idx in selected_indices_cpu
        if idx <= n_numerical
            push!(numerical_selected, idx)
        else
            # Map back to original categorical feature
            # This requires reverse mapping from encoded to original
            encoded_idx = idx - n_numerical
            
            # Find which categorical feature this belongs to
            for (cat_idx, offset) in enumerate(Array(metadata.encoding_offsets))
                if encoded_idx <= offset
                    push!(categorical_selected, Array(metadata.feature_indices)[cat_idx])
                    break
                end
            end
        end
    end
    
    return numerical_selected, categorical_selected
end

# Export types and functions
export CategoricalConfig, create_categorical_config
export CategoricalMetadata, SparseOneHotMatrix
export detect_categorical_features, create_categorical_metadata
export encode_categorical_features!, create_mixed_feature_matrix
export build_sparse_onehot, apply_selection_mixed_features

end # module CategoricalFeatures