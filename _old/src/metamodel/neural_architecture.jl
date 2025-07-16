module NeuralArchitecture

using Flux
using CUDA
using Statistics
using Random
using LinearAlgebra

# Custom MultiHeadAttention layer for Flux.jl
struct MultiHeadAttention
    num_heads::Int
    head_dim::Int
    W_q::Dense
    W_k::Dense
    W_v::Dense
    W_o::Dense
    dropout::Dropout
end


"""
Create a multi-head attention layer with specified number of heads
"""
function MultiHeadAttention(dim::Int, num_heads::Int; dropout_rate=0.1f0)
    @assert dim % num_heads == 0 "Dimension must be divisible by number of heads"
    
    head_dim = dim ÷ num_heads
    
    # Query, Key, Value projections
    W_q = Dense(dim, dim, bias=true)
    W_k = Dense(dim, dim, bias=true)
    W_v = Dense(dim, dim, bias=true)
    
    # Output projection
    W_o = Dense(dim, dim, bias=true)
    
    # Dropout
    dropout = Dropout(dropout_rate)
    
    return MultiHeadAttention(num_heads, head_dim, W_q, W_k, W_v, W_o, dropout)
end

# Make MultiHeadAttention callable
function (mha::MultiHeadAttention)(x::AbstractArray)
    batch_size = size(x, 2)
    seq_len = 1  # For feature vectors, sequence length is 1
    dim = size(x, 1)
    
    # Linear projections in batch (batch_size, dim)
    Q = mha.W_q(x)  # (dim, batch_size)
    K = mha.W_k(x)  # (dim, batch_size)
    V = mha.W_v(x)  # (dim, batch_size)
    
    # Reshape for multi-head attention
    # Split into multiple heads: (head_dim, num_heads, batch_size)
    Q = reshape(Q, mha.head_dim, mha.num_heads, batch_size)
    K = reshape(K, mha.head_dim, mha.num_heads, batch_size)
    V = reshape(V, mha.head_dim, mha.num_heads, batch_size)
    
    # Compute attention scores for each head
    # scores = Q^T * K / sqrt(head_dim)
    # Use similar to allocate on same device as input
    scores = similar(x, batch_size, batch_size, mha.num_heads)
    
    # Use batched matrix multiplication for efficiency
    scale = eltype(x)(1.0 / sqrt(mha.head_dim))
    for h in 1:mha.num_heads
        # Extract head h for all batches
        Q_h = Q[:, h, :]  # (head_dim, batch_size)
        K_h = K[:, h, :]  # (head_dim, batch_size)
        
        # Compute attention scores: Q_h' * K_h
        scores_h = @view scores[:, :, h]
        mul!(scores_h, Q_h', K_h)
        scores_h .*= scale
    end
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores, dims=2)
    
    # Apply dropout
    attention_weights = mha.dropout(attention_weights)
    
    # Apply attention to values
    output = similar(x, mha.head_dim, mha.num_heads, batch_size)
    
    for h in 1:mha.num_heads
        V_h = V[:, h, :]  # (head_dim, batch_size)
        weights_h = @view attention_weights[:, :, h]  # (batch_size, batch_size)
        output_h = @view output[:, h, :]
        
        # Weighted sum of values: V_h * weights_h'
        mul!(output_h, V_h, weights_h')
    end
    
    # Concatenate heads and reshape back
    output = reshape(output, dim, batch_size)
    
    # Final linear projection
    output = mha.W_o(output)
    
    # Add residual connection
    return output + x
end

# Define Flux.params for MultiHeadAttention
Flux.@functor MultiHeadAttention (W_q, W_k, W_v, W_o, dropout)

"""
Metamodel architecture configuration
"""
struct MetamodelConfig
    input_dim::Int
    hidden_dims::Vector{Int}
    num_heads::Int
    dropout_rate::Float32
    use_fp16::Bool
end

"""
Create default metamodel configuration
"""
function create_metamodel_config(;
    input_dim::Int = 500,
    hidden_dims::Vector{Int} = [256, 128, 64],
    num_heads::Int = 8,
    dropout_rate::Float32 = 0.2f0,
    use_fp16::Bool = false
)
    return MetamodelConfig(
        input_dim,
        hidden_dims,
        num_heads,
        dropout_rate,
        use_fp16
    )
end

"""
Main metamodel architecture
"""
struct Metamodel
    input_layer::Dense
    attention::MultiHeadAttention
    hidden_layers::Vector{Dense}
    dropout_layers::Vector{Dropout}
    output_layer::Dense
    config::MetamodelConfig
end

"""
Create metamodel with specified configuration
Following architecture: Input(500) → Dense(256,relu) → Dropout(0.2) → 
MultiHeadAttention(8 heads) → Dense(128,relu) → Dropout(0.2) → 
Dense(64,relu) → Dense(1,sigmoid)
"""
function create_metamodel(config::MetamodelConfig = create_metamodel_config())
    # Initialize random seed for reproducibility
    Random.seed!(42)
    
    # Input layer with Glorot uniform initialization
    input_layer = Dense(config.input_dim, config.hidden_dims[1], relu; 
                       init=Flux.glorot_uniform)
    
    # Multi-head attention layer
    attention = MultiHeadAttention(config.hidden_dims[1], config.num_heads, 
                                 dropout_rate=config.dropout_rate)
    
    # Hidden layers
    hidden_layers = Dense[]
    dropout_layers = Dropout[]
    
    # First dropout (before attention)
    push!(dropout_layers, Dropout(config.dropout_rate))
    
    # First hidden layer (after attention) - 256 to 128
    push!(hidden_layers, Dense(config.hidden_dims[1], config.hidden_dims[2], relu; 
                              init=Flux.glorot_uniform))
    
    # Second dropout (after first hidden layer)
    push!(dropout_layers, Dropout(config.dropout_rate))
    
    # Second hidden layer - 128 to 64
    push!(hidden_layers, Dense(config.hidden_dims[2], config.hidden_dims[3], relu; 
                              init=Flux.glorot_uniform))
    
    # Output layer with sigmoid activation
    output_layer = Dense(config.hidden_dims[3], 1, sigmoid; 
                        init=Flux.glorot_uniform)
    
    return Metamodel(
        input_layer,
        attention,
        hidden_layers,
        dropout_layers,
        output_layer,
        config
    )
end

# Make Metamodel callable
function (model::Metamodel)(x::AbstractArray)
    # Input layer
    h = model.input_layer(x)
    
    # First dropout
    h = model.dropout_layers[1](h)
    
    # Multi-head attention
    h = model.attention(h)
    
    # Hidden layers with dropout
    h = model.hidden_layers[1](h)
    h = model.dropout_layers[2](h)
    
    h = model.hidden_layers[2](h)
    
    # Output layer
    return model.output_layer(h)
end

# Define Flux.params for Metamodel
Flux.@functor Metamodel (input_layer, attention, hidden_layers, dropout_layers, output_layer)

"""
Convert model to FP16 for inference
"""
function to_fp16(model::Metamodel)
    if !model.config.use_fp16
        @warn "Model not configured for FP16. Create with use_fp16=true"
        return model
    end
    
    # Convert all parameters to Float16
    return Flux.f16(model)
end

"""
Convert model to FP32 for training
"""
function to_fp32(model::Metamodel)
    return Flux.f32(model)
end

"""
Initialize model weights using Glorot uniform
"""
function initialize_weights!(model::Metamodel)
    # Re-initialize all dense layers using custom implementation
    glorot_uniform!(model.input_layer.weight)
    model.input_layer.bias .= 0
    
    # Attention layers
    glorot_uniform!(model.attention.W_q.weight)
    glorot_uniform!(model.attention.W_k.weight)
    glorot_uniform!(model.attention.W_v.weight)
    glorot_uniform!(model.attention.W_o.weight)
    
    model.attention.W_q.bias .= 0
    model.attention.W_k.bias .= 0
    model.attention.W_v.bias .= 0
    model.attention.W_o.bias .= 0
    
    # Hidden layers
    for layer in model.hidden_layers
        glorot_uniform!(layer.weight)
        layer.bias .= 0
    end
    
    # Output layer
    glorot_uniform!(model.output_layer.weight)
    model.output_layer.bias .= 0
    
    return model
end

"""
Custom Glorot uniform initialization
"""
function glorot_uniform!(w::AbstractArray)
    fan_in, fan_out = size(w, 2), size(w, 1)
    scale = sqrt(6.0f0 / (fan_in + fan_out))
    if w isa CuArray
        # Use randn on GPU and convert to uniform distribution
        CUDA.randn!(w)
        w .*= scale
    else
        # CPU version
        w .= (rand(eltype(w), size(w)...) .- 0.5f0) .* 2.0f0 .* scale
    end
    return w
end

"""
Create model on GPU with mixed precision support
"""
function create_gpu_metamodel(config::MetamodelConfig = create_metamodel_config())
    model = create_metamodel(config)
    
    # Move to GPU
    model = model |> gpu
    
    # Initialize weights after moving to GPU
    initialize_weights!(model)
    
    return model
end

"""
Compute model size in parameters
"""
function count_parameters(model::Metamodel)
    total_params = 0
    
    # Input layer
    total_params += length(model.input_layer.weight) + length(model.input_layer.bias)
    
    # Attention layers
    total_params += length(model.attention.W_q.weight) + length(model.attention.W_q.bias)
    total_params += length(model.attention.W_k.weight) + length(model.attention.W_k.bias)
    total_params += length(model.attention.W_v.weight) + length(model.attention.W_v.bias)
    total_params += length(model.attention.W_o.weight) + length(model.attention.W_o.bias)
    
    # Hidden layers
    for layer in model.hidden_layers
        total_params += length(layer.weight) + length(layer.bias)
    end
    
    # Output layer
    total_params += length(model.output_layer.weight) + length(model.output_layer.bias)
    
    return total_params
end

"""
Estimate model memory usage
"""
function estimate_memory_usage(model::Metamodel)
    n_params = count_parameters(model)
    
    # Memory per parameter
    bytes_per_param = model.config.use_fp16 ? 2 : 4
    
    # Model weights
    model_memory = n_params * bytes_per_param
    
    # Gradient storage (only for FP32 during training)
    gradient_memory = n_params * 4
    
    # Optimizer state (e.g., Adam needs 2x parameters for momentum)
    optimizer_memory = n_params * 4 * 2
    
    total_memory = model_memory + gradient_memory + optimizer_memory
    
    return (
        model_memory = model_memory,
        gradient_memory = gradient_memory,
        optimizer_memory = optimizer_memory,
        total_memory = total_memory,
        total_memory_mb = total_memory / (1024 * 1024)
    )
end

"""
Create a batch-friendly forward pass for efficiency
"""
function batch_forward(model::Metamodel, x::CuArray{Float32, 2})
    # Ensure we're in evaluation mode (disables dropout)
    Flux.testmode!(model)
    
    # Forward pass
    output = model(x)
    
    # Back to training mode
    Flux.trainmode!(model)
    
    return output
end

"""
Model summary and architecture visualization
"""
function model_summary(model::Metamodel)
    println("Metamodel Architecture Summary")
    println("=" ^ 50)
    
    config = model.config
    println("Input dimension: $(config.input_dim)")
    println("Hidden dimensions: $(config.hidden_dims)")
    println("Number of attention heads: $(config.num_heads)")
    println("Dropout rate: $(config.dropout_rate)")
    println("FP16 mode: $(config.use_fp16)")
    println()
    
    # Layer-by-layer summary
    println("Layer Structure:")
    println("1. Input Layer: $(config.input_dim) → $(config.hidden_dims[1]) (ReLU)")
    println("2. Dropout: p=$(config.dropout_rate)")
    println("3. MultiHeadAttention: $(config.num_heads) heads")
    println("4. Dense: $(config.hidden_dims[1]) → $(config.hidden_dims[2]) (ReLU)")
    println("5. Dropout: p=$(config.dropout_rate)")
    println("6. Dense: $(config.hidden_dims[2]) → $(config.hidden_dims[3]) (ReLU)")
    println("7. Output Layer: $(config.hidden_dims[3]) → 1 (Sigmoid)")
    println()
    
    # Parameter count
    n_params = count_parameters(model)
    println("Total parameters: $(n_params)")
    
    # Memory usage
    mem_info = estimate_memory_usage(model)
    println("Estimated memory usage:")
    println("  - Model weights: $(round(mem_info.model_memory / 1024^2, digits=2)) MB")
    println("  - Gradients: $(round(mem_info.gradient_memory / 1024^2, digits=2)) MB")
    println("  - Optimizer state: $(round(mem_info.optimizer_memory / 1024^2, digits=2)) MB")
    println("  - Total: $(round(mem_info.total_memory_mb, digits=2)) MB")
end

# Export types and functions
export MetamodelConfig, create_metamodel_config
export Metamodel, create_metamodel, create_gpu_metamodel
export MultiHeadAttention
export initialize_weights!, to_fp16, to_fp32
export count_parameters, estimate_memory_usage
export batch_forward, model_summary
export glorot_uniform!

end # module NeuralArchitecture