#!/usr/bin/env julia

# Temporary fix: Load HSOF modules directly without full module system
println("Loading HSOF components directly...")

using CUDA
using DataFrames
using CSV
using Statistics
using LinearAlgebra

# Include only the essential modules we need
println("Loading Stage 1: Mutual Information...")
include("src/stage1_filter/gpu_memory_layout.jl")
include("src/stage1_filter/mutual_information.jl")

println("Loading Stage 2: MCTS GPU...")
include("src/gpu/kernels/mcts_types.jl")
include("src/gpu/mcts_gpu.jl")

println("Loading Stage 3: Cross Validation...")
include("src/stage3_evaluation/cross_validation.jl")

# Now run the real pipeline
println("\n" * "="^60)
println("PRAWDZIWY 3-STAGE HSOF PIPELINE")
println("="^60)

# Load Titanic data
csv_path = "competitions/Titanic/export/titanic_train_features.csv"
df = CSV.read(csv_path, DataFrame)
println("Loaded: $(nrow(df)) rows, $(ncol(df)) columns")

# Prepare data
target_col = "Survived"
exclude_cols = [target_col, "PassengerId", "id", "Id", "ID", "index"]
feature_cols = [col for col in names(df) if !(col in exclude_cols) && eltype(df[!, col]) <: Union{Number, Missing}]

X = Matrix{Float32}(df[:, feature_cols])
y = Float32.(df[!, target_col])

# Handle missing values
for i in 1:size(X, 2)
    col = X[:, i]
    if any(isnan.(col))
        X[isnan.(col), i] .= nanmean(col)
    end
end

println("Features: $(length(feature_cols))")
println("Samples: $(size(X, 1))")

# STAGE 1: Real Mutual Information on GPU
println("\nSTAGE 1: GPU Mutual Information")
X_gpu = CuArray(X)
y_gpu = CuArray(y)

using .GPUMemoryLayout
using .MutualInformation

# Create histogram buffers
hist_buffers = GPUMemoryLayout.create_histogram_buffers(size(X, 2), size(X, 1))

# Create MI config
mi_config = MutualInformation.create_mi_config(size(X, 2), size(X, 1))

# Compute MI scores
mi_scores = CUDA.zeros(Float32, size(X, 2))
MutualInformation.compute_mutual_information!(mi_scores, X_gpu, y_gpu, hist_buffers, mi_config)

mi_cpu = Array(mi_scores)
top_features = sortperm(mi_cpu, rev=true)[1:min(50, length(mi_cpu))]

println("Top 5 MI scores:")
for i in 1:5
    println("  $(feature_cols[top_features[i]]): $(mi_cpu[top_features[i]])")
end

# STAGE 2: Real MCTS GPU
println("\nSTAGE 2: GPU-MCTS") 
using .MCTSGPU

engine = MCTSGPU.MCTSGPUEngine()
MCTSGPU.initialize!(engine, X_gpu[:, top_features], y_gpu)

# Run MCTS
MCTSGPU.start!(engine)
sleep(5)  # Let it run for 5 seconds
MCTSGPU.stop!(engine)

best_features = MCTSGPU.get_best_features(engine, 20)
println("MCTS selected $(length(best_features)) features")

# STAGE 3: Real Model Evaluation
println("\nSTAGE 3: Model Evaluation")
using .ModelEvaluation

X_final = X[:, top_features[best_features]]
evaluator = ModelEvaluation.CrossValidationEvaluator(n_folds=5)
results = ModelEvaluation.evaluate_model(evaluator, X_final, y)

println("Cross-validation accuracy: $(results.mean_score) ± $(results.std_score)")

println("\n✅ PRAWDZIWY PIPELINE COMPLETE!")