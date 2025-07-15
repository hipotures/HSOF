using Test
using Random

# Include the pre-training data module
include("../../src/metamodel/pretraining_data.jl")

using .PretrainingData

println("Testing Pre-training Data Generation Basic Functionality...")

# Test 1: Configuration
config = create_pretraining_config(
    n_samples = 5,
    n_features = 20,
    min_features = 3,
    max_features = 8,
    n_cv_folds = 2,
    n_parallel_workers = 1,
    use_xgboost = false,
    use_random_forest = false,  # Disable ML for speed
    augmentation_factor = 1.0,
    output_path = tempname() * ".h5"
)
println("✓ Configuration created")

# Test 2: Synthetic data generation
X, y = generate_synthetic_data(100, 20)
println("✓ Synthetic data generated: $(size(X))")

# Test 3: Feature combinations
combos = generate_diverse_combinations(config, 5)
println("✓ Generated $(length(combos)) feature combinations")

# Test 4: Create mock results (without ML training)
results = FeatureCombination[]
for combo in combos
    # Create mock scores
    mock_score = 0.5f0 + 0.3f0 * rand()
    push!(results, FeatureCombination(
        combo,
        mock_score,
        mock_score,
        mock_score,
        Int32(length(combo)),
        0.01f0
    ))
end
println("✓ Created $(length(results)) mock results")

# Test 5: Save to HDF5
PretrainingData.save_to_hdf5(results, config)
println("✓ Saved to HDF5: $(config.output_path)")

# Test 6: Load from HDF5
loaded = load_pretraining_data(config.output_path)
println("✓ Loaded $(length(loaded)) results from HDF5")

# Clean up
rm(config.output_path)

println("\n✅ All basic tests passed!")