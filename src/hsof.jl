"""
GPU-only HSOF Pipeline: Hybrid Search for Optimal Features
3-stage GPU-accelerated feature selection with neural network metamodel.

Pipeline: N features → 500 → 50 → 10-20 features
Stages:
1. GPU correlation kernels (fast filtering)
2. MCTS with metamodel (intelligent search)  
3. Real model evaluation (precise selection)
"""

using CUDA, JSON3, YAML, Flux, Statistics, Random
using DataFrames, SQLite, Dates

# Include all pipeline components
include("config_loader.jl")
include("data_loader.jl")
include("metamodel.jl")
include("gpu_stage1.jl")
include("gpu_stage2.jl")
include("stage3_evaluation.jl")

"""
Main GPU-only HSOF pipeline function.
No CPU fallback - requires CUDA-capable GPU.
"""
function run_hsof_gpu_pipeline(yaml_path::String; config_path::String="config/hsof.yaml")
    println("="^80)
    println("GPU-ONLY HSOF PIPELINE")
    println("="^80)
    println("Hybrid Search for Optimal Features")
    println("3-Stage GPU-Accelerated Feature Selection")
    println("="^80)
    
    # Validate GPU requirements
    validate_gpu_setup()
    
    start_time = time()
    
    try
        # Load HSOF configuration
        hsof_config = load_hsof_config(config_path)
        print_config_summary(hsof_config)
        
        # Load dataset configuration and data
        config = load_config(yaml_path)
        X, y, feature_names = load_dataset(config)
        original_features = length(feature_names)
        
        println("\n" * "="^80)
        println("=== HSOF PIPELINE OVERVIEW ===")
        println("="^80)
        println("Dataset: $(config.name)")
        println("Problem type: $(config.problem_type)")
        println("Original features: $original_features")
        println("Pipeline stages: 3-stage intelligent feature reduction")
        println("GPU Device: $(CUDA.name(CUDA.device()))")
        println("Available VRAM: $(round(CUDA.available_memory()/1024^3, digits=1)) GB")
        
        # STAGE 1: GPU correlation filtering
        println("\n" * "="^80)
        println("STAGE 1: GPU CORRELATION FILTERING")
        println("="^80)
        
        stage1_start = time()
        X1, features1, indices1 = gpu_stage1_filter(
            X, y, feature_names,
            correlation_threshold=hsof_config["stage1"]["correlation_threshold"],
            min_features_to_keep=hsof_config["stage1"]["min_features_to_keep"],
            variance_threshold=hsof_config["stage1"]["variance_threshold"]
        )
        stage1_time = time() - stage1_start
        stage1_count = length(features1)
        
        println("Stage 1 completed in $(round(stage1_time, digits=2)) seconds")
        println("Features: $original_features → $stage1_count")
        
        # STAGE 2: MCTS with metamodel
        println("\n" * "="^80)
        println("STAGE 2: MCTS WITH METAMODEL")
        println("="^80)
        
        stage2_start = time()
        
        # Create and pre-train metamodel
        println("Creating metamodel...")
        metamodel = create_metamodel(
            stage1_count,
            hidden_sizes=hsof_config["stage2"]["hidden_sizes"],
            n_attention_heads=hsof_config["stage2"]["n_attention_heads"],
            dropout_rate=hsof_config["stage2"]["dropout_rate"]
        )
        
        println("Pre-training metamodel...")
        # Get XGBoost params for metamodel training from config
        xgb_params = get(hsof_config["stage2"], "metamodel_xgboost", Dict())
        parallel_threads = get(hsof_config["stage2"], "parallel_threads", 4)
        progress_interval = get(hsof_config["stage2"], "progress_interval", 500)
        pretrain_metamodel!(
            metamodel, X1, y, 
            n_samples=hsof_config["stage2"]["pretraining_samples"],
            epochs=hsof_config["stage2"]["pretraining_epochs"],
            batch_size=hsof_config["stage2"]["batch_size"],
            learning_rate=Float32(hsof_config["stage2"]["learning_rate"]),
            xgb_params=xgb_params,
            parallel_threads=parallel_threads,
            progress_interval=progress_interval
        )
        
        # Validate metamodel
        println("Validating metamodel...")
        correlation, mae = validate_metamodel_accuracy(metamodel, X1, y, n_test=50)
        
        if correlation < 0.6
            @warn "Metamodel correlation low ($correlation), results may be suboptimal"
        end
        
        # Run MCTS with metamodel
        X2, features2, indices2 = gpu_stage2_mcts_metamodel(
            X1, y, features1, metamodel, 
            total_iterations=hsof_config["stage2"]["total_iterations"],
            n_trees=hsof_config["stage2"]["n_trees"],
            exploration_constant=hsof_config["stage2"]["exploration_constant"],
            min_features=hsof_config["stage2"]["min_features"],
            max_features=hsof_config["stage2"]["max_features"]
        )
        
        stage2_time = time() - stage2_start
        stage2_count = length(features2)
        
        println("Stage 2 completed in $(round(stage2_time, digits=2)) seconds")
        println("Features: $stage1_count → $stage2_count")
        
        # STAGE 3: Real model evaluation
        println("\n" * "="^80)
        println("STAGE 3: REAL MODEL EVALUATION")
        println("="^80)
        
        stage3_start = time()
        final_features, final_score, best_model = stage3_precise_evaluation(
            X2, y, features2, config.problem_type,
            n_candidates=hsof_config["stage3"]["n_candidate_subsets"],
            target_range=(hsof_config["stage3"]["min_features_final"], hsof_config["stage3"]["max_features_final"]),
            cv_folds=hsof_config["stage3"]["cv_folds"],
            xgboost_params=hsof_config["stage3"]["xgboost"],
            rf_params=hsof_config["stage3"]["random_forest"]
        )
        stage3_time = time() - stage3_start
        final_count = length(final_features)
        
        println("Stage 3 completed in $(round(stage3_time, digits=2)) seconds")
        println("Features: $stage2_count → $final_count")
        
        # PIPELINE SUMMARY
        total_time = time() - start_time
        reduction_percent = round(100 * (1 - final_count / original_features), digits=1)
        
        println("\n" * "="^80)
        println("HSOF GPU PIPELINE COMPLETED")
        println("="^80)
        println("Total time: $(round(total_time, digits=2)) seconds")
        println("Pipeline: $original_features → $stage1_count → $stage2_count → $final_count")
        println("Reduction: $(reduction_percent)% features eliminated")
        println("Final CV score: $(round(final_score, digits=4)) ($(best_model))")
        
        # Stage timing breakdown
        println("\nStage timing breakdown:")
        println("  Stage 1 (GPU correlation): $(round(stage1_time, digits=2))s ($(round(100*stage1_time/total_time, digits=1))%)")
        println("  Stage 2 (MCTS+metamodel): $(round(stage2_time, digits=2))s ($(round(100*stage2_time/total_time, digits=1))%)")
        println("  Stage 3 (model evaluation): $(round(stage3_time, digits=2))s ($(round(100*stage3_time/total_time, digits=1))%)")
        
        
        # Export results
        results = generate_results_json(
            config, original_features, stage1_count, stage2_count, final_count,
            final_features, final_score, best_model, total_time,
            stage1_time, stage2_time, stage3_time, correlation, mae
        )
        
        output_file = "$(config.name)_hsof_gpu_results.json"
        JSON3.write(output_file, results)
        
        println("\n✅ Results exported to: $output_file")
        println("✅ GPU pipeline completed successfully!")
        
        return results
        
    catch e
        handle_pipeline_error(e)
        rethrow(e)
    finally
        # Cleanup GPU memory
        CUDA.reclaim()
    end
end


"""
Generate comprehensive results JSON.
"""
function generate_results_json(
    config, original_features, stage1_count, stage2_count, final_count,
    final_features, final_score, best_model, total_time,
    stage1_time, stage2_time, stage3_time, metamodel_correlation, metamodel_mae
)
    
    reduction_percent = round(100 * (1 - final_count / original_features), digits=1)
    
    return Dict(
        "pipeline_info" => Dict(
            "version" => "HSOF_GPU_v1.0",
            "timestamp" => string(now()),
            "gpu_device" => CUDA.name(CUDA.device()),
            "cuda_version" => string(CUDA.runtime_version()),
            "total_vram_gb" => round(CUDA.total_memory()/1024^3, digits=1)
        ),
        "dataset_info" => Dict(
            "name" => config.name,
            "problem_type" => config.problem_type,
            "original_features" => original_features,
            "database_path" => config.db_path,
            "target_column" => config.target
        ),
        "pipeline_stages" => Dict(
            "stage1" => Dict(
                "method" => "GPU_CUDA_correlation",
                "input_features" => original_features,
                "output_features" => stage1_count,
                "reduction_percent" => round(100 * (1 - stage1_count/original_features), digits=1),
                "time_seconds" => round(stage1_time, digits=2)
            ),
            "stage2" => Dict(
                "method" => "MCTS_with_metamodel",
                "input_features" => stage1_count,
                "output_features" => stage2_count,
                "reduction_percent" => round(100 * (1 - stage2_count/stage1_count), digits=1),
                "time_seconds" => round(stage2_time, digits=2),
                "metamodel_correlation" => round(metamodel_correlation, digits=4),
                "metamodel_mae" => round(metamodel_mae, digits=4)
            ),
            "stage3" => Dict(
                "method" => "real_model_evaluation",
                "input_features" => stage2_count,
                "output_features" => final_count,
                "reduction_percent" => round(100 * (1 - final_count/stage2_count), digits=1),
                "time_seconds" => round(stage3_time, digits=2),
                "best_model" => best_model,
                "best_cv_score" => round(final_score, digits=4)
            )
        ),
        "final_results" => Dict(
            "selected_features" => final_features,
            "feature_count" => final_count,
            "reduction_percent" => reduction_percent,
            "final_cv_score" => round(final_score, digits=4),
            "best_model" => best_model
        ),
        "performance" => Dict(
            "total_time_seconds" => round(total_time, digits=2),
            "total_time_minutes" => round(total_time/60, digits=2),
            "stage1_time_seconds" => round(stage1_time, digits=2),
            "stage2_time_seconds" => round(stage2_time, digits=2),
            "stage3_time_seconds" => round(stage3_time, digits=2),
            "gpu_utilization" => "optimized_for_rtx4090"
        ),
    )
end

"""
Handle pipeline errors with detailed diagnostics.
"""
function handle_pipeline_error(e::Exception)
    println("\n" * "="^60)
    println("=== PIPELINE ERROR ===")
    println("="^60)
    
    if isa(e, CUDA.OutOfGPUMemoryError)
        println("❌ GPU OUT OF MEMORY ERROR")
        println("Available VRAM: $(round(CUDA.available_memory()/1024^3, digits=1)) GB")
        println("Total VRAM: $(round(CUDA.total_memory()/1024^3, digits=1)) GB")
        println("Solutions:")
        println("  1. Use smaller dataset")
        println("  2. Reduce metamodel batch size")
        println("  3. Upgrade to larger GPU")
        
    elseif isa(e, SQLite.SQLiteException)
        println("❌ DATABASE ERROR")
        println("Check YAML configuration:")
        println("  - Database path exists")
        println("  - Table name is correct")
        println("  - Column names match")
        
    elseif isa(e, ArgumentError) && contains(string(e), "CUDA")
        println("❌ CUDA ERROR")
        println("CUDA not functional. GPU required for HSOF pipeline.")
        println("Solutions:")
        println("  1. Install CUDA drivers")
        println("  2. Check GPU compatibility")
        println("  3. Restart Julia session")
        
    else
        println("❌ UNEXPECTED ERROR: $(typeof(e))")
        println("Error message: $e")
        println("This is likely a bug - please report it.")
    end
    
    println("\nFor support, check:")
    println("  - GPU memory usage: nvidia-smi")
    println("  - Julia GPU status: using CUDA; CUDA.functional()")
    println("  - Dataset size vs GPU memory")
end

"""
Command-line interface for the GPU pipeline.
"""
function main()
    if length(ARGS) != 1
        println("Usage: julia --project=. src/hsof.jl <config.yaml>")
        println()
        println("Example:")
        println("  julia --project=. src/hsof.jl titanic.yaml")
        println()
        println("Requirements:")
        println("  - CUDA-capable GPU")
        println("  - SQLite database with features")
        println("  - YAML configuration file")
        exit(1)
    end
    
    yaml_path = ARGS[1]
    
    if !isfile(yaml_path)
        println("❌ Configuration file not found: $yaml_path")
        exit(1)
    end
    
    println("Starting GPU-only HSOF pipeline...")
    println("Configuration: $yaml_path")
    
    try
        results = run_hsof_gpu_pipeline(yaml_path)
        println("\n🎉 Pipeline completed successfully!")
        exit(0)
    catch e
        println("\n💥 Pipeline failed with error: $e")
        exit(1)
    end
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

"""
Interactive helper functions for development/testing.
"""

"""
Quick GPU test with synthetic data.
"""
function test_gpu_pipeline(n_samples::Int=1000, n_features::Int=100)
    println("=== GPU Pipeline Test ===")
    println("Generating synthetic data...")
    
    # Generate synthetic data
    X = randn(Float32, n_samples, n_features)
    y = rand(Float32, n_samples) .> 0.5
    y = Float32.(y)
    feature_names = ["feature_$i" for i in 1:n_features]
    
    # Create fake config
    config = (
        name = "test_dataset",
        problem_type = "binary_classification"
    )
    
    println("Running abbreviated pipeline...")
    
    try
        # Stage 1 test
        println("\nTesting Stage 1...")
        X1, features1, _ = gpu_stage1_filter(X, y, feature_names)
        
        # Stage 2 test (abbreviated)
        println("\nTesting Stage 2...")
        if size(X1, 2) > 64
            X1 = X1[:, 1:64]
            features1 = features1[1:64]
        end
        
        metamodel = create_metamodel(size(X1, 2))
        pretrain_metamodel!(metamodel, X1, y, n_samples=1000, epochs=10)
        
        X2, features2, _ = gpu_stage2_mcts_metamodel(
            X1, y, features1, metamodel,
            total_iterations=5000, n_trees=10
        )
        
        # Stage 3 test
        println("\nTesting Stage 3...")
        final_features, final_score, best_model = stage3_precise_evaluation(
            X2, y, features2, config.problem_type,
            n_candidates=20, target_range=(10, 15)
        )
        
        println("\n✅ GPU pipeline test completed successfully!")
        println("Final features: $(length(final_features))")
        println("Final score: $(round(final_score, digits=4))")
        
        return true
        
    catch e
        println("\n❌ GPU pipeline test failed: $e")
        return false
    end
end

"""
Benchmark GPU performance.
"""
function benchmark_gpu_performance()
    println("=== GPU Performance Benchmark ===")
    
    # Test different dataset sizes
    test_sizes = [
        (500, 50),
        (1000, 100),
        (2000, 200),
        (5000, 500)
    ]
    
    for (n_samples, n_features) in test_sizes
        println("\nTesting $n_samples samples × $n_features features...")
        
        # Generate test data
        X = randn(Float32, n_samples, n_features)
        y = rand(Float32, n_samples) .> 0.5
        y = Float32.(y)
        feature_names = ["feature_$i" for i in 1:n_features]
        
        # Benchmark Stage 1
        start_time = time()
        X1, features1, _ = gpu_stage1_filter(X, y, feature_names)
        stage1_time = time() - start_time
        
        println("  Stage 1: $(round(stage1_time, digits=3))s")
        println("  Features/second: $(round(n_features/stage1_time, digits=0))")
        
        # Memory usage
        gpu_memory_used = CUDA.total_memory() - CUDA.available_memory()
        println("  GPU memory used: $(round(gpu_memory_used/1024^3, digits=2)) GB")
    end
    
    println("\n✅ Performance benchmark completed")
end