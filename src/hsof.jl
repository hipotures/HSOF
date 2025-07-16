using JSON

include("config_loader.jl")
include("data_loader.jl")
include("stage1_filter.jl")
include("stage2_mcts.jl")
include("stage3_evaluation.jl")

function run_hsof(yaml_path::String)
    println("Starting HSOF pipeline...")
    println("Config: $yaml_path")
    
    # Load configuration
    config = load_config(yaml_path)
    
    # Load data
    X, y, feature_names = load_dataset(config)
    
    # Validate and adjust configuration based on actual data size
    config = validate_config(config, size(X, 2))
    
    # Stage 1: Fast filtering
    X1, features1, indices1 = stage1_filter(X, y, feature_names, config.stage1_features)
    
    # Stage 2: MCTS selection  
    X2, features2, indices2 = stage2_mcts(X1, y, features1, config.stage2_features)
    
    # Stage 3: Final evaluation
    final_features, final_score, best_model = stage3_evaluation(X2, y, features2, config.final_features, config.problem_type)
    
    # Save results
    results = Dict(
        "dataset" => config.dataset_name,
        "original_features" => length(feature_names),
        "final_features" => final_features,
        "final_score" => final_score,
        "best_model" => best_model,
        "reduction_ratio" => round(100 * (1 - length(final_features) / length(feature_names)), digits=1)
    )
    
    # Export results
    output_file = "$(config.dataset_name)_hsof_results.json"
    open(output_file, "w") do f
        JSON.print(f, results, 2)
    end
    
    println("\n=== HSOF COMPLETED ===")
    println("Original features: $(length(feature_names))")
    println("Final features: $(length(final_features))")
    println("Reduction: $(results["reduction_ratio"])%")
    println("Results saved: $output_file")
    
    return results
end

# Command line interface
function main()
    if length(ARGS) != 1
        println("Usage: julia hsof.jl <config.yaml>")
        println("Example: julia hsof.jl config/titanic.yaml")
        exit(1)
    end
    
    yaml_path = ARGS[1]
    run_hsof(yaml_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end