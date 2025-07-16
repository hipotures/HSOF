#!/usr/bin/env julia

"""
Main test script for GPU HSOF pipeline.

Usage:
    julia test_hsof.jl                    # Test all datasets
    julia test_hsof.jl titanic            # Test specific dataset
    julia test_hsof.jl --list             # List available datasets
    julia test_hsof.jl --help             # Show help
"""

using Pkg
Pkg.activate(".")

using CUDA
using JSON3, Dates

# Suppress XGBoost verbose output globally
ENV["XGBOOST_VERBOSITY"] = "0"

# Parse command line arguments
function parse_args()
    if length(ARGS) == 0
        return :all
    elseif ARGS[1] == "--help" || ARGS[1] == "-h"
        return :help
    elseif ARGS[1] == "--list" || ARGS[1] == "-l"
        return :list
    else
        return ARGS[1]
    end
end

# Show help
function show_help()
    println("""
    HSOF GPU Pipeline Test Script
    
    Usage:
        julia test_hsof.jl [options] [dataset]
    
    Options:
        --help, -h     Show this help message
        --list, -l     List available datasets
    
    Examples:
        julia test_hsof.jl                    # Test all datasets
        julia test_hsof.jl titanic            # Test titanic dataset
        julia test_hsof.jl playground_s4e11   # Test specific playground dataset
    """)
end

# List available datasets
function list_datasets()
    println("\nAvailable datasets:")
    config_dir = "/home/xai/.mdm/config/datasets"
    config_files = [
        "titanic",
        "playground_s4e11", 
        "playground_s4e2",
        "playground_s4e4",
        "playground_s5e7",
        "s4e12",
        "s5e6"
    ]
    
    for dataset in config_files
        yaml_path = joinpath(config_dir, "$dataset.yaml")
        if isfile(yaml_path)
            println("  ✓ $dataset")
        else
            println("  ✗ $dataset (config not found)")
        end
    end
end

# Main test function
function main()
    mode = parse_args()
    
    if mode == :help
        show_help()
        return
    elseif mode == :list
        list_datasets()
        return
    end
    
    println("="^80)
    println("GPU HSOF PIPELINE TEST")
    println("="^80)
    
    # Test GPU setup first
    global gpu_available = false
    println("\n🔧 TESTING GPU SETUP...")
    try
        if CUDA.functional()
            println("✅ CUDA is functional")
            println("   GPU: $(CUDA.name(CUDA.device()))")
            println("   VRAM: $(round(CUDA.total_memory()/1024^3, digits=1)) GB")
            println("   Available: $(round(CUDA.available_memory()/1024^3, digits=1)) GB")
            global gpu_available = true
        else
            println("❌ CUDA not functional")
            exit(1)
        end
    catch e
        println("❌ CUDA.jl not available: $e")
        exit(1)
    end
    
    # Load pipeline
    println("\n📦 LOADING HSOF PIPELINE...")
    try
        include("src/hsof.jl")
        println("✅ Pipeline loaded successfully")
    catch e
        println("❌ Failed to load pipeline: $e")
        exit(1)
    end
    
    # Load configuration
    config_path = "config/hsof.yaml"
    hsof_config = Base.invokelatest(load_hsof_config, config_path)
    println("\n📋 CONFIGURATION:")
    println("   Mode: $(hsof_config["mode"])")
    println("   Stage 1 - Correlation threshold: $(hsof_config["stage1"]["correlation_threshold"])")
    println("   Stage 2 - MCTS iterations: $(hsof_config["stage2"]["total_iterations"])")
    println("   Stage 3 - CV folds: $(hsof_config["stage3"]["cv_folds"])")
    
    # Determine which datasets to test
    config_dir = "/home/xai/.mdm/config/datasets"
    datasets_to_test = String[]
    
    if mode == :all
        println("\n📊 Testing all datasets...")
        push!(datasets_to_test, 
            "titanic",
            "playground_s4e11", 
            "playground_s4e2",
            "playground_s4e4",
            "playground_s5e7",
            "s4e12",
            "s5e6"
        )
    else
        # Single dataset
        dataset_name = string(mode)
        println("\n📊 Testing single dataset: $dataset_name")
        push!(datasets_to_test, dataset_name)
    end
    
    # Test results storage
    test_results = Dict()
    start_time = time()
    successful_tests = 0
    failed_tests = 0
    
    # Test each dataset
    for dataset in datasets_to_test
        yaml_path = joinpath(config_dir, "$dataset.yaml")
        
        if !isfile(yaml_path)
            println("\n⚠️  Skipping $dataset - config file not found at $yaml_path")
            failed_tests += 1
            continue
        end
        
        println("\n" * "="^80)
        println("📊 TESTING: $dataset")
        println("="^80)
        
        test_start = time()
        
        try
            # Run pipeline
            results = Base.invokelatest(run_hsof_gpu_pipeline, yaml_path, config_path=config_path)
            
            test_time = time() - test_start
            successful_tests += 1
            
            # Store results
            test_results[dataset] = Dict(
                "status" => "success",
                "time" => test_time,
                "stages" => Dict(
                    "original" => results["dataset_info"]["original_features"],
                    "stage1" => results["pipeline_stages"]["stage1"]["output_features"],
                    "stage2" => results["pipeline_stages"]["stage2"]["output_features"],
                    "final" => results["final_results"]["feature_count"],
                    "reduction" => results["final_results"]["reduction_percent"],
                    "model" => results["final_results"]["best_model"],
                    "score" => results["final_results"]["final_cv_score"]
                )
            )
            
            println("\n✅ Test completed in $(round(test_time, digits=2)) seconds")
            
            # Show quick summary for single dataset mode
            if mode != :all
                println("\n" * "-"^60)
                println("RESULTS SUMMARY:")
                println("-"^60)
                stages = test_results[dataset]["stages"]
                println("Feature reduction: $(stages["original"]) → $(stages["stage1"]) → $(stages["stage2"]) → $(stages["final"])")
                println("Total reduction: $(round(stages["reduction"], digits=1))%")
                println("Best model: $(stages["model"])")
                println("CV score: $(round(stages["score"], digits=4))")
                
                # Show selected features
                if haskey(results["final_results"], "selected_features")
                    println("\nSelected features:")
                    for (i, feat) in enumerate(results["final_results"]["selected_features"])
                        println("  $i. $feat")
                    end
                end
            end
            
        catch e
            failed_tests += 1
            test_results[dataset] = Dict(
                "status" => "failed",
                "error" => string(e),
                "time" => time() - test_start
            )
            
            println("\n❌ Test failed: $e")
            
            # Show stack trace in single dataset mode
            if mode != :all
                println("\nStack trace:")
                showerror(stdout, e, catch_backtrace())
            end
        end
        
        # Memory cleanup
        if gpu_available
            CUDA.reclaim()
        end
    end
    
    # Summary for all datasets
    if mode == :all && (successful_tests > 0 || failed_tests > 0)
        total_time = time() - start_time
        println("\n" * "="^80)
        println("📊 TEST SUMMARY")
        println("="^80)
        println("Total tests: $(successful_tests + failed_tests)")
        println("✅ Successful: $successful_tests")
        println("❌ Failed: $failed_tests")
        println("⏱️  Total time: $(round(total_time, digits=2)) seconds")
        
        # Results table
        if successful_tests > 0
            println("\n📈 RESULTS TABLE:")
            println("┌─────────────────────┬────────┬──────────────────────────┬──────────┬─────────┬──────────┐")
            println("│ Dataset             │ Time(s)│ Feature Reduction        │ Final %  │ Model   │ CV Score │")
            println("├─────────────────────┼────────┼──────────────────────────┼──────────┼─────────┼──────────┤")
            
            for (dataset, result) in sort(collect(test_results), by=x->x[1])
                if result["status"] == "success"
                    s = result["stages"]
                    dataset_display = length(dataset) > 18 ? dataset[1:15] * "..." : dataset
                    reduction_str = "$(s["original"])→$(s["stage1"])→$(s["stage2"])→$(s["final"])"
                    model_display = length(s["model"]) > 7 ? s["model"][1:7] : s["model"]
                    
                    println("│ $(rpad(dataset_display, 19)) │ " *
                            "$(lpad(round(result["time"], digits=1), 6)) │ " *
                            "$(rpad(reduction_str, 24)) │ " *
                            "$(lpad(round(s["reduction"], digits=1), 7))% │ " *
                            "$(rpad(model_display, 7)) │ " *
                            "$(lpad(round(s["score"], digits=3), 8)) │")
                end
            end
            println("└─────────────────────┴────────┴──────────────────────────┴──────────┴─────────┴──────────┘")
        end
        
        # Failed tests
        if failed_tests > 0
            println("\n⚠️  FAILED TESTS:")
            for (dataset, result) in test_results
                if result["status"] == "failed"
                    println("   - $dataset: $(result["error"])")
                end
            end
        end
    end
    
    # Save results for all datasets mode
    if mode == :all && !isempty(test_results)
        results_file = "hsof_test_results_$(Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")).json"
        open(results_file, "w") do f
            JSON3.write(f, test_results)
        end
        println("\n📄 Results saved to: $results_file")
    end
    
    println("\n✅ Testing completed!")
end

# Run main function
main()
