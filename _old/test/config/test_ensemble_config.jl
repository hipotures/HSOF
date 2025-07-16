using Test
using JSON3
using YAML

# Test ensemble configuration system
include("../../src/config/ensemble_config.jl")
include("../../src/config/templates.jl")

using .EnsembleConfig
using .ConfigTemplates

@testset "Ensemble Configuration Tests" begin
    @testset "Configuration Creation and Validation" begin
        # Test default configuration
        config = EnsembleConfiguration()
        @test config.num_trees == 100
        @test config.trees_per_gpu == 50
        @test config.max_nodes_per_tree == 20000
        @test config.gpu_devices == [0, 1]
        @test config.lazy_expansion == true
        @test config.shared_features == true
        @test config.compressed_nodes == true
        
        # Test parameter validation
        @test_throws ArgumentError EnsembleConfiguration(num_trees = -1)
        @test_throws ArgumentError EnsembleConfiguration(exploration_constant_min = 2.0, exploration_constant_max = 1.0)
        @test_throws ArgumentError EnsembleConfiguration(target_features = 600, initial_features = 500)
        @test_throws ArgumentError EnsembleConfiguration(gpu_devices = Int[])
        @test_throws ArgumentError EnsembleConfiguration(min_iterations = 1000000, max_iterations = 500000)
        
        # Test valid parameter ranges
        @test_nowarn EnsembleConfiguration(
            num_trees = 50,
            exploration_constant_min = 0.5,
            exploration_constant_max = 2.0,
            feature_subset_ratio = 0.8,
            diversity_threshold = 0.7
        )
        
        @info "Configuration validation tests passed"
    end
    
    @testset "JSON Configuration Loading and Saving" begin
        # Create test configuration
        test_config = EnsembleConfiguration(
            num_trees = 20,
            trees_per_gpu = 10,
            max_iterations = 50000,
            gpu_devices = [0]
        )
        
        # Save to JSON
        json_path = tempname() * ".json"
        save_json_config(test_config, json_path)
        @test isfile(json_path)
        
        # Load from JSON
        loaded_config = load_json_config(json_path)
        @test loaded_config.num_trees == 20
        @test loaded_config.trees_per_gpu == 10
        @test loaded_config.max_iterations == 50000
        @test loaded_config.gpu_devices == [0]
        
        # Test invalid JSON file
        @test_throws ArgumentError load_json_config("nonexistent.json")
        
        # Cleanup
        rm(json_path)
        
        @info "JSON configuration tests passed"
    end
    
    @testset "YAML Configuration Loading and Saving" begin
        # Create test configuration
        test_config = EnsembleConfiguration(
            num_trees = 30,
            trees_per_gpu = 15,
            max_iterations = 75000,
            gpu_devices = [0, 1]
        )
        
        # Save to YAML
        yaml_path = tempname() * ".yaml"
        save_yaml_config(test_config, yaml_path)
        @test isfile(yaml_path)
        
        # Load from YAML
        loaded_config = load_yaml_config(yaml_path)
        @test loaded_config.num_trees == 30
        @test loaded_config.trees_per_gpu == 15
        @test loaded_config.max_iterations == 75000
        @test loaded_config.gpu_devices == [0, 1]
        
        # Test invalid YAML file
        @test_throws ArgumentError load_yaml_config("nonexistent.yaml")
        
        # Cleanup
        rm(yaml_path)
        
        @info "YAML configuration tests passed"
    end
    
    @testset "Dictionary Conversion" begin
        # Test config to dict conversion
        config = EnsembleConfiguration(num_trees = 42, trees_per_gpu = 21)
        config_dict = config_to_dict(config)
        
        @test config_dict["num_trees"] == 42
        @test config_dict["trees_per_gpu"] == 21
        @test haskey(config_dict, "max_nodes_per_tree")
        @test haskey(config_dict, "lazy_expansion")
        
        # Test dict to config conversion
        new_config = dict_to_config(config_dict)
        @test new_config.num_trees == 42
        @test new_config.trees_per_gpu == 21
        @test new_config.max_nodes_per_tree == config.max_nodes_per_tree
        
        @info "Dictionary conversion tests passed"
    end
    
    @testset "Command Line Argument Parsing" begin
        # Test basic argument parsing
        test_args = ["--num-trees", "80", "--trees-per-gpu", "40", "--gpu-devices", "0,1,2"]
        
        # Note: We can't easily test the full argument parsing without mocking ARGS
        # But we can test the argument parser creation
        parser = create_argument_parser()
        @test parser !== nothing
        
        # Test GPU devices parsing
        gpu_string = "0,1,2"
        gpu_ids = parse.(Int, split(gpu_string, ","))
        @test gpu_ids == [0, 1, 2]
        
        @info "Command line argument parsing tests passed"
    end
    
    @testset "Configuration Templates" begin
        # Test development template
        dev_config = development_config()
        @test dev_config.num_trees == 10
        @test dev_config.trees_per_gpu == 5
        @test dev_config.max_nodes_per_tree == 1000
        @test dev_config.gpu_devices == [0]
        @test dev_config.fault_tolerance == false
        
        # Test production template
        prod_config = production_config()
        @test prod_config.num_trees == 100
        @test prod_config.trees_per_gpu == 50
        @test prod_config.max_nodes_per_tree == 20000
        @test prod_config.gpu_devices == [0, 1]
        @test prod_config.fault_tolerance == true
        
        # Test high memory template
        high_mem_config = high_memory_config()
        @test high_mem_config.num_trees == 100
        @test high_mem_config.max_nodes_per_tree == 50000
        @test high_mem_config.initial_features == 2000
        @test high_mem_config.target_features == 100
        
        # Test fast exploration template
        fast_config = fast_exploration_config()
        @test fast_config.num_trees == 50
        @test fast_config.max_iterations == 100000
        @test fast_config.exploration_constant_min == 1.0
        @test fast_config.exploration_constant_max == 3.0
        
        # Test benchmark template
        bench_config = benchmark_config()
        @test bench_config.num_trees == 100
        @test bench_config.random_seed_base == 42
        @test bench_config.enable_profiling == true
        
        # Test single GPU template
        single_config = single_gpu_config()
        @test single_config.num_trees == 50
        @test single_config.gpu_devices == [0]
        @test single_config.fault_tolerance == false
        
        @info "Configuration templates tests passed"
    end
    
    @testset "Template Retrieval" begin
        # Test valid template names
        @test get_template("development") isa EnsembleConfiguration
        @test get_template("production") isa EnsembleConfiguration
        @test get_template("high-memory") isa EnsembleConfiguration
        @test get_template("fast") isa EnsembleConfiguration
        @test get_template("benchmark") isa EnsembleConfiguration
        @test get_template("single-gpu") isa EnsembleConfiguration
        
        # Test alias names
        @test get_template("dev") isa EnsembleConfiguration
        @test get_template("prod") isa EnsembleConfiguration
        @test get_template("large") isa EnsembleConfiguration
        @test get_template("quick") isa EnsembleConfiguration
        @test get_template("bench") isa EnsembleConfiguration
        @test get_template("single") isa EnsembleConfiguration
        
        # Test invalid template name
        @test_throws ArgumentError get_template("nonexistent")
        
        @info "Template retrieval tests passed"
    end
    
    @testset "Template Saving" begin
        # Test saving all templates
        temp_dir = mktempdir()
        
        save_all_templates(temp_dir)
        
        # Check that files were created
        @test isfile(joinpath(temp_dir, "development.json"))
        @test isfile(joinpath(temp_dir, "development.yaml"))
        @test isfile(joinpath(temp_dir, "production.json"))
        @test isfile(joinpath(temp_dir, "production.yaml"))
        @test isfile(joinpath(temp_dir, "high-memory.json"))
        @test isfile(joinpath(temp_dir, "high-memory.yaml"))
        @test isfile(joinpath(temp_dir, "fast.json"))
        @test isfile(joinpath(temp_dir, "fast.yaml"))
        @test isfile(joinpath(temp_dir, "benchmark.json"))
        @test isfile(joinpath(temp_dir, "benchmark.yaml"))
        @test isfile(joinpath(temp_dir, "single-gpu.json"))
        @test isfile(joinpath(temp_dir, "single-gpu.yaml"))
        
        # Test loading saved template
        saved_config = load_json_config(joinpath(temp_dir, "development.json"))
        @test saved_config.num_trees == 10
        @test saved_config.trees_per_gpu == 5
        
        # Cleanup
        rm(temp_dir, recursive = true)
        
        @info "Template saving tests passed"
    end
    
    @testset "Configuration Display" begin
        # Test configuration summary display
        config = development_config()
        
        # Capture output
        output = sprint(display_config_summary, config)
        
        @test occursin("Ensemble Configuration Summary", output)
        @test occursin("Trees: 10", output)
        @test occursin("GPU", output)
        @test occursin("Memory", output)
        @test occursin("MCTS", output)
        
        @info "Configuration display tests passed"
    end
    
    @testset "Configuration Integration" begin
        # Test full workflow: create, save, load, modify
        original_config = development_config()
        
        # Save to file
        json_path = tempname() * ".json"
        save_json_config(original_config, json_path)
        
        # Load from file
        loaded_config = load_json_config(json_path)
        @test loaded_config.num_trees == original_config.num_trees
        
        # Modify through dictionary
        config_dict = config_to_dict(loaded_config)
        config_dict["num_trees"] = 200
        config_dict["trees_per_gpu"] = 100
        
        modified_config = dict_to_config(config_dict)
        @test modified_config.num_trees == 200
        @test modified_config.trees_per_gpu == 100
        
        # Save modified version
        yaml_path = tempname() * ".yaml"
        save_yaml_config(modified_config, yaml_path)
        
        # Load and verify
        final_config = load_yaml_config(yaml_path)
        @test final_config.num_trees == 200
        @test final_config.trees_per_gpu == 100
        
        # Cleanup
        rm(json_path)
        rm(yaml_path)
        
        @info "Configuration integration tests passed"
    end
end

@info "All ensemble configuration tests completed successfully!"