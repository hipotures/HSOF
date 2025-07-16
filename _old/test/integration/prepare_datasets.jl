#!/usr/bin/env julia

# Script to download and prepare all reference datasets for integration testing

using Pkg

# Ensure required packages are available
required_packages = ["CSV", "DataFrames", "MLDatasets", "Downloads", "DelimitedFiles"]
for pkg in required_packages
    if !haskey(Pkg.project().dependencies, pkg)
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
end

# Load dataset loaders
include("data/dataset_loaders.jl")
using .DatasetLoaders

function main()
    println("Preparing Integration Test Datasets")
    println("=" ^ 50)
    
    # Set data directory
    data_dir = "test/integration/data"
    mkpath(data_dir)
    
    # Load all datasets
    datasets = load_all_reference_datasets(data_dir=data_dir)
    
    println("\n" * "=" ^ 50)
    println("Dataset Summary:")
    println("=" ^ 50)
    
    for (name, dataset) in datasets
        println("\n$name:")
        println("  Samples: $(dataset.n_samples)")
        println("  Features: $(dataset.n_features)")
        println("  Classes: $(dataset.n_classes)")
        println("  Size: $(round(sizeof(dataset.X) / 1024^2, digits=2)) MB")
    end
    
    # Create dataset info file
    info_file = joinpath(data_dir, "datasets_info.txt")
    open(info_file, "w") do io
        println(io, "Integration Test Datasets")
        println(io, "Generated: $(Dates.now())")
        println(io, "=" ^ 50)
        
        for (name, dataset) in datasets
            println(io, "\n$name Dataset:")
            println(io, "  Samples: $(dataset.n_samples)")
            println(io, "  Features: $(dataset.n_features)")
            println(io, "  Classes: $(dataset.n_classes)")
            println(io, "  Feature names (first 10): $(dataset.feature_names[1:min(10, end)])")
        end
    end
    
    println("\n✓ All datasets prepared successfully!")
    println("  Dataset info saved to: $info_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end