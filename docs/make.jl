using Documenter
using HSOF

# Generate documentation
makedocs(
    sitename = "HSOF Documentation",
    authors = "HSOF Development Team",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://hsof.readthedocs.io/en/latest/",
        assets = String[],
    ),
    modules = [HSOF],
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting-started/installation.md",
            "Quick Start" => "getting-started/quickstart.md",
            "Requirements" => "getting-started/requirements.md",
        ],
        "Architecture" => [
            "Overview" => "architecture/overview.md",
            "Three-Stage Pipeline" => "architecture/pipeline.md",
            "GPU Architecture" => "architecture/gpu-design.md",
            "Algorithm Design" => "architecture/algorithms.md",
        ],
        "API Reference" => [
            "GPU Modules" => "api/gpu-modules.md",
            "Feature Selection" => "api/feature-selection.md",
            "MCTS" => "api/mcts.md",
            "Metamodel" => "api/metamodel.md",
            "Utilities" => "api/utilities.md",
        ],
        "Tutorials" => [
            "Basic Usage" => "tutorials/basic-usage.md",
            "GPU Programming" => "tutorials/gpu-programming.md",
            "Custom Kernels" => "tutorials/custom-kernels.md",
            "Configuration" => "tutorials/configuration.md",
        ],
        "Benchmarks" => [
            "Performance Metrics" => "benchmarks/performance.md",
            "GPU Benchmarks" => "benchmarks/gpu-benchmarks.md",
            "Comparison" => "benchmarks/comparison.md",
        ],
    ],
    doctest = true,
    checkdocs = :exports,
    strict = true,
)

# Deploy documentation (optional)
# deploydocs(
#     repo = "github.com/your-org/HSOF.jl.git",
#     target = "build",
#     branch = "gh-pages",
#     devbranch = "main",
# )