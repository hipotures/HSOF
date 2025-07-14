# Run GPU Kernel Tests

push!(LOAD_PATH, joinpath(@__DIR__, "../.."))

# Load configuration
include("../../configs/config_loader.jl")
ConfigLoader.load_configs("dev")

# Include kernel tests
include("kernel_tests.jl")

# Run all tests
println("\n🚀 Starting GPU Kernel Tests...")
println("="^60)

try
    results = KernelTests.run_all_tests(verbose=true)
    
    # Check if all tests passed
    all_passed = all(r -> r.passed, results)
    
    if all_passed
        println("\n✅ All kernel tests passed!")
    else
        println("\n❌ Some kernel tests failed.")
        exit(1)
    end
    
catch e
    println("\n💥 Kernel tests failed with error:")
    println(e)
    exit(1)
end