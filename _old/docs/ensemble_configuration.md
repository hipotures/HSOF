# Ensemble Configuration System

The Ensemble Configuration System provides comprehensive configuration management for the MCTS ensemble with parameter validation, file format support, and command-line overrides.

## Features

- **Parameter Validation**: Comprehensive validation of all configuration parameters with meaningful error messages
- **Multiple File Formats**: Support for JSON and YAML configuration files
- **Command-Line Overrides**: Full command-line argument parsing with parameter overrides
- **Configuration Templates**: Pre-built templates for common scenarios
- **Type Safety**: Strongly typed configuration with compile-time validation

## Configuration Structure

The `EnsembleConfiguration` struct contains all parameters needed for ensemble operation:

### Core Parameters
- `num_trees`: Number of trees in ensemble (default: 100)
- `trees_per_gpu`: Number of trees per GPU (default: 50)
- `max_nodes_per_tree`: Maximum nodes per tree (default: 20,000)
- `max_depth`: Maximum tree depth (default: 50)

### MCTS Parameters
- `exploration_constant_min/max`: Exploration constant range (default: 0.5-2.0)
- `virtual_loss`: Virtual loss value (default: 10)
- `max_iterations`: Maximum MCTS iterations (default: 1,000,000)

### Feature Selection
- `initial_features`: Starting feature count (default: 500)
- `target_features`: Target feature count (default: 50)
- `feature_subset_ratio`: Feature subset ratio per tree (default: 0.8)

### Memory Management
- `memory_pool_size`: Memory pool utilization (default: 0.8)
- `gc_threshold`: Garbage collection threshold (default: 0.75)
- `defrag_threshold`: Defragmentation threshold (default: 0.5)

### GPU Configuration
- `gpu_devices`: GPU device IDs (default: [0, 1])
- `memory_limit_gb`: Memory limit per GPU (default: 22.0 GB)

### Advanced Features
- `lazy_expansion`: Enable lazy node expansion (default: true)
- `shared_features`: Enable shared feature storage (default: true)
- `compressed_nodes`: Enable compressed node storage (default: true)
- `fault_tolerance`: Enable fault tolerance (default: true)

## Usage

### Basic Usage

```julia
using EnsembleConfig

# Create default configuration
config = EnsembleConfiguration()

# Create with specific parameters
config = EnsembleConfiguration(
    num_trees = 50,
    trees_per_gpu = 25,
    max_iterations = 500000
)
```

### Loading from Files

```julia
# Load from JSON
config = load_json_config("config/ensemble.json")

# Load from YAML
config = load_yaml_config("config/ensemble.yaml")
```

### Saving Configurations

```julia
# Save to JSON
save_json_config(config, "config/my_ensemble.json")

# Save to YAML
save_yaml_config(config, "config/my_ensemble.yaml")
```

### Command-Line Usage

```bash
# Load configuration with overrides
julia main.jl --config config/ensemble.json --num-trees 80 --gpu-devices 0,1,2

# Use specific template
julia main.jl --config config/development.json --enable-profiling

# Override boolean flags
julia main.jl --no-lazy-expansion --disable-dashboard
```

### Using Templates

```julia
using ConfigTemplates

# Get development configuration
dev_config = development_config()

# Get production configuration
prod_config = production_config()

# Get template by name
config = get_template("benchmark")

# List available templates
list_templates()

# Save all templates
save_all_templates("config/templates")
```

## Available Templates

### Development Template
- Reduced scale (10 trees, 5 per GPU)
- Single GPU, limited memory
- Quick convergence for testing
- Fault tolerance disabled for debugging

### Production Template
- Full scale (100 trees, 50 per GPU)
- Dual RTX 4090 GPUs
- Optimized for performance
- All features enabled

### High-Memory Template
- Large feature sets (2000 → 100 features)
- Increased memory usage
- Extended convergence settings
- Optimized for complex problems

### Fast Exploration Template
- Moderate scale (50 trees)
- Aggressive exploration parameters
- Quick convergence
- Speed-optimized settings

### Benchmark Template
- Standard configuration for performance testing
- Fixed random seed for reproducibility
- Comprehensive profiling enabled
- Consistent parameters across runs

### Single GPU Template
- Single GPU configuration
- Adjusted parameters for single device
- Fault tolerance disabled
- Optimized memory usage

## Parameter Validation

The configuration system provides comprehensive validation:

```julia
# These will throw ArgumentError
config = EnsembleConfiguration(num_trees = -1)  # Negative trees
config = EnsembleConfiguration(target_features = 600, initial_features = 500)  # Invalid reduction
config = EnsembleConfiguration(gpu_devices = Int[])  # Empty GPU list
```

## Integration with Ensemble

```julia
# Load configuration
config = parse_args_and_load_config()

# Create ensemble with configuration
ensemble = MemoryEfficientTreeEnsemble(
    device = CuDevice(config.gpu_devices[1]),
    max_trees = config.trees_per_gpu,
    max_nodes_per_tree = config.max_nodes_per_tree
)

# Initialize with configuration
initialize_ensemble!(ensemble, config)
```

## Configuration Files

### JSON Format
```json
{
  "num_trees": 100,
  "trees_per_gpu": 50,
  "max_nodes_per_tree": 20000,
  "gpu_devices": [0, 1],
  "lazy_expansion": true,
  "shared_features": true,
  "compressed_nodes": true
}
```

### YAML Format
```yaml
num_trees: 100
trees_per_gpu: 50
max_nodes_per_tree: 20000
gpu_devices: [0, 1]
lazy_expansion: true
shared_features: true
compressed_nodes: true
```

## Command-Line Reference

### Main Parameters
- `--config, -c`: Configuration file path
- `--num-trees`: Number of trees in ensemble
- `--trees-per-gpu`: Number of trees per GPU
- `--max-iterations`: Maximum MCTS iterations
- `--gpu-devices`: GPU device IDs (comma-separated)

### File Paths
- `--data-path`: Path to data file
- `--output-path`: Path to output results
- `--log-path`: Path to log file

### Boolean Flags
- `--enable-profiling` / `--disable-profiling`
- `--enable-dashboard` / `--disable-dashboard`
- `--lazy-expansion` / `--no-lazy-expansion`
- `--shared-features` / `--no-shared-features`
- `--compressed-nodes` / `--no-compressed-nodes`
- `--fault-tolerance` / `--no-fault-tolerance`

### Utility
- `--save-config`: Save final configuration to file
- `--version`: Show version information
- `--help`: Show help message

## Best Practices

1. **Use Templates**: Start with appropriate template for your use case
2. **Validate Early**: Configuration validation happens at construction time
3. **Override Carefully**: Use command-line overrides for experimentation
4. **Save Configurations**: Save working configurations for reproducibility
5. **Monitor Resources**: Adjust memory limits based on available GPU memory
6. **Test Thoroughly**: Use development template for initial testing

## Error Handling

The configuration system provides detailed error messages:

```julia
# Parameter validation error
julia> EnsembleConfiguration(num_trees = -1)
ERROR: ArgumentError: num_trees must be a positive integer, got: -1

# File not found error
julia> load_json_config("missing.json")
ERROR: ArgumentError: Configuration file not found: missing.json

# Invalid template error
julia> get_template("nonexistent")
ERROR: ArgumentError: Unknown template 'nonexistent'. Available templates: development, production, ...
```

## Testing

Run the comprehensive test suite:

```bash
julia test/config/test_ensemble_config.jl
```

The test suite covers:
- Parameter validation
- File loading and saving
- Template functionality
- Command-line parsing
- Integration workflows