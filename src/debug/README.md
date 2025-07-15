# Ensemble Debugging and Profiling Tools

This module provides comprehensive debugging and profiling capabilities for the HSOF ensemble MCTS system.

## Components

### 1. Tree State Visualizer (`tree_state_visualizer.jl`)
- Visualizes MCTS tree structure and node states
- Shows feature selections and exploration paths
- Generates tree evolution animations
- Color-coded node states (unexplored, exploring, selected, rejected, leaf)

### 2. Ensemble Timeline (`ensemble_timeline.jl`)
- Records and visualizes temporal execution events
- Tracks GPU synchronization points
- Creates Gantt charts for tree processing
- Identifies performance bottlenecks

### 3. Profiling Hooks (`profiling_hooks.jl`)
- Component-level performance profiling
- GPU kernel timing
- Memory allocation tracking
- Statistical analysis of execution times

### 4. Feature Heatmap (`feature_heatmap.jl`)
- Visualizes feature selection frequency across trees
- Shows consensus patterns
- Tracks tree diversity metrics
- Exports selection statistics

### 5. Debug Logger (`debug_logger.jl`)
- Multi-level logging (DEBUG, INFO, WARN, ERROR)
- Subsystem filtering
- Thread and GPU-aware logging
- Automatic log rotation and JSON export

### 6. Performance Analyzer (`performance_analyzer.jl`)
- Comprehensive performance analysis
- Bottleneck identification
- Optimization recommendations
- Implementation benchmarking

## Usage Example

```julia
include("src/debug/ensemble_debugger.jl")
using .EnsembleDebugger

# Create debug session
session = create_standard_debug_session(
    output_dir = "debug_output",
    log_level = :debug,
    profile_gpu = true
)

# Start debugging
start_debug_session(session)

# Your ensemble code here...
# The debugger will automatically track:
# - Tree state changes
# - GPU operations
# - Feature selections
# - Performance metrics

# Stop and generate reports
stop_debug_session(session)
```

## Output Files

The debugger generates several output files:

1. **Timeline Visualization**: `timeline_[session_id].json` and `.png`
2. **Profiling Report**: `profiling_[session_id].json`
3. **Feature Heatmap**: `feature_heatmap_[session_id].png`
4. **Debug Logs**: `ensemble_debug_[timestamp].log`
5. **Log Summary**: `log_summary_[timestamp].json`

## Performance Impact

The debugging tools are designed to have minimal impact on performance:
- Timeline recording: < 0.1ms per event
- Profiling overhead: < 5% for most components
- Memory tracking: < 10MB additional memory
- Can be completely disabled in production

## Integration with MCTS Ensemble

The debugger integrates seamlessly with the ensemble system through hooks:

```julia
# Hook into ensemble events
hook_ensemble_events!(debug_session, ensemble)

# The debugger will automatically track:
# - Tree selection events
# - GPU synchronization
# - Feature evaluations
# - Consensus building
```

## Visualization Examples

### Tree State Visualization
Shows the current state of MCTS trees with:
- Node colors indicating state
- Node size representing visit count
- Feature count annotations
- Tree statistics panel

### Timeline Visualization
Displays:
- Event sequence over time
- GPU activity periods
- Synchronization points
- Event frequency distribution

### Feature Heatmap
Presents:
- Feature selection frequency matrix
- Top selected features
- Tree diversity metrics
- Consensus analysis

## Best Practices

1. **Development**: Enable full debugging with all components
2. **Testing**: Use selective profiling for specific components
3. **Production**: Disable or use minimal logging only
4. **Analysis**: Run performance analyzer after major changes

## Troubleshooting

- **High memory usage**: Reduce `max_entries` in logger config
- **Slow visualization**: Limit tree depth in visualizer
- **Large log files**: Enable log rotation or filtering
- **GPU profiling issues**: Ensure CUDA.jl is properly configured