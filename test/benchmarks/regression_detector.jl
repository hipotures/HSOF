module RegressionDetector

using Statistics
using JSON
using Dates

export PerformanceBaseline, RegressionResult, RegressionReport
export save_baseline, load_baseline, detect_regressions
export compare_with_baseline, generate_regression_report

"""
Performance metric
"""
struct PerformanceMetric
    name::String
    value::Float64
    unit::String
    higher_is_better::Bool
end

"""
Performance baseline for comparison
"""
struct PerformanceBaseline
    timestamp::DateTime
    git_commit::String
    julia_version::String
    metrics::Dict{String, PerformanceMetric}
    configuration::Dict{String, Any}
    hardware_info::Dict{String, String}
end

"""
Regression detection result
"""
struct RegressionResult
    metric_name::String
    baseline_value::Float64
    current_value::Float64
    change_percent::Float64
    regression_detected::Bool
    severity::Symbol  # :minor, :moderate, :severe
    confidence::Float64
end

"""
Complete regression report
"""
struct RegressionReport
    baseline::PerformanceBaseline
    current_metrics::Dict{String, PerformanceMetric}
    regressions::Vector{RegressionResult}
    improvements::Vector{RegressionResult}
    summary::Dict{String, Any}
    generated_at::DateTime
end

"""
Get current system information
"""
function get_system_info()::Dict{String, String}
    info = Dict{String, String}()
    
    # Julia version
    info["julia_version"] = string(VERSION)
    
    # OS information
    info["os"] = string(Sys.KERNEL)
    info["arch"] = string(Sys.ARCH)
    info["cpu_threads"] = string(Sys.CPU_THREADS)
    
    # Git commit (if in git repo)
    try
        commit = strip(read(`git rev-parse HEAD`, String))
        info["git_commit"] = commit[1:min(8, length(commit))]
    catch
        info["git_commit"] = "unknown"
    end
    
    # CUDA information
    if @isdefined(CUDA) && CUDA.functional()
        info["cuda_version"] = string(CUDA.version())
        info["gpu_count"] = string(length(CUDA.devices()))
        info["gpu_0_name"] = CUDA.name(CUDA.CuDevice(0))
    end
    
    return info
end

"""
Create a performance baseline
"""
function create_baseline(
    metrics::Dict{String, PerformanceMetric};
    configuration::Dict{String, Any} = Dict{String, Any}()
)::PerformanceBaseline
    
    system_info = get_system_info()
    
    return PerformanceBaseline(
        now(),
        get(system_info, "git_commit", "unknown"),
        get(system_info, "julia_version", string(VERSION)),
        metrics,
        configuration,
        system_info
    )
end

"""
Save baseline to file
"""
function save_baseline(baseline::PerformanceBaseline, filepath::String)
    data = Dict(
        "timestamp" => string(baseline.timestamp),
        "git_commit" => baseline.git_commit,
        "julia_version" => baseline.julia_version,
        "metrics" => Dict(
            name => Dict(
                "name" => m.name,
                "value" => m.value,
                "unit" => m.unit,
                "higher_is_better" => m.higher_is_better
            ) for (name, m) in baseline.metrics
        ),
        "configuration" => baseline.configuration,
        "hardware_info" => baseline.hardware_info
    )
    
    open(filepath, "w") do f
        JSON.print(f, data, 2)
    end
end

"""
Load baseline from file
"""
function load_baseline(filepath::String)::PerformanceBaseline
    data = JSON.parsefile(filepath)
    
    # Reconstruct metrics
    metrics = Dict{String, PerformanceMetric}()
    for (name, m) in data["metrics"]
        metrics[name] = PerformanceMetric(
            m["name"],
            m["value"],
            m["unit"],
            m["higher_is_better"]
        )
    end
    
    return PerformanceBaseline(
        DateTime(data["timestamp"]),
        data["git_commit"],
        data["julia_version"],
        metrics,
        data["configuration"],
        data["hardware_info"]
    )
end

"""
Detect performance regressions
"""
function detect_regressions(
    baseline::PerformanceBaseline,
    current_metrics::Dict{String, PerformanceMetric};
    thresholds::Dict{Symbol, Float64} = Dict(
        :minor => 0.05,      # 5% regression
        :moderate => 0.10,   # 10% regression
        :severe => 0.20      # 20% regression
    ),
    confidence_threshold::Float64 = 0.95
)::Vector{RegressionResult}
    
    results = RegressionResult[]
    
    for (name, current) in current_metrics
        if haskey(baseline.metrics, name)
            baseline_metric = baseline.metrics[name]
            
            # Calculate change
            change = if baseline_metric.higher_is_better
                (current.value - baseline_metric.value) / baseline_metric.value
            else
                (baseline_metric.value - current.value) / baseline_metric.value
            end
            
            change_percent = change * 100
            
            # Determine if regression
            regression_detected = change < -thresholds[:minor]
            
            # Determine severity
            severity = if abs(change) >= thresholds[:severe]
                :severe
            elseif abs(change) >= thresholds[:moderate]
                :moderate
            else
                :minor
            end
            
            # Simple confidence based on relative change
            confidence = min(1.0, abs(change) / thresholds[:minor])
            
            if regression_detected || change > thresholds[:minor]  # Also track improvements
                push!(results, RegressionResult(
                    name,
                    baseline_metric.value,
                    current.value,
                    change_percent,
                    regression_detected,
                    severity,
                    confidence
                ))
            end
        end
    end
    
    return results
end

"""
Statistical regression detection with multiple runs
"""
function detect_regressions_statistical(
    baseline_runs::Vector{Float64},
    current_runs::Vector{Float64};
    higher_is_better::Bool = false,
    significance_level::Float64 = 0.05
)::Tuple{Bool, Float64, Float64}
    
    # Basic statistics
    baseline_mean = mean(baseline_runs)
    current_mean = mean(current_runs)
    
    # Calculate change
    change = if higher_is_better
        (current_mean - baseline_mean) / baseline_mean
    else
        (baseline_mean - current_mean) / baseline_mean
    end
    
    # Welch's t-test for unequal variances
    n1, n2 = length(baseline_runs), length(current_runs)
    s1, s2 = std(baseline_runs), std(current_runs)
    
    t_stat = (baseline_mean - current_mean) / sqrt(s1^2/n1 + s2^2/n2)
    
    # Approximate degrees of freedom
    df = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))
    
    # Simple p-value approximation (would need proper distribution in practice)
    p_value = 2 * (1 - min(0.999, 0.5 + 0.5 * erf(abs(t_stat) / sqrt(2))))
    
    regression_detected = p_value < significance_level && change < 0
    
    return regression_detected, change, p_value
end

"""
Compare current performance with baseline
"""
function compare_with_baseline(
    baseline_path::String,
    current_metrics::Dict{String, PerformanceMetric};
    thresholds::Dict{Symbol, Float64} = Dict(:minor => 0.05, :moderate => 0.10, :severe => 0.20)
)::RegressionReport
    
    # Load baseline
    baseline = load_baseline(baseline_path)
    
    # Detect regressions
    all_results = detect_regressions(baseline, current_metrics, thresholds=thresholds)
    
    # Separate regressions and improvements
    regressions = filter(r -> r.regression_detected, all_results)
    improvements = filter(r -> !r.regression_detected && r.change_percent > 5.0, all_results)
    
    # Generate summary
    summary = Dict{String, Any}(
        "total_metrics" => length(current_metrics),
        "comparable_metrics" => length(all_results),
        "regressions_found" => length(regressions),
        "improvements_found" => length(improvements),
        "severe_regressions" => count(r -> r.severity == :severe, regressions),
        "moderate_regressions" => count(r -> r.severity == :moderate, regressions),
        "minor_regressions" => count(r -> r.severity == :minor, regressions)
    )
    
    return RegressionReport(
        baseline,
        current_metrics,
        regressions,
        improvements,
        summary,
        now()
    )
end

"""
Generate regression report
"""
function generate_regression_report(report::RegressionReport; verbose::Bool = true)
    if verbose
        println("\nPerformance Regression Report")
        println("=" ^ 80)
        println("Generated: $(report.generated_at)")
        println("Baseline: $(report.baseline.timestamp) (commit: $(report.baseline.git_commit))")
        println()
        
        # Summary
        println("Summary:")
        println("  Total metrics: $(report.summary["total_metrics"])")
        println("  Regressions: $(report.summary["regressions_found"])")
        println("    - Severe: $(report.summary["severe_regressions"])")
        println("    - Moderate: $(report.summary["moderate_regressions"])")
        println("    - Minor: $(report.summary["minor_regressions"])")
        println("  Improvements: $(report.summary["improvements_found"])")
        
        # Regressions
        if !isempty(report.regressions)
            println("\nRegressions Detected:")
            println("-" * 80)
            println("Metric                          | Baseline | Current  | Change   | Severity")
            println("-" * 80)
            
            for r in sort(report.regressions, by=r->r.change_percent)
                severity_icon = r.severity == :severe ? "🔴" : 
                               r.severity == :moderate ? "🟡" : "🟢"
                
                @printf("%-30s | %8.2f | %8.2f | %+7.1f%% | %s %s\n",
                    r.metric_name,
                    r.baseline_value,
                    r.current_value,
                    r.change_percent,
                    severity_icon,
                    r.severity
                )
            end
        end
        
        # Improvements
        if !isempty(report.improvements)
            println("\nImprovements:")
            println("-" * 80)
            println("Metric                          | Baseline | Current  | Change")
            println("-" * 80)
            
            for r in sort(report.improvements, by=r->r.change_percent, rev=true)
                @printf("%-30s | %8.2f | %8.2f | %+7.1f%%\n",
                    r.metric_name,
                    r.baseline_value,
                    r.current_value,
                    r.change_percent
                )
            end
        end
        
        println("-" * 80)
        
        # Action items
        if report.summary["severe_regressions"] > 0
            println("\n⚠️  SEVERE REGRESSIONS DETECTED - Investigation required!")
        elseif report.summary["moderate_regressions"] > 0
            println("\n⚠️  Moderate regressions detected - Review recommended")
        elseif report.summary["regressions_found"] > 0
            println("\nℹ️  Minor regressions detected - Monitor trend")
        else
            println("\n✅ No significant regressions detected")
        end
    end
    
    return report
end

"""
Track performance over time
"""
function track_performance_history(
    metrics_dir::String,
    current_metrics::Dict{String, PerformanceMetric};
    max_history::Int = 100
)
    # Create filename with timestamp
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(metrics_dir, "metrics_$timestamp.json")
    
    # Save current metrics
    baseline = create_baseline(current_metrics)
    save_baseline(baseline, filename)
    
    # Clean old files if exceeding max_history
    files = sort(filter(f -> endswith(f, ".json"), readdir(metrics_dir)))
    if length(files) > max_history
        for f in files[1:length(files)-max_history]
            rm(joinpath(metrics_dir, f))
        end
    end
end

"""
Analyze performance trends
"""
function analyze_trends(
    metrics_dir::String,
    metric_name::String;
    window_size::Int = 10
)
    # Load all metrics files
    files = sort(filter(f -> endswith(f, ".json"), readdir(metrics_dir)))
    
    if length(files) < 2
        println("Insufficient data for trend analysis")
        return
    end
    
    # Extract metric values
    timestamps = DateTime[]
    values = Float64[]
    
    for file in files[max(1, end-window_size+1):end]
        baseline = load_baseline(joinpath(metrics_dir, file))
        if haskey(baseline.metrics, metric_name)
            push!(timestamps, baseline.timestamp)
            push!(values, baseline.metrics[metric_name].value)
        end
    end
    
    if length(values) < 2
        println("Insufficient data for metric: $metric_name")
        return
    end
    
    # Calculate trend
    x = 1:length(values)
    y = values
    
    # Simple linear regression
    x_mean = mean(x)
    y_mean = mean(y)
    
    slope = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    intercept = y_mean - slope * x_mean
    
    # Trend direction
    trend = slope > 0 ? "increasing" : "decreasing"
    change_per_run = abs(slope)
    
    println("\nTrend Analysis: $metric_name")
    println("=" * 60)
    println("Window: $(length(values)) measurements")
    println("Period: $(timestamps[1]) to $(timestamps[end])")
    println("Trend: $trend")
    println("Change per run: $(round(change_per_run, digits=3))")
    println("Current value: $(round(values[end], digits=3))")
    println("Average: $(round(mean(values), digits=3))")
    println("Std dev: $(round(std(values), digits=3))")
end

end # module