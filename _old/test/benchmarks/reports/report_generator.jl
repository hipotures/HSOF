module ReportGenerator

using Dates
using Statistics
using Printf

export BenchmarkReport, generate_report, save_report
export generate_html_report, generate_markdown_report

"""
Complete benchmark report
"""
struct BenchmarkReport
    title::String
    timestamp::DateTime
    configuration::Dict{String, Any}
    gpu_profiles::Dict{String, Any}
    memory_profiles::Dict{String, Any}
    latency_profiles::Dict{String, Any}
    regression_results::Union{Nothing, Any}
    summary::Dict{String, Any}
end

"""
Generate comprehensive benchmark report
"""
function generate_report(
    title::String;
    gpu_profiles::Dict{String, Any} = Dict{String, Any}(),
    memory_profiles::Dict{String, Any} = Dict{String, Any}(),
    latency_profiles::Dict{String, Any} = Dict{String, Any}(),
    regression_results::Union{Nothing, Any} = nothing,
    configuration::Dict{String, Any} = Dict{String, Any}()
)::BenchmarkReport
    
    # Generate summary
    summary = generate_summary(gpu_profiles, memory_profiles, latency_profiles)
    
    return BenchmarkReport(
        title,
        now(),
        configuration,
        gpu_profiles,
        memory_profiles,
        latency_profiles,
        regression_results,
        summary
    )
end

"""
Generate summary statistics
"""
function generate_summary(gpu_profiles, memory_profiles, latency_profiles)
    summary = Dict{String, Any}()
    
    # GPU summary
    if !isempty(gpu_profiles)
        gpu_utils = [p["avg_utilization"] for p in values(gpu_profiles) if haskey(p, "avg_utilization")]
        summary["gpu_utilization_avg"] = isempty(gpu_utils) ? 0.0 : mean(gpu_utils)
        summary["gpu_utilization_max"] = isempty(gpu_utils) ? 0.0 : maximum(gpu_utils)
    end
    
    # Memory summary
    if !isempty(memory_profiles)
        host_peaks = [p["host_peak_mb"] for p in values(memory_profiles) if haskey(p, "host_peak_mb")]
        device_peaks = [p["device_peak_mb"] for p in values(memory_profiles) if haskey(p, "device_peak_mb")]
        summary["memory_host_peak_mb"] = isempty(host_peaks) ? 0.0 : maximum(host_peaks)
        summary["memory_device_peak_mb"] = isempty(device_peaks) ? 0.0 : maximum(device_peaks)
    end
    
    # Latency summary
    if !isempty(latency_profiles)
        p99_latencies = Float64[]
        for profile_set in values(latency_profiles)
            for profile in values(profile_set)
                if haskey(profile, "percentiles") && haskey(profile["percentiles"], 99)
                    push!(p99_latencies, profile["percentiles"][99])
                end
            end
        end
        summary["latency_p99_us"] = isempty(p99_latencies) ? 0.0 : mean(p99_latencies)
    end
    
    return summary
end

"""
Generate text report
"""
function generate_text_report(report::BenchmarkReport)::String
    io = IOBuffer()
    
    # Header
    println(io, "=" ^ 80)
    println(io, report.title)
    println(io, "=" * 80)
    println(io, "Generated: $(report.timestamp)")
    println(io)
    
    # Configuration
    if !isempty(report.configuration)
        println(io, "Configuration:")
        for (key, value) in report.configuration
            println(io, "  $key: $value")
        end
        println(io)
    end
    
    # Summary
    println(io, "Performance Summary:")
    println(io, "-" * 40)
    for (key, value) in report.summary
        if isa(value, Float64)
            println(io, "  $key: $(round(value, digits=2))")
        else
            println(io, "  $key: $value")
        end
    end
    println(io)
    
    # GPU Profiles
    if !isempty(report.gpu_profiles)
        println(io, "\nGPU Performance:")
        println(io, "-" * 40)
        for (name, profile) in report.gpu_profiles
            println(io, "\n$name:")
            if isa(profile, Dict)
                for (metric, value) in profile
                    if isa(value, Float64)
                        println(io, "  $metric: $(round(value, digits=2))")
                    else
                        println(io, "  $metric: $value")
                    end
                end
            end
        end
    end
    
    # Memory Profiles
    if !isempty(report.memory_profiles)
        println(io, "\nMemory Usage:")
        println(io, "-" * 40)
        for (name, profile) in report.memory_profiles
            println(io, "\n$name:")
            if isa(profile, Dict)
                println(io, "  Host Peak: $(round(get(profile, "host_peak_mb", 0.0), digits=2)) MB")
                println(io, "  Device Peak: $(round(get(profile, "device_peak_mb", 0.0), digits=2)) MB")
                println(io, "  Allocations: $(get(profile, "allocation_count", 0))")
            end
        end
    end
    
    # Latency Profiles
    if !isempty(report.latency_profiles)
        println(io, "\nLatency Analysis:")
        println(io, "-" * 40)
        for (config_name, profiles) in report.latency_profiles
            println(io, "\n$config_name:")
            for (batch_size, profile) in profiles
                if isa(profile, Dict) && haskey(profile, "percentiles")
                    println(io, "  Batch $batch_size:")
                    println(io, "    Mean: $(round(get(profile, "mean_us", 0.0), digits=1)) μs")
                    println(io, "    P99: $(round(get(profile["percentiles"], 99, 0.0), digits=1)) μs")
                    println(io, "    Throughput: $(round(get(profile, "throughput_ops_per_sec", 0.0), digits=0)) ops/s")
                end
            end
        end
    end
    
    # Regression Results
    if !isnothing(report.regression_results)
        println(io, "\nRegression Analysis:")
        println(io, "-" * 40)
        # Add regression details here
    end
    
    return String(take!(io))
end

"""
Generate markdown report
"""
function generate_markdown_report(report::BenchmarkReport)::String
    io = IOBuffer()
    
    # Header
    println(io, "# $(report.title)")
    println(io)
    println(io, "_Generated: $(report.timestamp)_")
    println(io)
    
    # Summary
    println(io, "## Performance Summary")
    println(io)
    println(io, "| Metric | Value |")
    println(io, "|--------|-------|")
    for (key, value) in report.summary
        if isa(value, Float64)
            println(io, "| $key | $(round(value, digits=2)) |")
        else
            println(io, "| $key | $value |")
        end
    end
    println(io)
    
    # GPU Performance
    if !isempty(report.gpu_profiles)
        println(io, "## GPU Performance")
        println(io)
        
        # Create table
        println(io, "| Operation | Execution Time (ms) | Avg Utilization (%) | Peak Utilization (%) | Memory (MB) |")
        println(io, "|-----------|-------------------|-------------------|---------------------|-------------|")
        
        for (name, profile) in report.gpu_profiles
            if isa(profile, Dict)
                exec_time = round(get(profile, "execution_time_ms", 0.0), digits=2)
                avg_util = round(get(profile, "avg_utilization", 0.0), digits=1)
                peak_util = round(get(profile, "peak_utilization", 0.0), digits=1)
                memory = round(get(profile, "memory_allocated_mb", 0.0), digits=1)
                
                println(io, "| $name | $exec_time | $avg_util | $peak_util | $memory |")
            end
        end
        println(io)
    end
    
    # Memory Usage
    if !isempty(report.memory_profiles)
        println(io, "## Memory Usage")
        println(io)
        
        println(io, "| Operation | Host Peak (MB) | Device Peak (MB) | Total Allocated (MB) | GC Events |")
        println(io, "|-----------|----------------|------------------|---------------------|-----------|")
        
        for (name, profile) in report.memory_profiles
            if isa(profile, Dict)
                host_peak = round(get(profile, "host_peak_mb", 0.0), digits=1)
                device_peak = round(get(profile, "device_peak_mb", 0.0), digits=1)
                total_alloc = round(get(profile, "host_allocated_mb", 0.0) + get(profile, "device_allocated_mb", 0.0), digits=1)
                gc_events = get(profile, "gc_events", 0)
                
                println(io, "| $name | $host_peak | $device_peak | $total_alloc | $gc_events |")
            end
        end
        println(io)
    end
    
    # Latency Analysis
    if !isempty(report.latency_profiles)
        println(io, "## Latency Analysis")
        println(io)
        
        for (config_name, profiles) in report.latency_profiles
            println(io, "### $config_name")
            println(io)
            
            println(io, "| Batch Size | Mean (μs) | P50 (μs) | P90 (μs) | P99 (μs) | Throughput (ops/s) |")
            println(io, "|------------|-----------|----------|----------|----------|-------------------|")
            
            for (batch_size, profile) in sort(collect(profiles))
                if isa(profile, Dict)
                    mean_us = round(get(profile, "mean_us", 0.0), digits=1)
                    p50 = round(get(get(profile, "percentiles", Dict()), 50, 0.0), digits=1)
                    p90 = round(get(get(profile, "percentiles", Dict()), 90, 0.0), digits=1)
                    p99 = round(get(get(profile, "percentiles", Dict()), 99, 0.0), digits=1)
                    throughput = round(get(profile, "throughput_ops_per_sec", 0.0), digits=0)
                    
                    println(io, "| $batch_size | $mean_us | $p50 | $p90 | $p99 | $throughput |")
                end
            end
            println(io)
        end
    end
    
    return String(take!(io))
end

"""
Generate HTML report with charts
"""
function generate_html_report(report::BenchmarkReport)::String
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>$(report.title)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #007bff; color: white; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 20px; margin: 10px 0; display: inline-block; min-width: 200px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .metric-label { color: #666; font-size: 14px; }
            .chart-container { margin: 20px 0; }
            canvas { max-width: 100%; height: 300px; }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>$(report.title)</h1>
            <p><em>Generated: $(report.timestamp)</em></p>
            
            <h2>Performance Summary</h2>
            <div class="metrics">
    """
    
    # Add summary metrics
    for (key, value) in report.summary
        if isa(value, Number)
            formatted_value = round(value, digits=2)
            html *= """
                <div class="metric-card">
                    <div class="metric-label">$(replace(key, "_" => " "))</div>
                    <div class="metric-value">$formatted_value</div>
                </div>
            """
        end
    end
    
    html *= """
            </div>
            
            <h2>GPU Performance</h2>
            <table>
                <tr>
                    <th>Operation</th>
                    <th>Execution Time (ms)</th>
                    <th>Avg Utilization (%)</th>
                    <th>Peak Utilization (%)</th>
                    <th>Memory (MB)</th>
                </tr>
    """
    
    # Add GPU performance data
    for (name, profile) in report.gpu_profiles
        if isa(profile, Dict)
            html *= """
                <tr>
                    <td>$name</td>
                    <td>$(round(get(profile, "execution_time_ms", 0.0), digits=2))</td>
                    <td>$(round(get(profile, "avg_utilization", 0.0), digits=1))</td>
                    <td>$(round(get(profile, "peak_utilization", 0.0), digits=1))</td>
                    <td>$(round(get(profile, "memory_allocated_mb", 0.0), digits=1))</td>
                </tr>
            """
        end
    end
    
    html *= """
            </table>
            
            <h2>Latency Analysis</h2>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
            
            <script>
                // Prepare latency data for chart
                const latencyData = {
                    labels: [],
                    datasets: []
                };
                
                // Add chart initialization here
                const ctx = document.getElementById('latencyChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: latencyData,
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Latency (μs)'
                                }
                            }
                        }
                    }
                });
            </script>
        </div>
    </body>
    </html>
    """
    
    return html
end

"""
Save report to file
"""
function save_report(report::BenchmarkReport, filepath::String; format::Symbol = :text)
    content = if format == :text
        generate_text_report(report)
    elseif format == :markdown
        generate_markdown_report(report)
    elseif format == :html
        generate_html_report(report)
    else
        error("Unsupported format: $format. Use :text, :markdown, or :html")
    end
    
    open(filepath, "w") do f
        write(f, content)
    end
end

"""
Generate performance visualization data
"""
function generate_visualization_data(report::BenchmarkReport)
    viz_data = Dict{String, Any}()
    
    # GPU utilization over time
    if !isempty(report.gpu_profiles)
        gpu_data = Dict(
            "labels" => collect(keys(report.gpu_profiles)),
            "utilization" => [get(p, "avg_utilization", 0.0) for p in values(report.gpu_profiles)],
            "memory" => [get(p, "memory_allocated_mb", 0.0) for p in values(report.gpu_profiles)]
        )
        viz_data["gpu"] = gpu_data
    end
    
    # Latency distribution
    if !isempty(report.latency_profiles)
        latency_data = Dict{String, Any}()
        for (config, profiles) in report.latency_profiles
            batch_sizes = sort(collect(keys(profiles)))
            latency_data[config] = Dict(
                "batch_sizes" => batch_sizes,
                "mean" => [get(profiles[bs], "mean_us", 0.0) for bs in batch_sizes],
                "p99" => [get(get(profiles[bs], "percentiles", Dict()), 99, 0.0) for bs in batch_sizes]
            )
        end
        viz_data["latency"] = latency_data
    end
    
    return viz_data
end

end # module