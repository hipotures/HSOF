#!/usr/bin/env julia

"""
Simple Parquet to CSV converter using Arrow.jl
"""

using Arrow, DataFrames, CSV

function read_parquet_file(file_path)
    println("📊 Attempting to read: $file_path")
    
    try
        # Try different Arrow.jl methods
        println("🔄 Method 1: Arrow.Table...")
        table = Arrow.Table(file_path)
        df = DataFrame(table)
        println("✅ Success with Arrow.Table: $(size(df))")
        return df
    catch e1
        println("❌ Arrow.Table failed: $e1")
        
        try
            println("🔄 Method 2: Arrow.Stream...")
            stream = Arrow.Stream(file_path)
            df = DataFrame(stream)
            println("✅ Success with Arrow.Stream: $(size(df))")
            return df
        catch e2
            println("❌ Arrow.Stream failed: $e2")
            
            try
                println("🔄 Method 3: Direct file read...")
                df = DataFrame(Arrow.readfile(file_path))
                println("✅ Success with readfile: $(size(df))")
                return df
            catch e3
                println("❌ All Arrow methods failed")
                println("Error 1: $e1")
                println("Error 2: $e2") 
                println("Error 3: $e3")
                return nothing
            end
        end
    end
end

if length(ARGS) != 1
    println("Usage: julia simple_parquet_reader.jl <file.parquet>")
    exit(1)
end

file_path = ARGS[1]
df = read_parquet_file(file_path)

if df !== nothing
    println("📋 Columns: $(names(df)[1:min(10, ncol(df))])")
    if ncol(df) > 10
        println("    ... and $(ncol(df) - 10) more")
    end
    
    output_file = replace(file_path, ".parquet" => ".csv")
    CSV.write(output_file, df)
    println("💾 Saved to: $output_file")
else
    println("💥 Failed to read parquet file")
end