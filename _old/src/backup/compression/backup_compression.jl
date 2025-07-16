module BackupCompression

using CodecZlib
using CodecBzip2
using Tar
using Logging

export CompressionType, CompressedBackup, compress_directory, decompress_backup, calculate_compression_ratio

# Compression types
@enum CompressionType begin
    NONE = 0
    GZIP = 1
    BZIP2 = 2
    ZSTD = 3
end

# Compressed backup metadata
struct CompressedBackup
    original_path::String
    compressed_path::String
    compression_type::CompressionType
    original_size::Int64
    compressed_size::Int64
    compression_ratio::Float64
    checksum_original::String
    checksum_compressed::String
    
    function CompressedBackup(
        original_path::String,
        compressed_path::String,
        compression_type::CompressionType,
        original_size::Int64,
        compressed_size::Int64,
        checksum_original::String,
        checksum_compressed::String
    )
        ratio = original_size > 0 ? compressed_size / original_size : 1.0
        new(original_path, compressed_path, compression_type, original_size, 
            compressed_size, ratio, checksum_original, checksum_compressed)
    end
end

"""
Compress a directory into a compressed archive
"""
function compress_directory(
    source_dir::String, 
    output_path::String, 
    compression_type::CompressionType = GZIP,
    compression_level::Int = 6
)::Union{CompressedBackup, Nothing}
    
    @info "Compressing directory: $source_dir -> $output_path (type: $compression_type, level: $compression_level)"
    
    if !isdir(source_dir)
        @error "Source directory does not exist: $source_dir"
        return nothing
    end
    
    try
        # Calculate original size
        original_size = calculate_directory_size(source_dir)
        original_checksum = calculate_directory_checksum(source_dir)
        
        # Create temporary tar file
        temp_tar = "$output_path.tmp.tar"
        
        # Create tar archive
        @info "Creating tar archive..."
        create_tar_archive(source_dir, temp_tar)
        
        # Compress the tar file
        @info "Applying compression..."
        compressed_path = apply_compression(temp_tar, output_path, compression_type, compression_level)
        
        # Clean up temporary tar file
        rm(temp_tar, force=true)
        
        # Calculate compressed size and checksum
        compressed_size = filesize(compressed_path)
        compressed_checksum = calculate_file_checksum(compressed_path)
        
        result = CompressedBackup(
            source_dir, compressed_path, compression_type,
            original_size, compressed_size, original_checksum, compressed_checksum
        )
        
        @info "Compression completed: $(round(result.compression_ratio * 100, digits=1))% of original size"
        return result
        
    catch e
        @error "Failed to compress directory $source_dir" exception=e
        # Clean up any temporary files
        rm("$output_path.tmp.tar", force=true)
        rm(output_path, force=true)
        return nothing
    end
end

"""
Decompress a backup archive to a directory
"""
function decompress_backup(
    compressed_path::String,
    output_dir::String,
    compression_type::CompressionType = GZIP
)::Bool
    
    @info "Decompressing backup: $compressed_path -> $output_dir (type: $compression_type)"
    
    if !isfile(compressed_path)
        @error "Compressed file does not exist: $compressed_path"
        return false
    end
    
    try
        # Create output directory
        mkpath(output_dir)
        
        # Decompress to temporary tar file
        temp_tar = "$compressed_path.tmp.tar"
        
        @info "Decompressing archive..."
        if !apply_decompression(compressed_path, temp_tar, compression_type)
            return false
        end
        
        # Extract tar archive
        @info "Extracting tar archive..."
        extract_tar_archive(temp_tar, output_dir)
        
        # Clean up temporary tar file
        rm(temp_tar, force=true)
        
        @info "Decompression completed successfully"
        return true
        
    catch e
        @error "Failed to decompress backup $compressed_path" exception=e
        # Clean up any temporary files
        rm("$compressed_path.tmp.tar", force=true)
        return false
    end
end

"""
Create tar archive from directory
"""
function create_tar_archive(source_dir::String, tar_path::String)
    # Change to parent directory of source to create relative paths in tar
    parent_dir = dirname(source_dir)
    dir_name = basename(source_dir)
    
    # Create tar archive with relative paths
    open(tar_path, "w") do tar_file
        Tar.create(source_dir, tar_file)
    end
end

"""
Extract tar archive to directory
"""
function extract_tar_archive(tar_path::String, output_dir::String)
    open(tar_path, "r") do tar_file
        Tar.extract(tar_file, output_dir)
    end
end

"""
Apply compression to a file
"""
function apply_compression(
    input_path::String, 
    output_path::String, 
    compression_type::CompressionType, 
    level::Int
)::String
    
    final_output = if compression_type == GZIP
        "$output_path.tar.gz"
    elseif compression_type == BZIP2
        "$output_path.tar.bz2"
    elseif compression_type == ZSTD
        "$output_path.tar.zst"
    else
        # No compression, just rename
        cp(input_path, "$output_path.tar")
        return "$output_path.tar"
    end
    
    if compression_type == GZIP
        open(input_path, "r") do input_file
            open(final_output, "w") do output_file
                # Use GzipCompressor with specified level
                codec = GzipCompressor(level=level)
                stream = CodecZlib.TranscodingStream(codec, output_file)
                write(stream, input_file)
                close(stream)
            end
        end
    elseif compression_type == BZIP2
        open(input_path, "r") do input_file
            open(final_output, "w") do output_file
                # Use Bzip2Compressor
                codec = Bzip2Compressor(level=level)
                stream = CodecBzip2.TranscodingStream(codec, output_file)
                write(stream, input_file)
                close(stream)
            end
        end
    else
        # For ZSTD or others, use gzip as fallback
        open(input_path, "r") do input_file
            open(final_output, "w") do output_file
                codec = GzipCompressor(level=level)
                stream = CodecZlib.TranscodingStream(codec, output_file)
                write(stream, input_file)
                close(stream)
            end
        end
    end
    
    return final_output
end

"""
Apply decompression to a file
"""
function apply_decompression(
    input_path::String, 
    output_path::String, 
    compression_type::CompressionType
)::Bool
    
    try
        if compression_type == GZIP || endswith(input_path, ".gz")
            open(input_path, "r") do input_file
                open(output_path, "w") do output_file
                    codec = GzipDecompressor()
                    stream = CodecZlib.TranscodingStream(codec, input_file)
                    write(output_file, stream)
                    close(stream)
                end
            end
        elseif compression_type == BZIP2 || endswith(input_path, ".bz2")
            open(input_path, "r") do input_file
                open(output_path, "w") do output_file
                    codec = Bzip2Decompressor()
                    stream = CodecBzip2.TranscodingStream(codec, input_file)
                    write(output_file, stream)
                    close(stream)
                end
            end
        else
            # No compression, just copy
            cp(input_path, output_path)
        end
        
        return true
        
    catch e
        @error "Failed to decompress file $input_path" exception=e
        return false
    end
end

"""
Calculate total size of directory
"""
function calculate_directory_size(dir_path::String)::Int64
    total_size = 0
    
    for (root, dirs, files) in walkdir(dir_path)
        for file in files
            file_path = joinpath(root, file)
            try
                total_size += filesize(file_path)
            catch e
                @debug "Could not get size of file $file_path" exception=e
            end
        end
    end
    
    return total_size
end

"""
Calculate checksum for directory (hash of all file hashes)
"""
function calculate_directory_checksum(dir_path::String)::String
    file_hashes = String[]
    
    for (root, dirs, files) in walkdir(dir_path)
        for file in sort(files)  # Sort for consistent ordering
            file_path = joinpath(root, file)
            try
                file_hash = calculate_file_checksum(file_path)
                rel_path = relpath(file_path, dir_path)
                push!(file_hashes, "$rel_path:$file_hash")
            catch e
                @debug "Could not calculate hash for file $file_path" exception=e
            end
        end
    end
    
    # Create hash of all file hashes
    combined_hash = join(file_hashes, "|")
    return string(hash(combined_hash))
end

"""
Calculate checksum for a single file
"""
function calculate_file_checksum(file_path::String)::String
    if !isfile(file_path)
        return "missing"
    end
    
    try
        content = read(file_path)
        return string(hash(content))
    catch e
        @debug "Could not read file $file_path for checksum" exception=e
        return "error"
    end
end

"""
Calculate compression ratio
"""
function calculate_compression_ratio(original_size::Int64, compressed_size::Int64)::Float64
    return original_size > 0 ? compressed_size / original_size : 1.0
end

"""
Get recommended compression type based on data characteristics
"""
function recommend_compression_type(data_path::String)::CompressionType
    if !isdir(data_path) && !isfile(data_path)
        return GZIP  # Default fallback
    end
    
    # Sample some files to determine best compression
    sample_size = 0
    text_files = 0
    binary_files = 0
    
    paths_to_check = if isfile(data_path)
        [data_path]
    else
        # Sample first 10 files from directory
        files = String[]
        for (root, dirs, dir_files) in walkdir(data_path)
            for file in dir_files
                push!(files, joinpath(root, file))
                if length(files) >= 10
                    break
                end
            end
            if length(files) >= 10
                break
            end
        end
        files
    end
    
    for file_path in paths_to_check
        try
            if filesize(file_path) > 1024 * 1024  # Skip files > 1MB for sampling
                continue
            end
            
            sample_size += 1
            
            # Simple heuristic: check if file contains mostly text
            content = read(file_path, String)
            if is_mostly_text(content)
                text_files += 1
            else
                binary_files += 1
            end
        catch e
            # Assume binary if can't read as text
            binary_files += 1
        end
    end
    
    # Recommend compression based on file types
    if sample_size == 0
        return GZIP
    elseif text_files > binary_files
        return BZIP2  # Better for text data
    else
        return GZIP   # Faster for binary data
    end
end

"""
Check if content is mostly text
"""
function is_mostly_text(content::String)::Bool
    if isempty(content)
        return true
    end
    
    # Count printable characters
    printable_chars = count(c -> isprint(c) || isspace(c), content)
    ratio = printable_chars / length(content)
    
    return ratio > 0.7  # 70% printable characters
end

"""
Estimate compression ratio for data
"""
function estimate_compression_ratio(data_path::String, compression_type::CompressionType)::Float64
    # Simple estimation based on file types and compression method
    recommended_type = recommend_compression_type(data_path)
    
    base_ratio = if recommended_type == compression_type
        0.3  # Good match
    else
        0.5  # Suboptimal
    end
    
    # Adjust based on compression type
    if compression_type == BZIP2
        base_ratio *= 0.85  # Slightly better compression
    elseif compression_type == NONE
        base_ratio = 1.0    # No compression
    end
    
    return base_ratio
end

"""
Test compression performance on sample data
"""
function test_compression_performance(sample_path::String)::Dict{CompressionType, Float64}
    results = Dict{CompressionType, Float64}()
    
    if !isfile(sample_path) && !isdir(sample_path)
        return results
    end
    
    # Test each compression type
    for comp_type in [NONE, GZIP, BZIP2]
        try
            temp_output = tempname()
            
            start_time = time()
            if isdir(sample_path)
                compressed = compress_directory(sample_path, temp_output, comp_type, 6)
            else
                # For single file, create temp dir
                temp_dir = mktempdir()
                cp(sample_path, joinpath(temp_dir, basename(sample_path)))
                compressed = compress_directory(temp_dir, temp_output, comp_type, 6)
                rm(temp_dir, recursive=true, force=true)
            end
            duration = time() - start_time
            
            if compressed !== nothing
                results[comp_type] = duration
                rm(compressed.compressed_path, force=true)
            end
            
        catch e
            @debug "Failed to test compression type $comp_type" exception=e
        end
    end
    
    return results
end

end  # module BackupCompression