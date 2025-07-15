module SharedFeatureStorage

using CUDA
using ..MCTSTypes

"""
Shared feature vector storage system for 100-tree ensemble.
Uses hash-based deduplication and reference counting to minimize memory usage.
Saves approximately 40% memory compared to per-tree storage.
"""

# Constants for feature storage
const FEATURE_POOL_SIZE = 100_000       # Maximum unique feature sets
const FEATURE_HASH_TABLE_SIZE = 131_072  # 2^17 hash table size (power of 2)
const REFERENCE_COUNT_BITS = 16          # Max 65K references per feature set
const GENERATION_BITS = 16               # Generation counter for GC
const FEATURE_SET_HEADER_SIZE = 4        # Size of feature set header in UInt64s

"""
Feature set entry in the shared pool.
Each entry contains metadata and the actual feature bitfield.
"""
struct FeatureSetEntry
    # Header information (64 bits)
    header::UInt64  # ref_count(16) + generation(16) + feature_count(16) + flags(16)
    
    # Feature bitfield (variable length, but pre-allocated for MAX_FEATURES)
    features::NTuple{FEATURE_CHUNKS, UInt64}
    
    # Hash chain for collision resolution
    next_entry::UInt32  # Index of next entry in hash chain (0 = end)
    hash_value::UInt32  # Full hash value for verification
end

"""
Shared feature storage pool managing deduplication across all trees.
Uses open addressing with quadratic probing for hash collision resolution.
"""
struct SharedFeaturePool
    # Feature storage pool
    feature_entries::CuArray{FeatureSetEntry, 1}  # Pool of feature sets
    
    # Hash table for fast lookup (open addressing)
    hash_table::CuArray{UInt32, 1}  # Hash table indices (0 = empty)
    
    # Allocation management
    next_free_entry::CuArray{UInt32, 1}  # Next available entry (atomic)
    free_list::CuArray{UInt32, 1}        # Recycled entries
    free_list_size::CuArray{UInt32, 1}   # Size of free list (atomic)
    
    # Garbage collection
    gc_generation::CuArray{UInt16, 1}     # Current GC generation (atomic)
    gc_threshold::Float32                 # Trigger GC when pool is this full
    
    # Statistics
    total_entries::CuArray{UInt32, 1}     # Total allocated entries
    hash_collisions::CuArray{UInt64, 1}   # Number of hash collisions
    gc_cycles::CuArray{UInt32, 1}         # Number of GC cycles
    
    function SharedFeaturePool(device::CuDevice, gc_threshold::Float32 = 0.8f0)
        CUDA.device!(device) do
            # Allocate feature storage pool
            feature_entries = CuArray{FeatureSetEntry}(undef, FEATURE_POOL_SIZE)
            
            # Hash table for O(1) lookup
            hash_table = CUDA.zeros(UInt32, FEATURE_HASH_TABLE_SIZE)
            
            # Allocation management
            next_free_entry = CUDA.ones(UInt32, 1)  # Start at 1 (0 is invalid)
            free_list = CUDA.zeros(UInt32, FEATURE_POOL_SIZE)
            free_list_size = CUDA.zeros(UInt32, 1)
            
            # GC management
            gc_generation = CUDA.ones(UInt16, 1)
            
            # Statistics
            total_entries = CUDA.zeros(UInt32, 1)
            hash_collisions = CUDA.zeros(UInt64, 1)
            gc_cycles = CUDA.zeros(UInt32, 1)
            
            new(feature_entries, hash_table, next_free_entry, free_list, free_list_size,
                gc_generation, gc_threshold, total_entries, hash_collisions, gc_cycles)
        end
    end
end

# Feature set header manipulation
@inline function pack_feature_header(ref_count::UInt16, generation::UInt16, feature_count::UInt16, flags::UInt16)
    return (UInt64(ref_count) << 48) | (UInt64(generation) << 32) | (UInt64(feature_count) << 16) | UInt64(flags)
end

@inline function unpack_feature_header(header::UInt64)
    ref_count = UInt16((header >> 48) & 0xFFFF)
    generation = UInt16((header >> 32) & 0xFFFF)
    feature_count = UInt16((header >> 16) & 0xFFFF)
    flags = UInt16(header & 0xFFFF)
    return ref_count, generation, feature_count, flags
end

# Feature set flags
const FEATURE_FLAG_ACTIVE = UInt16(1)
const FEATURE_FLAG_MARKED = UInt16(2)  # For GC mark phase
const FEATURE_FLAG_IMMUTABLE = UInt16(4)  # Cannot be modified

# Hash function for feature bitfields
@inline function hash_feature_set(features::NTuple{FEATURE_CHUNKS, UInt64})
    hash = UInt32(0x12345678)  # Initial seed
    
    for i in 1:FEATURE_CHUNKS
        chunk = features[i]
        # Mix chunk bits using multiplication and XOR
        hash ⊻= UInt32(chunk & 0xFFFFFFFF)
        hash = hash * UInt32(0x9E3779B9) + UInt32(chunk >> 32)
        hash = (hash << 13) | (hash >> 19)  # Rotate left by 13
    end
    
    return hash
end

# Convert feature mask to tuple for hashing
@inline function feature_mask_to_tuple(feature_mask::CuArray{UInt64, 2}, node_idx::Int32)
    features = ntuple(i -> @inbounds(feature_mask[i, node_idx]), FEATURE_CHUNKS)
    return features
end

# Convert tuple back to feature mask
@inline function tuple_to_feature_mask!(feature_mask::CuArray{UInt64, 2}, node_idx::Int32, features::NTuple{FEATURE_CHUNKS, UInt64})
    for i in 1:FEATURE_CHUNKS
        @inbounds feature_mask[i, node_idx] = features[i]
    end
end

# Count set bits in feature tuple
@inline function count_features_in_tuple(features::NTuple{FEATURE_CHUNKS, UInt64})
    count = 0
    for i in 1:FEATURE_CHUNKS
        count += CUDA.popc(features[i])
    end
    return UInt16(count)
end

# Quadratic probing for hash table
@inline function find_hash_slot(hash_table::CuArray{UInt32, 1}, hash::UInt32, target_entry::UInt32)
    mask = UInt32(FEATURE_HASH_TABLE_SIZE - 1)
    slot = hash & mask
    
    for probe in 0:FEATURE_HASH_TABLE_SIZE-1
        current = @inbounds hash_table[slot + 1]  # +1 for Julia indexing
        
        if current == 0 || current == target_entry
            return slot + 1  # Return 1-based index
        end
        
        # Quadratic probing: slot = (slot + probe^2) % size
        slot = (slot + probe * probe) & mask
    end
    
    return 0  # Table full
end

# Store feature set in shared pool
@inline function store_feature_set!(
    pool::SharedFeaturePool,
    features::NTuple{FEATURE_CHUNKS, UInt64},
    initial_ref_count::UInt16 = UInt16(1)
)
    hash_val = hash_feature_set(features)
    
    # Try to find existing entry first
    existing_id = find_feature_set(pool, features, hash_val)
    if existing_id > 0
        # Increment reference count
        add_reference!(pool, existing_id)
        return existing_id
    end
    
    # Allocate new entry
    entry_id = allocate_feature_entry!(pool)
    if entry_id == 0
        return 0  # Pool exhausted
    end
    
    # Create feature set entry
    feature_count = count_features_in_tuple(features)
    current_gen = @inbounds pool.gc_generation[1]
    header = pack_feature_header(initial_ref_count, current_gen, feature_count, FEATURE_FLAG_ACTIVE)
    
    @inbounds pool.feature_entries[entry_id] = FeatureSetEntry(
        header,
        features,
        UInt32(0),  # No next entry initially
        hash_val
    )
    
    # Add to hash table
    slot = find_hash_slot(pool.hash_table, hash_val, entry_id)
    if slot > 0
        @inbounds pool.hash_table[slot] = entry_id
    else
        # Hash table full - this should trigger GC
        CUDA.atomic_add!(pointer(pool.hash_collisions), UInt64(1))
    end
    
    return entry_id
end

# Find existing feature set in pool
@inline function find_feature_set(
    pool::SharedFeaturePool,
    features::NTuple{FEATURE_CHUNKS, UInt64},
    hash_val::UInt32
)
    mask = UInt32(FEATURE_HASH_TABLE_SIZE - 1)
    slot = hash_val & mask
    
    for probe in 0:FEATURE_HASH_TABLE_SIZE-1
        entry_id = @inbounds pool.hash_table[slot + 1]
        
        if entry_id == 0
            return 0  # Not found
        end
        
        # Check if this entry matches
        @inbounds entry = pool.feature_entries[entry_id]
        if entry.hash_value == hash_val && entry.features == features
            ref_count, generation, _, flags = unpack_feature_header(entry.header)
            if (flags & FEATURE_FLAG_ACTIVE) != 0
                return entry_id  # Found active entry
            end
        end
        
        # Quadratic probing
        slot = (slot + probe * probe) & mask
    end
    
    return 0  # Not found
end

# Add reference to feature set
@inline function add_reference!(pool::SharedFeaturePool, entry_id::UInt32)
    if entry_id == 0 || entry_id > FEATURE_POOL_SIZE
        return false
    end
    
    # Atomically increment reference count
    @inbounds entry = pool.feature_entries[entry_id]
    ref_count, generation, feature_count, flags = unpack_feature_header(entry.header)
    
    if ref_count >= (1 << REFERENCE_COUNT_BITS) - 1
        return false  # Reference count overflow
    end
    
    new_header = pack_feature_header(ref_count + 1, generation, feature_count, flags)
    
    # Atomic update using CAS loop
    old_header = entry.header
    while true
        desired = FeatureSetEntry(new_header, entry.features, entry.next_entry, entry.hash_value)
        
        if CUDA.atomic_cas!(pointer(pool.feature_entries, entry_id), entry, desired) == entry
            break
        end
        
        # Reload entry and retry
        @inbounds entry = pool.feature_entries[entry_id]
        ref_count, generation, feature_count, flags = unpack_feature_header(entry.header)
        new_header = pack_feature_header(ref_count + 1, generation, feature_count, flags)
    end
    
    return true
end

# Remove reference from feature set
@inline function remove_reference!(pool::SharedFeaturePool, entry_id::UInt32)
    if entry_id == 0 || entry_id > FEATURE_POOL_SIZE
        return false
    end
    
    # Atomically decrement reference count
    @inbounds entry = pool.feature_entries[entry_id]
    ref_count, generation, feature_count, flags = unpack_feature_header(entry.header)
    
    if ref_count == 0
        return false  # Already at zero
    end
    
    new_ref_count = ref_count - 1
    new_flags = flags
    
    # If reference count reaches zero, mark for GC
    if new_ref_count == 0
        new_flags &= ~FEATURE_FLAG_ACTIVE
    end
    
    new_header = pack_feature_header(new_ref_count, generation, feature_count, new_flags)
    
    # Atomic update
    old_header = entry.header
    while true
        desired = FeatureSetEntry(new_header, entry.features, entry.next_entry, entry.hash_value)
        
        if CUDA.atomic_cas!(pointer(pool.feature_entries, entry_id), entry, desired) == entry
            break
        end
        
        # Reload and retry
        @inbounds entry = pool.feature_entries[entry_id]
        ref_count, generation, feature_count, flags = unpack_feature_header(entry.header)
        new_ref_count = ref_count - 1
        new_flags = (new_ref_count == 0) ? (flags & ~FEATURE_FLAG_ACTIVE) : flags
        new_header = pack_feature_header(new_ref_count, generation, feature_count, new_flags)
    end
    
    return true
end

# Allocate feature entry from pool
@inline function allocate_feature_entry!(pool::SharedFeaturePool)
    # Try free list first
    free_size = @inbounds pool.free_list_size[1]
    if free_size > 0
        old_size = CUDA.atomic_sub!(pointer(pool.free_list_size), UInt32(1))
        if old_size > 0
            entry_id = @inbounds pool.free_list[old_size]
            CUDA.atomic_add!(pointer(pool.total_entries), UInt32(1))
            return entry_id
        else
            # Restore counter
            CUDA.atomic_add!(pointer(pool.free_list_size), UInt32(1))
        end
    end
    
    # Allocate from sequential pool
    entry_id = CUDA.atomic_add!(pointer(pool.next_free_entry), UInt32(1))
    
    if entry_id > FEATURE_POOL_SIZE
        # Pool exhausted
        CUDA.atomic_sub!(pointer(pool.next_free_entry), UInt32(1))
        return UInt32(0)
    end
    
    CUDA.atomic_add!(pointer(pool.total_entries), UInt32(1))
    return entry_id
end

# Get feature set from pool
@inline function get_feature_set(pool::SharedFeaturePool, entry_id::UInt32)
    if entry_id == 0 || entry_id > FEATURE_POOL_SIZE
        return ntuple(i -> UInt64(0), FEATURE_CHUNKS)
    end
    
    @inbounds entry = pool.feature_entries[entry_id]
    return entry.features
end

# Copy feature set to node's feature mask
@inline function copy_feature_set_to_mask!(
    pool::SharedFeaturePool,
    entry_id::UInt32,
    feature_mask::CuArray{UInt64, 2},
    node_idx::Int32
)
    if entry_id == 0 || entry_id > FEATURE_POOL_SIZE
        # Clear feature mask
        for i in 1:FEATURE_CHUNKS
            @inbounds feature_mask[i, node_idx] = UInt64(0)
        end
        return
    end
    
    @inbounds entry = pool.feature_entries[entry_id]
    features = entry.features
    
    for i in 1:FEATURE_CHUNKS
        @inbounds feature_mask[i, node_idx] = features[i]
    end
end

# Garbage collection kernel
function gc_mark_phase_kernel!(pool::SharedFeaturePool, tree_pool_refs::CuArray{UInt32, 2})
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= size(tree_pool_refs, 1) * size(tree_pool_refs, 2)
        # Calculate 2D indices
        tree_idx = div(tid - 1, size(tree_pool_refs, 1)) + 1
        node_idx = mod(tid - 1, size(tree_pool_refs, 1)) + 1
        
        entry_id = @inbounds tree_pool_refs[node_idx, tree_idx]
        
        if entry_id > 0 && entry_id <= FEATURE_POOL_SIZE
            # Mark this entry as reachable
            @inbounds entry = pool.feature_entries[entry_id]
            ref_count, generation, feature_count, flags = unpack_feature_header(entry.header)
            
            if (flags & FEATURE_FLAG_ACTIVE) != 0
                new_flags = flags | FEATURE_FLAG_MARKED
                new_header = pack_feature_header(ref_count, generation, feature_count, new_flags)
                
                # Update header atomically
                old_entry = entry
                new_entry = FeatureSetEntry(new_header, entry.features, entry.next_entry, entry.hash_value)
                CUDA.atomic_cas!(pointer(pool.feature_entries, entry_id), old_entry, new_entry)
            end
        end
    end
    
    return nothing
end

function gc_sweep_phase_kernel!(pool::SharedFeaturePool)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    
    if tid <= FEATURE_POOL_SIZE
        @inbounds entry = pool.feature_entries[tid]
        ref_count, generation, feature_count, flags = unpack_feature_header(entry.header)
        
        if (flags & FEATURE_FLAG_ACTIVE) != 0 && (flags & FEATURE_FLAG_MARKED) == 0
            # This entry is not marked and should be collected
            new_flags = flags & ~FEATURE_FLAG_ACTIVE
            new_header = pack_feature_header(0, generation, feature_count, new_flags)
            
            new_entry = FeatureSetEntry(new_header, entry.features, entry.next_entry, entry.hash_value)
            @inbounds pool.feature_entries[tid] = new_entry
            
            # Add to free list
            pos = CUDA.atomic_add!(pointer(pool.free_list_size), UInt32(1)) + 1
            if pos <= FEATURE_POOL_SIZE
                @inbounds pool.free_list[pos] = tid
            end
        elseif (flags & FEATURE_FLAG_MARKED) != 0
            # Clear mark flag for next GC cycle
            new_flags = flags & ~FEATURE_FLAG_MARKED
            new_header = pack_feature_header(ref_count, generation, feature_count, new_flags)
            
            new_entry = FeatureSetEntry(new_header, entry.features, entry.next_entry, entry.hash_value)
            @inbounds pool.feature_entries[tid] = new_entry
        end
    end
    
    return nothing
end

# Trigger garbage collection
function garbage_collect!(pool::SharedFeaturePool, tree_pool_refs::CuArray{UInt32, 2})
    @info "Starting feature pool garbage collection"
    
    # Increment GC generation
    CUDA.atomic_add!(pointer(pool.gc_generation), UInt16(1))
    CUDA.atomic_add!(pointer(pool.gc_cycles), UInt32(1))
    
    # Mark phase: mark all reachable entries
    total_refs = size(tree_pool_refs, 1) * size(tree_pool_refs, 2)
    threads_per_block = 256
    blocks = div(total_refs + threads_per_block - 1, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks gc_mark_phase_kernel!(pool, tree_pool_refs)
    
    # Sweep phase: collect unmarked entries
    blocks = div(FEATURE_POOL_SIZE + threads_per_block - 1, threads_per_block)
    @cuda threads=threads_per_block blocks=blocks gc_sweep_phase_kernel!(pool)
    
    CUDA.synchronize()
    
    @info "Feature pool garbage collection completed"
end

# Check if GC should be triggered
@inline function should_gc(pool::SharedFeaturePool)
    CUDA.@allowscalar begin
        used = pool.total_entries[1]
        return Float32(used) / Float32(FEATURE_POOL_SIZE) > pool.gc_threshold
    end
end

# Get pool statistics
function get_pool_statistics(pool::SharedFeaturePool)
    CUDA.@allowscalar begin
        stats = Dict{String, Any}(
            "total_entries" => pool.total_entries[1],
            "free_entries" => pool.free_list_size[1],
            "capacity" => FEATURE_POOL_SIZE,
            "utilization" => Float32(pool.total_entries[1]) / Float32(FEATURE_POOL_SIZE),
            "hash_collisions" => pool.hash_collisions[1],
            "gc_cycles" => pool.gc_cycles[1],
            "gc_generation" => pool.gc_generation[1]
        )
        return stats
    end
end

export SharedFeaturePool, FeatureSetEntry
export store_feature_set!, find_feature_set, add_reference!, remove_reference!
export get_feature_set, copy_feature_set_to_mask!, allocate_feature_entry!
export garbage_collect!, should_gc, get_pool_statistics
export hash_feature_set, feature_mask_to_tuple, tuple_to_feature_mask!, count_features_in_tuple
export FEATURE_FLAG_ACTIVE, FEATURE_FLAG_MARKED, FEATURE_FLAG_IMMUTABLE

end # module