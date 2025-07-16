module ConnectionPool

using SQLite
using DataFrames
using Dates
using Base.Threads: SpinLock, lock, unlock

export SQLitePool, acquire_connection, release_connection, close_pool, pool_stats

"""
Connection wrapper with metadata
"""
mutable struct PooledConnection
    db::SQLite.DB
    id::Int
    created_at::DateTime
    last_used::DateTime
    use_count::Int
    in_use::Bool
end

"""
Thread-safe SQLite connection pool
"""
mutable struct SQLitePool
    db_path::String
    connections::Vector{PooledConnection}
    min_size::Int
    max_size::Int
    timeout_seconds::Float64
    max_uses::Int
    max_lifetime_hours::Float64
    lock::SpinLock
    closed::Bool
end

"""
    SQLitePool(db_path; kwargs...)

Create a new SQLite connection pool for read-only access.

# Arguments
- `db_path`: Path to the SQLite database file
- `min_size=5`: Minimum number of connections to maintain
- `max_size=20`: Maximum number of connections allowed
- `timeout_seconds=30.0`: Connection acquisition timeout
- `max_uses=100`: Maximum uses before connection recycling
- `max_lifetime_hours=1.0`: Maximum connection lifetime
"""
function SQLitePool(
    db_path::String;
    min_size::Int = 5,
    max_size::Int = 20,
    timeout_seconds::Float64 = 30.0,
    max_uses::Int = 100,
    max_lifetime_hours::Float64 = 1.0
)
    # Validate database file exists
    if !isfile(db_path)
        throw(ArgumentError("Database file not found: $db_path"))
    end
    
    # Validate parameters
    if min_size < 1 || max_size < min_size
        throw(ArgumentError("Invalid pool size: min=$min_size, max=$max_size"))
    end
    
    pool = SQLitePool(
        db_path,
        PooledConnection[],
        min_size,
        max_size,
        timeout_seconds,
        max_uses,
        max_lifetime_hours,
        SpinLock(),
        false
    )
    
    # Initialize minimum connections
    lock(pool.lock) do
        for i in 1:min_size
            conn = create_connection(pool, i)
            push!(pool.connections, conn)
        end
    end
    
    return pool
end

"""
Create a new read-only connection
"""
function create_connection(pool::SQLitePool, id::Int)
    db = SQLite.DB(pool.db_path)
    
    # Set read-only mode
    SQLite.execute(db, "PRAGMA query_only = ON")
    
    # Optimize for read performance
    SQLite.execute(db, "PRAGMA temp_store = MEMORY")
    SQLite.execute(db, "PRAGMA mmap_size = 268435456")  # 256MB memory map
    SQLite.execute(db, "PRAGMA cache_size = -64000")    # 64MB cache
    
    return PooledConnection(
        db,
        id,
        now(),
        now(),
        0,
        false
    )
end

"""
Check if connection needs recycling
"""
function needs_recycling(conn::PooledConnection, pool::SQLitePool)
    if conn.use_count >= pool.max_uses
        return true
    end
    
    lifetime_hours = (now() - conn.created_at).value / (1000 * 60 * 60)
    if lifetime_hours >= pool.max_lifetime_hours
        return true
    end
    
    return false
end

"""
Validate connection health
"""
function is_healthy(conn::PooledConnection)
    try
        # Simple health check query
        result = DBInterface.execute(conn.db, "SELECT 1") |> DataFrame
        return size(result, 1) == 1
    catch
        return false
    end
end

"""
    acquire_connection(pool::SQLitePool) -> PooledConnection

Acquire a connection from the pool with automatic retry and health checking.
"""
function acquire_connection(pool::SQLitePool)
    if pool.closed
        throw(ErrorException("Connection pool is closed"))
    end
    
    start_time = time()
    
    while true
        lock(pool.lock) do
            # Find available connection
            for conn in pool.connections
                if !conn.in_use && !needs_recycling(conn, pool)
                    if is_healthy(conn)
                        conn.in_use = true
                        conn.last_used = now()
                        conn.use_count += 1
                        return conn
                    else
                        # Replace unhealthy connection
                        SQLite.close(conn.db)
                        new_conn = create_connection(pool, conn.id)
                        pool.connections[conn.id] = new_conn
                        new_conn.in_use = true
                        new_conn.use_count = 1
                        return new_conn
                    end
                end
            end
            
            # Recycle old connections
            for i in 1:length(pool.connections)
                conn = pool.connections[i]
                if !conn.in_use && needs_recycling(conn, pool)
                    SQLite.close(conn.db)
                    new_conn = create_connection(pool, i)
                    pool.connections[i] = new_conn
                    new_conn.in_use = true
                    new_conn.use_count = 1
                    return new_conn
                end
            end
            
            # Create new connection if under max_size
            if length(pool.connections) < pool.max_size
                id = length(pool.connections) + 1
                conn = create_connection(pool, id)
                conn.in_use = true
                conn.use_count = 1
                push!(pool.connections, conn)
                return conn
            end
        end
        
        # Check timeout
        if time() - start_time > pool.timeout_seconds
            throw(ErrorException("Connection acquisition timeout after $(pool.timeout_seconds) seconds"))
        end
        
        # Brief sleep before retry
        sleep(0.1)
    end
end

"""
    release_connection(pool::SQLitePool, conn::PooledConnection)

Release a connection back to the pool.
"""
function release_connection(pool::SQLitePool, conn::PooledConnection)
    lock(pool.lock) do
        conn.in_use = false
        conn.last_used = now()
    end
end

"""
    close_pool(pool::SQLitePool)

Close all connections and shut down the pool.
"""
function close_pool(pool::SQLitePool)
    lock(pool.lock) do
        pool.closed = true
        
        for conn in pool.connections
            try
                SQLite.close(conn.db)
            catch e
                @warn "Error closing connection $(conn.id): $e"
            end
        end
        
        empty!(pool.connections)
    end
end

"""
    pool_stats(pool::SQLitePool)

Get current pool statistics.
"""
function pool_stats(pool::SQLitePool)
    lock(pool.lock) do
        total = length(pool.connections)
        in_use = count(c -> c.in_use, pool.connections)
        available = total - in_use
        
        return Dict(
            "total_connections" => total,
            "in_use" => in_use,
            "available" => available,
            "min_size" => pool.min_size,
            "max_size" => pool.max_size,
            "closed" => pool.closed
        )
    end
end

"""
    with_connection(f, pool::SQLitePool)

Execute a function with an automatically managed connection.

# Example
```julia
result = with_connection(pool) do conn
    DBInterface.execute(conn.db, "SELECT * FROM features") |> DataFrame
end
```
"""
function with_connection(f::Function, pool::SQLitePool)
    conn = acquire_connection(pool)
    try
        return f(conn)
    finally
        release_connection(pool, conn)
    end
end

end # module