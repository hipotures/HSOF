#!/bin/bash
# Run HSOF with parallel threads for metamodel training

# Set number of Julia threads
export JULIA_NUM_THREADS=4

echo "Running HSOF with $JULIA_NUM_THREADS parallel threads..."
julia test_hsof.jl "$@"