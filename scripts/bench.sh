#!/bin/bash

set -e

# Directories
BIN_DIR=bin
DATA_DIR=data
RESULTS_DIR=results

# Ensure necessary directories exist
mkdir -p $DATA_DIR $RESULTS_DIR

# Output file for benchmark results
RESULTS_FILE=$RESULTS_DIR/benchmark_results.json

# Configurations to test
POINTS=(10000 50000 100000 150000) 
DIMS=(2 3 4 5)
CLUSTERS=(3 5 10)
MAX_ITERS=100

# Initialize the results JSON
echo "[" > $RESULTS_FILE

for n_points in "${POINTS[@]}"; do
    for dims in "${DIMS[@]}"; do
        for k in "${CLUSTERS[@]}"; do
            # Generate input data
            INPUT_FILE="$DATA_DIR/input_${n_points}_${dims}.txt"
            OUTPUT_CPU_FILE="$DATA_DIR/results_cpu_${n_points}_${dims}_${k}.txt"
            OUTPUT_GPU_FILE="$DATA_DIR/results_gpu_${n_points}_${dims}_${k}.txt"
            echo "Generating data for N=$n_points, dims=$dims"
            ./bin/generate_data $INPUT_FILE $n_points $dims > /dev/null

            # Run CPU benchmark
            echo "Running CPU for N=$n_points, dims=$dims, k=$k"
            CPU_START=$(date +%s%N)
            ./bin/kmeans_cpu $INPUT_FILE $OUTPUT_CPU_FILE $k $MAX_ITERS > /dev/null
            CPU_END=$(date +%s%N)
            CPU_TIME=$((($CPU_END - $CPU_START) / 1000000))

            # Run GPU benchmark
            echo "Running GPU for N=$n_points, dims=$dims, k=$k"
            GPU_START=$(date +%s%N)
            ./bin/kmeans_gpu $INPUT_FILE $OUTPUT_GPU_FILE $k $MAX_ITERS > /dev/null
            GPU_END=$(date +%s%N)
            GPU_TIME=$((($GPU_END - $GPU_START) / 1000000))

            # Append results to JSON
            echo "  {" >> $RESULTS_FILE
            echo "    \"n_points\": $n_points," >> $RESULTS_FILE
            echo "    \"dims\": $dims," >> $RESULTS_FILE
            echo "    \"k\": $k," >> $RESULTS_FILE
            echo "    \"cpu_time_ms\": $CPU_TIME," >> $RESULTS_FILE
            echo "    \"gpu_time_ms\": $GPU_TIME" >> $RESULTS_FILE
            echo "  }," >> $RESULTS_FILE
        done
    done
done

# Finalize the JSON array
truncate -s-2 $RESULTS_FILE  # Remove trailing comma
echo "]" >> $RESULTS_FILE

echo "Benchmarking complete. Results saved to $RESULTS_FILE"
