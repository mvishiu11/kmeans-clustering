#!/bin/bash

set -e

# Directories
BIN_DIR=bin
DATA_DIR=data
OUTPUT_DIR=output

# Ensure directories exist
mkdir -p $DATA_DIR $OUTPUT_DIR

# Generate data
echo "Generating input data..."
GEN_START=$(date +%s%N | cut -b1-13)
$BIN_DIR/generate_data $DATA_DIR/input.txt 100000 3
GEN_END=$(date +%s%N | cut -b1-13)
echo "Data generation completed in $(($GEN_END - $GEN_START)) ms."

# Run GPU k-means
echo "Running GPU k-means..."
GPU_START=$(date +%s%N | cut -b1-13)
$BIN_DIR/kmeans_gpu $DATA_DIR/input.txt $OUTPUT_DIR/results_gpu.txt 3 100
GPU_END=$(date +%s%N | cut -b1-13)
echo "GPU k-means completed in $(($GPU_END - $GPU_START)) ms."

# Run CPU k-means
echo "Running CPU k-means..."
CPU_START=$(date +%s%N | cut -b1-13)
$BIN_DIR/kmeans_cpu $DATA_DIR/input.txt $OUTPUT_DIR/results_cpu.txt 3 100
CPU_END=$(date +%s%N | cut -b1-13)
echo "CPU k-means completed in $(($CPU_END - $CPU_START)) ms."

# Timing summary
echo "Execution times:"
echo "  Data generation: $(($GEN_END - $GEN_START)) ms"
echo "  GPU k-means: $(($GPU_END - $GPU_START)) ms"
echo "  CPU k-means: $(($CPU_END - $CPU_START)) ms"
