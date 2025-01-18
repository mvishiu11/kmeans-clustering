# Variables
CC = nvcc
CPP = g++
CFLAGS = -std=c++11 -O2
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include
DATA_DIR = data
BIN_DIR = bin

# Targets
GENERATE_DATA = $(BIN_DIR)/generate_data
KMEANS_GPU = $(BIN_DIR)/kmeans_gpu
KMEANS_CPU = $(BIN_DIR)/kmeans_cpu

# Default target
all: $(GENERATE_DATA) $(KMEANS_GPU) $(KMEANS_CPU)

# Build generate_data
$(GENERATE_DATA): $(SRC_DIR)/generate_input.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $<

# Build kmeans_gpu
$(KMEANS_GPU): $(SRC_DIR)/kmeans_gpu.cu
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $<

# Build kmeans_cpu
$(KMEANS_CPU): $(SRC_DIR)/kmeans_cpu.cpp
	mkdir -p $(BIN_DIR)
	$(CPP) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $<

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(DATA_DIR)

.PHONY: all clean