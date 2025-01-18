#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cfloat>

#define ERR(source) (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source), exit(EXIT_FAILURE))

/**
 * @brief Assigns each point to the nearest cluster.
 *
 * @param points Pointer to the input points array.
 * @param centroids Pointer to the centroids array.
 * @param assignments Pointer to the array storing cluster assignments for each point.
 * @param N Number of points.
 * @param dims Dimensionality of the points.
 * @param k Number of clusters.
 */
__global__ void assign_clusters(const float *points, const float *centroids, int *assignments, int N, int dims, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float min_distance = FLT_MAX;
    int best_cluster = 0;

    for (int c = 0; c < k; ++c) {
        float distance = 0.0;
        for (int d = 0; d < dims; ++d) {
            float diff = points[idx * dims + d] - centroids[c * dims + d];
            distance += diff * diff;
        }
        if (distance < min_distance) {
            min_distance = distance;
            best_cluster = c;
        }
    }

    assignments[idx] = best_cluster;
}

/**
 * @brief Computes new centroids using shared memory.
 *
 * @param points Pointer to the input points array.
 * @param assignments Pointer to the cluster assignments array.
 * @param centroids Pointer to the centroids array to be updated.
 * @param cluster_sizes Pointer to the array storing sizes of each cluster.
 * @param N Number of points.
 * @param dims Dimensionality of the points.
 * @param k Number of clusters.
 */
__global__ void compute_centroids_shared(const float *points, int *assignments, float *centroids, int *cluster_sizes, int N, int dims, int k) {
    extern __shared__ float shared_centroids[];
    int idx = threadIdx.x;

    // Initialize shared memory
    for (int d = idx; d < k * dims; d += blockDim.x) {
        shared_centroids[d] = 0.0;
    }
    __syncthreads();

    // Accumulate points in shared memory
    for (int i = blockIdx.x * blockDim.x + idx; i < N; i += blockDim.x * gridDim.x) {
        int cluster = assignments[i];
        for (int d = 0; d < dims; ++d) {
            atomicAdd(&shared_centroids[cluster * dims + d], points[i * dims + d]);
        }
        atomicAdd(&cluster_sizes[cluster], 1);
    }
    __syncthreads();

    // Write shared memory back to global memory
    for (int d = idx; d < k * dims; d += blockDim.x) {
        centroids[d] = shared_centroids[d];
    }
}

/**
 * @brief Main function for the GPU-based k-means clustering algorithm.
 *
 * @param argc Argument count.
 * @param argv Argument vector. Expected arguments:
 *   - argv[1]: Input file path.
 *   - argv[2]: Output file path.
 *   - argv[3]: Number of clusters (k).
 *   - argv[4]: Maximum iterations.
 * @return Exit code.
 */
int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <k> <max_iters>\n";
        exit(EXIT_FAILURE);
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int k = std::atoi(argv[3]);
    int max_iters = std::atoi(argv[4]);

    int N, dims;

    // Read input points
    std::ifstream file(input_file);
    if (!file.is_open()) ERR("ifstream.open");
    file >> N >> dims;

    std::vector<float> h_points(N * dims);
    for (int i = 0; i < N * dims; ++i) {
        file >> h_points[i];
    }
    file.close();

    // Allocate memory on GPU
    float *d_points, *d_centroids;
    int *d_assignments, *d_cluster_sizes;
    cudaMalloc(&d_points, sizeof(float) * N * dims);
    cudaMalloc(&d_centroids, sizeof(float) * k * dims);
    cudaMalloc(&d_assignments, sizeof(int) * N);
    cudaMalloc(&d_cluster_sizes, sizeof(int) * k);

    cudaMemcpy(d_points, h_points.data(), sizeof(float) * N * dims, cudaMemcpyHostToDevice);

    // Initialize centroids randomly
    std::vector<float> h_centroids(k * dims);
    for (int i = 0; i < k * dims; ++i) {
        h_centroids[i] = h_points[i];
    }
    cudaMemcpy(d_centroids, h_centroids.data(), sizeof(float) * k * dims, cudaMemcpyHostToDevice);

    // Main k-means loop
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    for (int iter = 0; iter < max_iters; ++iter) {
        assign_clusters<<<blocks, threads>>>(d_points, d_centroids, d_assignments, N, dims, k);
        cudaDeviceSynchronize();

        compute_centroids_shared<<<blocks, threads, k * dims * sizeof(float)>>>(d_points, d_assignments, d_centroids, d_cluster_sizes, N, dims, k);
        cudaDeviceSynchronize();
    }

    // Copy results back
    std::vector<int> h_assignments(N);
    cudaMemcpy(h_assignments.data(), d_assignments, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Write results to output file
    std::ofstream output(output_file);
    if (!output.is_open()) ERR("ofstream.open");

    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < dims; ++d) {
            output << h_points[i * dims + d] << " ";
        }
        output << h_assignments[i] << "\n";
    }

    output.close();

    // Free memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_cluster_sizes);

    return 0;
}
