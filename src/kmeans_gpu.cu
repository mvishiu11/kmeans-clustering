#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cfloat>

using namespace std;

/**
 * CUDA kernel to assign points to the nearest cluster.
 *
 * @param points Array of points.
 * @param centroids Array of centroids.
 * @param cluster_assignments Output array for cluster assignments.
 * @param N Number of points.
 * @param n Dimensionality of each point.
 * @param k Number of clusters.
 */
__global__ void assign_clusters(const float* points, float* centroids, int* cluster_assignments, int N, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float min_distance = FLT_MAX;
        int best_cluster = -1;

        for (int cluster = 0; cluster < k; ++cluster) {
            float distance = 0.0f;

            for (int dim = 0; dim < n; ++dim) {
                float diff = points[idx * n + dim] - centroids[cluster * n + dim];
                distance += diff * diff;
            }

            if (distance < min_distance) {
                min_distance = distance;
                best_cluster = cluster;
            }
        }

        cluster_assignments[idx] = best_cluster;
    }
}

/**
 * CUDA kernel to compute new centroids by aggregating points in each cluster.
 *
 * @param points Array of points.
 * @param centroids Output array for new centroids.
 * @param cluster_assignments Array of cluster assignments for each point.
 * @param cluster_sizes Output array for the number of points in each cluster.
 * @param N Number of points.
 * @param n Dimensionality of each point.
 * @param k Number of clusters.
 */
__global__ void compute_centroids(const float* points, float* centroids, const int* cluster_assignments, int* cluster_sizes, int N, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int cluster = cluster_assignments[idx];

        for (int dim = 0; dim < n; ++dim) {
            atomicAdd(&centroids[cluster * n + dim], points[idx * n + dim]);
        }

        atomicAdd(&cluster_sizes[cluster], 1);
    }
}

/**
 * Initialize centroids randomly from the dataset.
 *
 * @param centroids Output vector to store initial centroids.
 * @param points Input vector of points.
 * @param N Number of points.
 * @param n Dimensionality of each point.
 * @param k Number of clusters.
 */
void initialize_centroids(vector<float>& centroids, const vector<float>& points, int N, int n, int k) {
    srand(42);
    for (int i = 0; i < k; ++i) {
        int random_index = rand() % N;
        for (int dim = 0; dim < n; ++dim) {
            centroids[i * n + dim] = points[random_index * n + dim];
        }
    }
}

/**
 * Main function to run the k-means clustering algorithm.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char** argv) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file> <k> <max_iters>\n";
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];
    int k = stoi(argv[3]);
    int max_iters = stoi(argv[4]);

    // Read input file
    ifstream infile(input_file);
    if (!infile) {
        cerr << "Error: Unable to open input file." << endl;
        return 1;
    }

    int N, n;
    infile >> N >> n;

    vector<float> points(N * n);
    for (int i = 0; i < N * n; ++i) {
        infile >> points[i];
    }

    infile.close();

    // Allocate memory for centroids and initialize
    vector<float> centroids(k * n, 0.0f);
    initialize_centroids(centroids, points, N, n, k);

    // Device memory allocations
    float* d_points;
    float* d_centroids;
    int* d_cluster_assignments;
    int* d_cluster_sizes;

    cudaMalloc(&d_points, N * n * sizeof(float));
    cudaMalloc(&d_centroids, k * n * sizeof(float));
    cudaMalloc(&d_cluster_assignments, N * sizeof(int));
    cudaMalloc(&d_cluster_sizes, k * sizeof(int));

    cudaMemcpy(d_points, points.data(), N * n * sizeof(float), cudaMemcpyHostToDevice);

    // Iterative k-means computation
    for (int iter = 0; iter < max_iters; ++iter) {
        cudaMemcpy(d_centroids, centroids.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 1024;
        int blocks = (N + threads - 1) / threads;
        assign_clusters<<<blocks, threads>>>(d_points, d_centroids, d_cluster_assignments, N, n, k);

        cudaMemset(d_centroids, 0, k * n * sizeof(float));
        cudaMemset(d_cluster_sizes, 0, k * sizeof(int));
        compute_centroids<<<blocks, threads>>>(d_points, d_centroids, d_cluster_assignments, d_cluster_sizes, N, n, k);

        vector<float> new_centroids(k * n, 0.0f);
        vector<int> cluster_sizes(k, 0);
        cudaMemcpy(new_centroids.data(), d_centroids, k * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(cluster_sizes.data(), d_cluster_sizes, k * sizeof(int), cudaMemcpyDeviceToHost);

        for (int cluster = 0; cluster < k; ++cluster) {
            if (cluster_sizes[cluster] > 0) {
                for (int dim = 0; dim < n; ++dim) {
                    new_centroids[cluster * n + dim] /= cluster_sizes[cluster];
                }
            }
        }

        centroids = new_centroids;
    }

    vector<int> cluster_assignments(N);
    cudaMemcpy(cluster_assignments.data(), d_cluster_assignments, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output file
    ofstream outfile(output_file);
    if (!outfile) {
        cerr << "Error: Unable to open output file." << endl;
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        for (int dim = 0; dim < n; ++dim) {
            outfile << points[i * n + dim] << " ";
        }
        outfile << cluster_assignments[i] << "\n";
    }

    outfile.close();

    // Clean up device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_cluster_assignments);
    cudaFree(d_cluster_sizes);

    return 0;
}