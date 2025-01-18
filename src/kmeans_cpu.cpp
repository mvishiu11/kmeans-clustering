#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>

#define ERR(source) (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source), exit(EXIT_FAILURE))

/**
 * @brief Calculates the Euclidean distance between a point and a centroid.
 *
 * @param point A vector representing the coordinates of the point.
 * @param centroid A vector representing the coordinates of the centroid.
 * @param dims The dimensionality of the points and centroids.
 * @return The Euclidean distance between the point and the centroid.
 */
float calculate_distance(const std::vector<float> &point, const std::vector<float> &centroid, int dims) {
    float distance = 0.0;
    for (int d = 0; d < dims; ++d) {
        float diff = point[d] - centroid[d];
        distance += diff * diff;
    }
    return distance;
}

/**
 * @brief Assigns each point to the nearest centroid.
 *
 * @param points A vector of points, where each point is a vector of coordinates.
 * @param centroids A vector of centroids, where each centroid is a vector of coordinates.
 * @param assignments A vector to store the cluster assignment for each point.
 */
void assign_clusters(const std::vector<std::vector<float>> &points, 
                     const std::vector<std::vector<float>> &centroids, 
                     std::vector<int> &assignments) {
    int N = points.size();
    int k = centroids.size();
    int dims = points[0].size();

    for (int i = 0; i < N; ++i) {
        float min_distance = std::numeric_limits<float>::max();
        int best_cluster = 0;

        for (int c = 0; c < k; ++c) {
            float distance = calculate_distance(points[i], centroids[c], dims);
            if (distance < min_distance) {
                min_distance = distance;
                best_cluster = c;
            }
        }
        assignments[i] = best_cluster;
    }
}

/**
 * @brief Recomputes the centroids based on current cluster assignments.
 *
 * @param points A vector of points, where each point is a vector of coordinates.
 * @param centroids A vector of centroids to be updated.
 * @param assignments A vector containing the cluster assignment for each point.
 * @param k The number of clusters.
 */
void recompute_centroids(const std::vector<std::vector<float>> &points, 
                         std::vector<std::vector<float>> &centroids, 
                         const std::vector<int> &assignments, 
                         int k) {
    int dims = points[0].size();
    std::vector<int> cluster_sizes(k, 0);

    // Reset centroids
    for (int c = 0; c < k; ++c) {
        std::fill(centroids[c].begin(), centroids[c].end(), 0.0);
    }

    // Accumulate points in each cluster
    for (int i = 0; i < points.size(); ++i) {
        int cluster = assignments[i];
        for (int d = 0; d < dims; ++d) {
            centroids[cluster][d] += points[i][d];
        }
        cluster_sizes[cluster]++;
    }

    // Compute average for each centroid
    for (int c = 0; c < k; ++c) {
        if (cluster_sizes[c] > 0) {
            for (int d = 0; d < dims; ++d) {
                centroids[c][d] /= cluster_sizes[c];
            }
        }
    }
}

/**
 * @brief Executes the k-means clustering algorithm and writes results to a file.
 *
 * @param points A vector of points, where each point is a vector of coordinates.
 * @param k The number of clusters.
 * @param max_iters The maximum number of iterations.
 * @param output_file The file path to write the clustering results.
 */
void kmeans(const std::vector<std::vector<float>> &points, int k, int max_iters, const char *output_file) {
    int N = points.size();
    int dims = points[0].size();

    // Initialize centroids randomly
    std::srand(std::time(0));
    std::vector<std::vector<float>> centroids(k, std::vector<float>(dims, 0.0));
    for (int c = 0; c < k; ++c) {
        centroids[c] = points[std::rand() % N];
    }

    std::vector<int> assignments(N, 0);

    // Refine centroids through iterations
    for (int iter = 0; iter < max_iters; ++iter) {
        assign_clusters(points, centroids, assignments);
        recompute_centroids(points, centroids, assignments, k);
    }

    // Write results to the output file
    std::ofstream output(output_file);
    if (!output.is_open()) ERR("ofstream.open");

    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < dims; ++d) {
            output << points[i][d] << " ";
        }
        output << assignments[i] << "\n";
    }

    output.close();
}

/**
 * @brief Entry point for the k-means clustering program.
 *
 * @param argc The number of command-line arguments.
 * @param argv The command-line arguments. Expected:
 *   - argv[1]: Input file path containing points.
 *   - argv[2]: Output file path for clustering results.
 *   - argv[3]: Number of clusters (k).
 *   - argv[4]: Maximum number of iterations.
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

    // Load data from the input file
    std::ifstream file(input_file);
    if (!file.is_open()) ERR("ifstream.open");

    int N, dims;
    file >> N >> dims;

    std::vector<std::vector<float>> points(N, std::vector<float>(dims, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < dims; ++d) {
            file >> points[i][d];
        }
    }
    file.close();

    // Run k-means clustering
    kmeans(points, k, max_iters, output_file);

    return 0;
}
