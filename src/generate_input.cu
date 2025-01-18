#include <iostream>
#include <fstream>
#include <curand_kernel.h>
#include <cstdlib>
#include <ctime>

#define ERR(source) (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source), exit(EXIT_FAILURE))

/**
 * @brief CUDA kernel to generate random points in an n-dimensional space.
 *
 * @param output Pointer to the array where generated points are stored.
 * @param N Number of points to generate.
 * @param dims Dimensionality of each point.
 */
__global__ void generate_points(float *output, int N, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState_t state;
    curand_init(clock64(), idx, 0, &state);

    for (int d = 0; d < dims; ++d) {
        output[idx * dims + d] = curand_uniform(&state) * 100.0;  // Points in [0, 100]
    }
}

/**
 * @brief Writes generated points to a file.
 *
 * @param file_path Path to the output file.
 * @param output Pointer to the array containing generated points.
 * @param N Number of points.
 * @param dims Dimensionality of each point.
 */
void write_data_to_file(const char *file_path, float *output, int N, int dims) {
    std::ofstream file(file_path);
    if (!file.is_open()) ERR("ofstream.open");

    file << N << " " << dims << "\n";
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < dims; ++d) {
            file << output[i * dims + d] << " ";
        }
        file << "\n";
    }
    file.close();
}

/**
 * @brief Main function for generating random data points and writing them to a file.
 *
 * @param argc Argument count.
 * @param argv Argument vector. Expected arguments:
 *   - argv[1]: Output file path.
 *   - argv[2]: Number of points to generate (N).
 *   - argv[3]: Dimensionality of points (dims).
 * @return Exit code.
 */
int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <output_file> <N> <dims>\n";
        exit(EXIT_FAILURE);
    }

    const char *file_path = argv[1];
    int N = std::atoi(argv[2]);
    int dims = std::atoi(argv[3]);

    float *output;
    cudaMallocManaged(&output, sizeof(float) * N * dims);
    if (!output) ERR("cudaMallocManaged");

    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    generate_points<<<blocks, threads>>>(output, N, dims);
    cudaDeviceSynchronize();

    write_data_to_file(file_path, output, N, dims);

    cudaFree(output);
    return 0;
}
