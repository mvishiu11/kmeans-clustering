import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_benchmark_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Visualization 1: CPU vs. GPU Time Across Configurations
def plot_cpu_vs_gpu_time(df, output_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df.melt(id_vars=["n_points", "dims", "k"], 
                     value_vars=["cpu_time_ms", "gpu_time_ms"],
                     var_name="Device", value_name="Time (ms)"),
                     x="n_points", y="Time (ms)", hue="Device", 
                     style="Device", markers=True, errorbar=None
    )
    plt.title("CPU vs. GPU Execution Time Across Configurations")
    plt.xlabel("Number of Points")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "cpu_vs_gpu_time.png"))
    plt.close()

# Visualization 2: Speedup Factor
def plot_speedup_factor(df, output_path):
    df["speedup"] = df["cpu_time_ms"] / df["gpu_time_ms"]
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, 
        x="n_points", 
        y="speedup", 
        hue="dims", 
        style="dims", 
        markers=True, 
        palette="deep",
        errorbar=None
    )
    
    plt.title("Speedup Factor (CPU vs. GPU)")
    plt.xlabel("Number of Points")
    plt.ylabel("Speedup Factor")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "speedup_factor.png"))
    plt.close()

# Visualization 3: Impact of Dimensionality
def plot_impact_of_dimensionality(df, output_path):
    plt.figure(figsize=(10, 6))
    
    df_melted = df.melt(
        id_vars=["dims", "k"], 
        value_vars=["cpu_time_ms", "gpu_time_ms"], 
        var_name="Device", 
        value_name="Execution Time (ms)"
    )
    
    sns.lineplot(
        data=df_melted, 
        x="dims", 
        y="Execution Time (ms)", 
        hue="Device", 
        style="Device", 
        markers=True,
        errorbar=None
    )
    
    plt.title("Impact of Dimensionality on Execution Time (CPU vs. GPU)")
    plt.xlabel("Dimensionality")
    plt.ylabel("Execution Time (ms)")
    plt.grid(axis="y")
    plt.savefig(os.path.join(output_path, "impact_of_dimensionality.png"))
    plt.close()


# Visualization 4: Impact of Number of Clusters
def plot_impact_of_clusters(df, output_path):
    plt.figure(figsize=(10, 6))
    
    # Combine CPU and GPU times for plotting
    df_melted = df.melt(
        id_vars=["k", "dims"], 
        value_vars=["cpu_time_ms", "gpu_time_ms"], 
        var_name="Device", 
        value_name="Execution Time (ms)"
    )
    
    sns.lineplot(
        data=df_melted, 
        x="k", 
        y="Execution Time (ms)", 
        hue="Device", 
        style="Device", 
        markers=True,
        errorbar=None
    )
    
    plt.title("Impact of Number of Clusters on Execution Time (CPU vs. GPU)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "impact_of_clusters.png"))
    plt.close()


# Visualization 5: Combined Performance Heatmap
def plot_performance_heatmap(df, output_path):
    pivot_table = df.pivot_table(index="dims", columns="n_points", values="gpu_time_ms", aggfunc="mean")
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "GPU Time (ms)"})
    plt.title("Heatmap of GPU Execution Time")
    plt.xlabel("Number of Points")
    plt.ylabel("Dimensionality")
    plt.savefig(os.path.join(output_path, "performance_heatmap.png"))
    plt.close()

def main():
    benchmark_file = os.path.join(RESULTS_DIR, "benchmark_results.json")
    if not os.path.exists(benchmark_file):
        print(f"Benchmark file not found at {benchmark_file}")
        return

    df = load_benchmark_data(benchmark_file)

    print("Generating visualizations...")
    plot_cpu_vs_gpu_time(df, RESULTS_DIR)
    plot_speedup_factor(df, RESULTS_DIR)
    plot_impact_of_dimensionality(df, RESULTS_DIR)
    plot_impact_of_clusters(df, RESULTS_DIR)
    plot_performance_heatmap(df, RESULTS_DIR)

    print("Visualizations saved in the 'results' directory.")

if __name__ == "__main__":
    main()