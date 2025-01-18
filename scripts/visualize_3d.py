import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

def visualize_clusters_static(input_file, output_image):
    df = pd.read_csv(input_file, sep=" ", header=None, names=["x", "y", "z", "cluster"])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        df["x"], df["y"], df["z"], 
        c=df["cluster"], cmap="viridis", s=20
    )

    ax.set_title("Static 3D Visualization of Clusters (n=3)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"3D visualization saved as {output_image}")
    plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    visualize_clusters_static("output/results_gpu.txt", "results/3d_visualization.png")
