import sys
import os
import math
import numpy as np
import networkx as nx
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm


def plot_pruned_graph(G_pruned, G_smooth, frame_number=None, save_dir=None):
    pos_pruned = {(row, col): (col, row) for row, col in G_pruned.nodes()}
    pos = {node: (node[1], node[0]) for node in G_smooth.nodes()}
    plt.figure(figsize=(8, 8))
    nx.draw(G_smooth, pos, node_size=10, node_color='gray', with_labels=False)
    nx.draw(G_pruned, pos_pruned, node_size=30, node_color='red', with_labels=False)
    plt.title('Graph Component overlayed on Original Graph')
    plt.gca().invert_yaxis()
    # save_path = os.path.join(save_dir, f'pruned_{frame_number}.png')
    # if save_path:
    #     plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.close()  # Close the figure to free up memory
    plt.show()


def calculate_contour_length(comp):
    length = 0.0
    for u, v in comp.edges():
        length += math.dist(u, v)
    return length


def calculate_begin_to_end_length(comp):
    leaf_nodes = [node for node in comp.nodes() if comp.degree(node) == 1]
    if len(leaf_nodes) != 2:
        return 0.0 
    return math.dist(leaf_nodes[0], leaf_nodes[1])


def calculate_begin_to_end_angle(comp):
    leaf_nodes = [node for node in comp.nodes() if comp.degree(node) == 1]
    if len(leaf_nodes) != 2:
        return 0.0 
    (y1, x1), (y2, x2) = leaf_nodes
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 180
    # if angle_deg > 90:
    #     angle_deg -= 180
    # elif angle_deg < -90:
    #     angle_deg += 180

    return angle_deg


def calculate_fitting_angle(comp):
    ordered_nodes = list(nx.dfs_preorder_nodes(comp))
    if len(ordered_nodes) <= 1:
        return 0.0
    x = np.array([node[1] for node in ordered_nodes])
    y = np.array([node[0] for node in ordered_nodes])
    tck, u = splprep([x, y], s=0.0, k=min(3, len(x)-1))
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_smooth, y_smooth = splev(u_new, tck)
    points = np.column_stack((x_smooth, y_smooth))
    pca = PCA(n_components=2)
    pca.fit(points)
    principal_component = pca.components_[0]
    slope = principal_component[1] / principal_component[0] if principal_component[0] != 0 else np.inf
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)

    # if angle_deg > 90:
    #     angle_deg -= 180
    # elif angle_deg < -90:
    #     angle_deg += 180
    if abs(angle_deg) < 1e-1:
        angle_deg = 0.0
    
    if angle_deg < 0:
        angle_deg += 180

    return angle_deg


def is_diagonal_edge(node1_position, node2_position):
    x1, y1 = node1_position
    x2, y2 = node2_position
    return abs(x1 - x2) == 1 and abs(y1 - y2) == 1


def detect_cycles(graph):
    cycles = list(nx.cycle_basis(graph))
    if not cycles:
        print("No cycles found.")
    else:
        for cycle in cycles:
            cycle_nodes = ", ".join([str(node) for node in cycle])
        for cycle in cycles:
            if len(cycle) == 3:  
                node1, node2, node3 = cycle
                if graph.has_edge(node1, node2) and is_diagonal_edge(node1, node2):
                    graph.remove_edge(node1, node2)
                elif graph.has_edge(node2, node3) and is_diagonal_edge(node2, node3):
                    graph.remove_edge(node2, node3)
                elif graph.has_edge(node1, node3) and is_diagonal_edge(node1, node3):
                    graph.remove_edge(node1, node3)


def create_graph_from_skeleton(frame):
    skeleton = np.array(frame) < 128  
    rows, cols = skeleton.shape
    G = nx.Graph()
    for row in range(rows):
        for col in range(cols):
            if skeleton[row, col]:
                G.add_node((row, col))
                for drow, dcol in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    new_row = row + drow
                    new_col = col + dcol
                    if 0 <= new_row < rows and 0 <= new_col < cols and skeleton[new_row, new_col]:
                        G.add_edge((row, col), (new_row, new_col))
    return G


def create_graphs_for_fiber(graph):
    #components = sorted(nx.connected_components(graph), key=len, reverse=True)
    components = list(sorted(nx.connected_components(graph), key=len, reverse=True))[:10]
    subgraphs = []
    for idx, component in enumerate(components):
        subgraph = graph.subgraph(component).copy()
        subgraphs.append(subgraph)
    return subgraphs


def partition_graph(G):
    components = []  
    visited_edges = set()  

    leaf_nodes = [node for node in G.nodes() if G.degree(node) == 1]
    branch_nodes = [node for node in G.nodes() if G.degree(node) == 3]

    def trace_path(start_node, stop_at_branch=True):
        path = [start_node]
        current_node = start_node
        prev_node = None

        while True:
            neighbors = [n for n in G.neighbors(current_node) if n != prev_node]
            unvisited_neighbors = [n for n in neighbors if (current_node, n) not in visited_edges and (n, current_node) not in visited_edges]

            if not unvisited_neighbors:
                break

            next_node = unvisited_neighbors[0]
            path.append(next_node)
            visited_edges.add((current_node, next_node))
            if G.degree(next_node) == 1 or (stop_at_branch and G.degree(next_node) == 3):
                break

            prev_node = current_node
            current_node = next_node

        return path

    for leaf in leaf_nodes:
        for neighbor in G.neighbors(leaf):
            if (leaf, neighbor) not in visited_edges and (neighbor, leaf) not in visited_edges:
                path = trace_path(leaf, stop_at_branch=True)
                if len(path) > 1:
                    subgraph = G.edge_subgraph([(path[i], path[i + 1]) for i in range(len(path) - 1)]).copy()
                    components.append(subgraph)

    for branch in branch_nodes:
        for neighbor in G.neighbors(branch):
            if (branch, neighbor) not in visited_edges and (neighbor, branch) not in visited_edges:
                path = trace_path(branch, stop_at_branch=True)
                if len(path) > 1:
                    subgraph = G.edge_subgraph([(path[i], path[i + 1]) for i in range(len(path) - 1)]).copy()
                    components.append(subgraph)
    
    components.sort(key=lambda g: g.number_of_edges(), reverse=True)
    return components


def process_components_for_fibers(fiber_subgraphs, component_length_threshold):
    results = []
    for graph_id, graph in enumerate(fiber_subgraphs):
        components = partition_graph(graph)
        for component_id, comp in enumerate(components):
            #plot_pruned_graph(comp, graph)
            contour_length = calculate_contour_length(comp)
            begin_to_end_length = calculate_begin_to_end_length(comp)
            fitting_angle = calculate_fitting_angle(comp)
            begin_to_end_angle = calculate_begin_to_end_angle(comp)
            ##Removing the noise
            if begin_to_end_length < component_length_threshold:
                continue
            component_data = {
                "graph_id": graph_id,
                "component_id_for_graph_id": component_id,
                "contour_length": contour_length,
                "begin_to_end_length": begin_to_end_length,
                "fitting_angle": fitting_angle,
                "begin_to_end_angle": begin_to_end_angle
            }
            results.append(component_data)
      
    return results


def get_angle_length_distribution(results, bin_size):
    """Returns dictionary of angle bins with average lengths"""
    angles = [data["fitting_angle"] for data in results]
    lengths = [data["contour_length"] for data in results]
    bins = np.arange(0, 180 + bin_size, bin_size)
    bin_ranges = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    bin_lengths = {br: [] for br in bin_ranges}    
    for angle, length in zip(angles, lengths):
        for i in range(len(bins)-1):
            if bins[i] <= angle < bins[i+1]:
                bin_range = bin_ranges[i]
                bin_lengths[bin_range].append(length)
                break    
    distribution = {}
    for br in bin_ranges:
        avg = np.mean(bin_lengths[br]) if bin_lengths[br] else 0.0
        distribution[br] = avg    
    return distribution


def generate_3d_surface_plot(df, output_dir, base_name, bin_size, component_length_threshold):
    # Extract y-values (frame numbers)
    y_vals = df['frame'].values  # shape: (num_frames,)
    # Extract x-values (angle bin midpoints)
    angle_bins = df.columns[1:]  # Exclude 'frame' column
    x_vals = []
    for label in angle_bins:
        try:
            left, right = label.split('-')
            midpoint = (float(left) + float(right)) / 2
            x_vals.append(midpoint)
        except:
            print(f"Skipping invalid bin label: {label}")
            continue
    x_vals = np.array(x_vals)
    # Extract Z data (shape: [num_frames, num_bins])
    Z = df.iloc[:, 1:].values.astype(float)
    # Create meshgrid
    X, Y = np.meshgrid(x_vals, y_vals)
    # Plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    # Use X, Y, Z with matching shapes
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, edgecolor='none')
    # Axis labels and title with larger bold font
    ax.set_xlabel("Angle (degrees)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Frame number", fontsize=16, fontweight='bold')
    ax.set_zlabel("Total Length", fontsize=16, fontweight='bold')
    ax.set_title("3D Angle-Length Distribution", fontsize=18, fontweight='bold')
    # Increase and bold axis tick labels
    ax.tick_params(axis='x', labelsize=14)  # Size of tick labels on x-axis
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    # Make tick labels bold
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_zticklabels():
        tick.set_fontweight('bold')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label("Total Length", fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)  # Set size first
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')
    # Save the plot
    output_path = os.path.join(output_dir, f"{base_name}_3d_surface_plot.png")
    plt.savefig(output_path)
    plt.close()
    return output_path


def generate_heatmap(df, output_dir, base_name, bin_size, component_length_threshold):
    """Generate heatmap with angle bins on x-axis and frames on y-axis"""
    fig, ax = plt.subplots(figsize=(20, 12))   
    heatmap_data = df.set_index('frame')
    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        cbar_kws={'label': 'Average Length'},
        yticklabels=50,  # Show every 50th frame tick
        xticklabels=5,    # Show every 5th angle bin tick
        ax=ax,
        square=False      # Allow rectangular cells
    )
    ax.invert_yaxis()    
    ax.set_aspect('auto')    
    ax.set_xlabel(f"Angle Bins ({bin_size}° increments)", fontsize=12)
    ax.set_ylabel("Frame Number", fontsize=12)
    plt.title(f"Fiber Length Distribution by Angle Over Frames: {base_name}", fontsize=14)    
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_locator(plt.MaxNLocator(20))  # Show 20 frame ticks
    heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap_final.png")
    #print("First frame:", heatmap_data.index[0])  
    #print("Last frame:", heatmap_data.index[-1]) 
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    return heatmap_path


def generate_animation(df, output_dir, base_name, bin_size, component_length_threshold):
    """Generate and save animation from matrix data"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x_labels = [float(br.split('-')[0]) + bin_size/2 for br in df.columns[1:]]

    def animate(i):
        ax.clear()
        y_values = df.iloc[i, 1:].values
        ax.bar(x_labels, y_values, width=bin_size*0.8)
        ax.set_title(f"Frame {df.iloc[i]['frame']}")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, df.values[:, 1:].max() * 1.1)
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Average Length")
        return ax,

    ani = FuncAnimation(fig, animate, frames=len(df), interval=100)
    video_path = os.path.join(output_dir, f"{base_name}_evolution.mp4")
    ani.save(video_path, writer='ffmpeg', fps=10)
    plt.close()
    return video_path


def process_tiff_file(tiff_path, output_dir, bin_size, component_length_threshold):
    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    matrix_data = []
    frame_numbers = []
    img = Image.open(tiff_path)
    total_frames = img.n_frames
    print(f'total_frames {total_frames}')
    for frame_idx in range(1, total_frames, 5):
        img.seek(frame_idx)
        frame = img.convert('L')   
        G = create_graph_from_skeleton(frame)
        detect_cycles(G)
        fiber_subgraphs = create_graphs_for_fiber(G)
        results = process_components_for_fibers(fiber_subgraphs, component_length_threshold)             
        distribution = get_angle_length_distribution(results, bin_size)
        matrix_data.append(distribution)
        frame_numbers.append(frame_idx)
        print(f'Processed frame number {frame_idx}')
    
    df = pd.DataFrame(matrix_data)
    df.insert(0, 'frame', frame_numbers)
    csv_path = os.path.join(output_dir, f"{base_name}_matrix.csv")
    df.to_csv(csv_path, index=False)
    heatmap_path = generate_heatmap(df, output_dir, base_name, bin_size, component_length_threshold)
    video_path = generate_animation(df, output_dir, base_name, bin_size, component_length_threshold)
    surface3d_path = generate_3d_surface_plot(df, output_dir, base_name, bin_size, component_length_threshold)

    return {
        'csv_path': csv_path,
        'heatmap_path': heatmap_path,
        'video_path': video_path,
        'surface3d_path': surface3d_path
    }

    
#Example command - python analyzing_angle_length_distribution.py <path to skeleton file> [bin_size] [component_length_threshold]
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_fibers.py <skeleton.tif> [bin_size] [component_length_threshold]")
        sys.exit(1)
    filename = sys.argv[1]
    bin_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    component_length_threshold = int(sys.argv[3]) if len(sys.argv) >= 4 else 10
    output_dir = "fiber_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    tiff_path = filename
    print(f"\nProcessing skeleton file: {filename} Angle Bin Size {bin_size} Length Threshold {component_length_threshold} ")
    result = process_tiff_file(tiff_path, output_dir, bin_size, component_length_threshold)
    if result:
        print(f"Successfully processed {filename}:")
        print(f" - Matrix CSV: {result['csv_path']}")
        if result['heatmap_path']:
            print(f" - Heatmap: {result['heatmap_path']}")
        if result['video_path']:
            print(f" - Animation: {result['video_path']}")
        if result['surface3d_path']:
            print(f" - 3D Heatmap: {result['surface3d_path']}")            


if __name__ == "__main__":
    main()