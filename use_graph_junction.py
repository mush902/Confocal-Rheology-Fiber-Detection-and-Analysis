import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys


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


def is_diagonal_edge(node1, node2):
    return abs(node1[0] - node2[0]) == 1 and abs(node1[1] - node2[1]) == 1


def detect_cycles(graph):
    cycles = list(nx.cycle_basis(graph))
    for cycle in cycles:
        if len(cycle) == 3:
            node1, node2, node3 = cycle
            if graph.has_edge(node1, node2) and is_diagonal_edge(node1, node2):
                graph.remove_edge(node1, node2)
            elif graph.has_edge(node2, node3) and is_diagonal_edge(node2, node3):
                graph.remove_edge(node2, node3)
            elif graph.has_edge(node1, node3) and is_diagonal_edge(node1, node3):
                graph.remove_edge(node1, node3)


def process_graph(G, confocal_frame, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_output_path = os.path.join(output_dir, 'junction_nodes.csv')
    degree_distribution_csv = os.path.join(output_dir, 'degree_distribution.csv')
    junction_nodes = []
    degree_count = {}

    for node in G.nodes():
        degree = G.degree(node)
        if degree >= 3:
            junction_nodes.append((node[0], node[1], degree))
        degree_count[degree] = degree_count.get(degree, 0) + 1

    junction_df = pd.DataFrame(junction_nodes, columns=["X", "Y", "Degree"])
    junction_df.to_csv(csv_output_path, index=False)

    width, height = confocal_frame.size 
    total_area = width * height  
    print(f"Image dimensions: {width} x {height} pixels")
    print(f"Total pixel area: {total_area} pixels²")

    degree_dist_df = pd.DataFrame(list(degree_count.items()), columns=["Degree", "Total Pixel Count"])
    degree_dist_df["Total Area"] = total_area
    degree_dist_df.to_csv(degree_distribution_csv, index=False)

    return junction_df


def visualize_graph(G, junction_df, frame):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(frame)
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    nx.draw(G, pos, node_size=1, node_color="white", edge_color="gray", alpha=0.5, with_labels=False)
    degree_3 = junction_df[junction_df["Degree"] == 3]
    degree_4 = junction_df[junction_df["Degree"] == 4]
    degree_5_or_more = junction_df[junction_df["Degree"] >= 5]
    plt.scatter(degree_3["Y"], degree_3["X"], s=20, color="red", label="Degree = 3")
    plt.scatter(degree_4["Y"], degree_4["X"], s=20, color="green", label="Degree = 4")
    plt.scatter(degree_5_or_more["Y"], degree_5_or_more["X"], s=20, color="blue", label="Degree ≥ 5")
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True, 
               facecolor='white', edgecolor='black', fontsize='small')
    plt.tight_layout()    
    plt.axis('off')
    plt.title("Skeleton Graph with Junction Nodes", pad=10)
    plt.gca().invert_yaxis()
    plt.show()


def process_tiff_file(skeleton_file, confocal_file, output_dir):
    skeleton_img = Image.open(skeleton_file)
    confocal_img = Image.open(confocal_file)
    total_frames = skeleton_img.n_frames
    print(f'total_frames {total_frames}')
    for frame_idx in range(0, total_frames, 1):
        skeleton_img.seek(frame_idx)
        skeleton_frame = skeleton_img.convert('L')
        confocal_img.seek(frame_idx)
        confocal_frame = confocal_img.copy()
        G = create_graph_from_skeleton(skeleton_frame)
        detect_cycles(G)
        junction_df = process_graph(G, confocal_frame, output_dir)
        visualize_graph(G, junction_df, confocal_frame)


#Example command to launch run - python3 use_graph_junction.py <path to skeleton file> <path to confocal file>
def main():
    if len(sys.argv) < 3:
        print("Usage: python use_graph_junction.py <skeleton.tif> <confocal.tif>")
        sys.exit(1)
    skeleton_file = sys.argv[1]
    confocal_file = sys.argv[2]
    output_dir = "junction_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing skelton file: {skeleton_file} confocal file {confocal_file}")
    process_tiff_file(skeleton_file, confocal_file, output_dir) 


if __name__ == "__main__":
    main()
