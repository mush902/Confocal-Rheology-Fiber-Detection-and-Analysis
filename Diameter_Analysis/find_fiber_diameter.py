import sys
import os
import math
import numpy as np
import networkx as nx
from PIL import Image
import csv
import matplotlib.pyplot as plt
from collections import defaultdict


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


def extend_until_white_float(start_y, start_x, dy, dx, frame, max_length=100):
    for i in range(1, max_length):
        y = start_y + i * dy
        x = start_x + i * dx
        yi, xi = int(round(y)), int(round(x))
        if yi < 0 or yi >= frame.shape[0] or xi < 0 or xi >= frame.shape[1]:
            break
        if frame[yi, xi] > 0:
            return i
    return i - 1


def interpolate_point(p1, p2, ratio):
    y1, x1 = p1
    y2, x2 = p2
    return (y1 + ratio * (y2 - y1), x1 + ratio * (x2 - x1))


def find_middle_and_perpendicular(G, binarised_frame, unit_distance=1.0, perp_length=10, max_extension=100):
    endpoints = [n for n in G.nodes if G.degree(n) == 1]
    if len(endpoints) != 2:
        raise ValueError("Graph must have exactly 2 endpoints.")
    path = nx.shortest_path(G, source=endpoints[0], target=endpoints[1])
    distances = [0.0]
    for i in range(1, len(path)):
        prev, curr = path[i-1], path[i]
        d = math.dist(prev, curr)
        distances.append(distances[-1] + d)
    total_length = distances[-1]
    half_length = total_length / 2
    before_dist = max(0, half_length - unit_distance)
    after_dist = min(total_length, half_length + unit_distance)

    def get_interpolated_point(target_dist):
        for i in range(1, len(distances)):
            if distances[i-1] <= target_dist <= distances[i]:
                seg_len = distances[i] - distances[i-1]
                ratio = (target_dist - distances[i-1]) / seg_len
                return interpolate_point(path[i-1], path[i], ratio)
        return path[-1]
    
    middle = get_interpolated_point(half_length)
    middle_minus_1 = get_interpolated_point(before_dist)
    middle_plus_1 = get_interpolated_point(after_dist)

    # Compute perpendicular line
    y1, x1 = middle_minus_1
    y2, x2 = middle_plus_1

    if x2 - x1 == 0:
        perp_end1 = (middle[0], middle[1] + perp_length / 2)
        perp_end2 = (middle[0], middle[1] - perp_length / 2)
    else:
        slope = (y2 - y1) / (x2 - x1)
        perp_slope = -1 / slope if slope != 0 else float('inf')

        dx = perp_length / 2 / math.sqrt(1 + perp_slope**2) if perp_slope != float('inf') else 0
        dy = perp_slope * dx if perp_slope != float('inf') else perp_length / 2

        perp_end1 = (middle[0] + dy, middle[1] + dx)
        perp_end2 = (middle[0] - dy, middle[1] - dx)
    
    #Extend the line till you reach white pixel on the overlay
    if x2 - x1 == 0:
        perp_dx = 1
        perp_dy = 0
    else:
        slope = (y2 - y1) / (x2 - x1)
        perp_slope = -1 / slope if slope != 0 else float('inf')

        if perp_slope == float('inf'):
            perp_dx = 0
            perp_dy = 1
        else:
            perp_dx = 1 / math.sqrt(1 + perp_slope**2)
            perp_dy = perp_slope * perp_dx

    mid_y, mid_x = middle
    frame_np = np.array(binarised_frame)

    # Extend in both directions
    len1 = extend_until_white_float(mid_y, mid_x, perp_dy, perp_dx, frame_np, max_extension)
    len2 = extend_until_white_float(mid_y, mid_x, -perp_dy, -perp_dx, frame_np, max_extension)

    end1 = (mid_y + perp_dy * len1, mid_x + perp_dx * len1)
    end2 = (mid_y - perp_dy * len2, mid_x - perp_dx * len2)

    diameter = len1 + len2

    return {
        "middle_node": middle,
        "middle_minus_1": middle_minus_1,
        "middle_plus_1": middle_plus_1,
        "perpendicular_line": [perp_end1, perp_end2],
        "extended_perpendicular_line": [end1, end2],
        "diameter": diameter, 
        "total_length":total_length
    }


def calculate_contour_length(comp):
    length = 0.0
    for u, v in comp.edges():
        length += math.dist(u, v)
    return length


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
    components = list(sorted(nx.connected_components(graph), key=len, reverse=True))
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


def process_components_for_fibers(fiber_subgraphs, binarised_frame):
    results = []
    for graph_id, graph in enumerate(fiber_subgraphs):
        components = partition_graph(graph)
        for component_id, comp in enumerate(components):
            contour_length = calculate_contour_length(comp)
            if contour_length < 10:
                continue
            diameter_auxillary_data = find_middle_and_perpendicular(comp, binarised_frame)
            component_data = {
                "graph_id": graph_id,
                "component_id_for_graph_id": component_id,
                "component": comp, 
                "diameter_auxillary_data": diameter_auxillary_data
            }
            results.append(component_data)

    return results        


def plot_components_for_fiber(result, binarised_frame):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(binarised_frame, cmap="gray")

    '''
    component_data = {
        "graph_id": graph_id,
        "component_id_for_graph_id": component_id,
        "component": comp, 
        "diameter_auxillary_data": diameter_auxillary_data
    }
    diameter_auxillary_data =  {
        "middle_node": middle,
        "middle_minus_1": middle_minus_1,
        "middle_plus_1": middle_plus_1,
        "perpendicular_line": [perp_end1, perp_end2],
        "extended_perpendicular_line": [end1, end2],
        "diameter": diameter
    }
    '''
    for component_data in result:
        component = component_data["component"]
        auxillary_data = component_data["diameter_auxillary_data"]
        pos = {node: (node[1], node[0]) for node in component.nodes()}
        nx.draw(component, pos, node_size=20, node_color="red", edge_color="gray", alpha=0.5, with_labels=False)
        middle_node = auxillary_data["middle_node"]
        # minus_one = middle_nodes["middle_minus_1"]
        # plus_one = middle_nodes["middle_plus_1"]
        plt.scatter(middle_node[1], middle_node[0], s=20, color="cyan")
        # plt.scatter(minus_one[1], minus_one[0], s=20, color="cyan")
        # plt.scatter(plus_one[1], plus_one[0], s=20, color="cyan")
        perp_line_extended = auxillary_data["extended_perpendicular_line"]
        (y1, x1), (y2, x2) = perp_line_extended
        plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=5, linestyle='-')

        # perp_line = auxillary_data["perpendicular_line"]
        # (y1, x1), (y2, x2) = perp_line
        # plt.plot([x1, x2], [y1, y2], color='cyan', linewidth=2, linestyle='-')
        # plt.scatter(x1, y1, s=20, color="cyan")
        #plt.scatter(x2, y2, s=20, color="cyan")

    plt.tight_layout()
    plt.axis('off')
    plt.title("Skeleton Graph", pad=10)
    #plt.gca().invert_yaxis()
    plt.show()


def write_diameters_to_csv(result, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'component_diameters.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['graph_id', 'component_id', 'diameter'])
        diameters_length_by_graph = defaultdict(list)
        all_diameters = []
        for component_data in result:
            graph_id = component_data['graph_id']
            component_id = component_data['component_id_for_graph_id']
            diameter = component_data['diameter_auxillary_data']['diameter']
            length = component_data['diameter_auxillary_data']['total_length']
            writer.writerow([graph_id, component_id, diameter])
            diameters_length_by_graph[graph_id].append((diameter, length))
            all_diameters.append(diameter)
        
        writer.writerow([])
        writer.writerow(['graph_id', 'average_diameter', 'total_length'])
        for graph_id, data in diameters_length_by_graph.items():
            diameters = [d for d, _ in data]
            lengths = [l for _, l in data]
            avg = sum(diameters) / len(diameters) if diameters else 0
            total_length = sum(lengths)
            writer.writerow([graph_id, f"{avg:.4f}", total_length])

        # Empty row and overall average
        writer.writerow([])
        overall_avg = sum(all_diameters) / len(all_diameters) if all_diameters else 0
        writer.writerow(['overall_average_diameter', f"{overall_avg:.4f}"])

    # print("Average diameter per graph:")
    # for graph_id, diameters in diameters_length_by_graph.items():
    #     avg = sum(diameters) / len(diameters)
    #     print(f"Graph ID {graph_id}: {avg:.4f}")

    # overall_avg = sum(all_diameters) / len(all_diameters) if all_diameters else 0
    # print(f"\nOverall average diameter: {overall_avg:.4f}")


def process_tiff_file(skeleton_file, binarised_file, output_dir):
    skeleton_img = Image.open(skeleton_file)
    binarised_img = Image.open(binarised_file)
    total_frames = skeleton_img.n_frames
    print(f'total_frames {total_frames}')
    for frame_idx in range(0, total_frames, 1):
        skeleton_img.seek(frame_idx)
        skeleton_frame = skeleton_img.convert('L')
        binarised_img.seek(frame_idx)
        binarised_frame = binarised_img.convert('L')
        G = create_graph_from_skeleton(skeleton_frame)
        detect_cycles(G)
        fiber_subgraphs = create_graphs_for_fiber(G)
        result = process_components_for_fibers(fiber_subgraphs, binarised_frame)
        plot_components_for_fiber(result, binarised_frame)
        print(f'Processed frame number {frame_idx}')
        write_diameters_to_csv(result, output_dir)


#Example Command -  python find_fiber_diameter.py <path to skeleton file> <path to binarised file>
def main():
    if len(sys.argv) < 3:
        print("Usage: python find_fiber_diameter.py <skeleton.tif> <binarised.tif>")
        sys.exit(1)
    skeleton_file = sys.argv[1]
    binarised_file = sys.argv[2]
    output_dir = "fiber_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing skelton file: {skeleton_file} binarised file {binarised_file}")
    process_tiff_file(skeleton_file, binarised_file, output_dir)           


if __name__ == "__main__":
    main()