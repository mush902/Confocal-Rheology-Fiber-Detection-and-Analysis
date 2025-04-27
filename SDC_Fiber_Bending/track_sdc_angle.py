import sys
import os
import math
import numpy as np
import networkx as nx
from PIL import Image
import csv
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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


def get_largest_connected_component(graph):
    components = list(nx.connected_components(graph))
    largest_component = max(components, key=len)
    return graph.subgraph(largest_component).copy()


def find_leaf_nodes(G):
    leaf_nodes = []
    for node in G.nodes():
        if G.degree(node) == 1:
            visited = set()
            stack = [node]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                if G.degree(current) >= 3:
                    leaf_nodes.append((node[0], node[1]))
                    break
                
                for neighbor in G.neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)
    
    return leaf_nodes


def find_edge_leaf_nodes(leaf_nodes):

    """
    color_list = ["red", "blue", "green", "orange"]
    label_list = ["left", "right", "top", "bottom"]
    node[1] is X 
    node[0] is Y
    Y is inverted therefore, minimum Y will appear to be on the top on the plot
    Algorithm should take the top-most node (visually) which happens to be the bottom most if we consider Y. 
    Find the height range between the top-most node (visually) and left most node. 
    When you try to find the top-leftmost node, then always check for the height threshold with the top-most node
    """
    if not leaf_nodes:
        return None, None
    
    # Find leftmost node
    edge_nodes = []
    required_nodes = []
    leftmost_node = leaf_nodes[0]
    for node in leaf_nodes:
        if node[1] < leftmost_node[1]:
            leftmost_node = node
    edge_nodes.append(leftmost_node)

    rightmost_node = leaf_nodes[0]
    for node in leaf_nodes:
        if node[1] > rightmost_node[1]:
            rightmost_node = node
    edge_nodes.append(rightmost_node)

    topmost_node = leaf_nodes[0]
    for node in leaf_nodes:
        if node[0] < topmost_node[0]:
            topmost_node = node
    edge_nodes.append(topmost_node)

    bottommost_node = leaf_nodes[0]
    for node in leaf_nodes:
        if node[0] > bottommost_node[0]:
            bottommost_node = node
    edge_nodes.append(bottommost_node)

    total_height = abs(topmost_node[0] - bottommost_node[0])
    top_height_threshold = math.floor(0.35*total_height)
    #print(f"top_height {total_height} top_height_threshold {top_height_threshold}")
    top_leftmost_node = topmost_node
    for node in leaf_nodes:
        if node[1] < top_leftmost_node[1]:
            #print(f"Node is to left of topmost_node with height {abs(node[0]-topmost_node[0])}")
            if abs(node[0]-topmost_node[0]) < top_height_threshold:
                top_leftmost_node = node
    required_nodes.append(top_leftmost_node)
    
    second_rightmost_node = None
    for node in leaf_nodes:
        if node == rightmost_node:
            continue
        if (second_rightmost_node is None) or (node[1] > second_rightmost_node[1]):
            second_rightmost_node = node

    #print(f"second_rightmost_node {second_rightmost_node}")
    bottom_rightmost_node = None
    total_width = abs(rightmost_node[1] - leftmost_node[1])
    width_threshold = math.floor(0.1 * total_width)
    if rightmost_node[0] > second_rightmost_node[0]:
        bottom_rightmost_node = rightmost_node
    else:
        width = abs(rightmost_node[1]-second_rightmost_node[1])
        if width < width_threshold:
            bottom_rightmost_node = second_rightmost_node
        else:
            bottom_rightmost_node = rightmost_node

    required_nodes.append(bottom_rightmost_node)    
    return edge_nodes, required_nodes


def visualize_graph_for_leaf(bin_frame, G, edge_nodes, required_nodes, length_to_fixed_node, leaf_to_fixed_node_third_point, third_point, right_branch_node):
    plt.figure(figsize=(10, 10))
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.imshow(bin_frame, cmap='gray')
    
    pos = {node: (node[1], node[0]) for node in G.nodes()}
    
    nx.draw(G, pos, node_size=1, node_color="white", edge_color="gray", alpha=0.5, with_labels=False)
    
    idx = 0
    color_list = ["red", "blue", "green", "orange", "violet", "cyan", "black", "grey", "yellow", "yellow", "yellow"]
    label_list = ["left", "right", "top", "bottom", "top_left", "second_right", "third_point_leaf", "third_point_branch", "node_70", "node_120", "node_170"]
    for node in edge_nodes:
        plt.scatter(node[1], node[0], s=200, color=color_list[idx], label=label_list[idx])
        idx = idx + 1
    
    shortest_leaf_to_leaf_path = nx.shortest_path(G, required_nodes[0], required_nodes[1])
    G_shortest_leaf_to_leaf = G.subgraph(shortest_leaf_to_leaf_path)
    for node in required_nodes:
        plt.scatter(node[1], node[0], s=200, color=color_list[idx], label=label_list[idx])
        idx = idx + 1
    pos2 = {node: (node[1], node[0]) for node in G_shortest_leaf_to_leaf.nodes()}
    nx.draw(G_shortest_leaf_to_leaf, pos2, node_size=30, node_color="red", edge_color="red", alpha=0.5, with_labels=False)

    if right_branch_node is not None:
        plt.scatter(right_branch_node[1], right_branch_node[0], s=200, color="brown", label="right_branch_node")
        shortest_leaf_to_branch_path = nx.shortest_path(G, required_nodes[0], right_branch_node)
        G_shortest_leaf_to_branch = G.subgraph(shortest_leaf_to_branch_path)
        pos3 = {node: (node[1], node[0]) for node in G_shortest_leaf_to_branch.nodes()}
        nx.draw(G_shortest_leaf_to_branch, pos3, node_size=10, node_color="lime", edge_color="lime", alpha=0.5, with_labels=False)

    for node in third_point:
        plt.scatter(node[1], node[0], s=200, color=color_list[idx], label=label_list[idx])
        idx = idx + 1
    
    # for length, node in length_to_fixed_node.items():
    #     if node is not None:
    #         plt.scatter(node[1], node[0], s=200, color=color_list[idx], label=label_list[idx])
    #     # third_points = leaf_to_fixed_node_third_point[length]
    #     # for point in third_points:
    #     #     plt.scatter(point[1], point[0], s=100, color='green', label='third_point_new')`
    #     idx = idx + 1
    bold_font = FontProperties(weight='bold', size='large')

    plt.legend(loc='lower right', frameon=True, 
               facecolor='white', edgecolor='black', prop=bold_font)
    
    plt.tight_layout()
    
    plt.axis('off')
    
    plt.title("Skeleton Graph", pad=10)
    #plt.gca().invert_yaxis()
    plt.show()


def find_branch_node(G_shortest_leaf_to_leaf, node, count):
    '''
    count defines which branch point to pick, first or second
    '''
    visited = set()
    stack = [node]
    branch_points_found = []

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        if G_shortest_leaf_to_leaf.degree(current) >= 3:
            branch_points_found.append(current)
            if len(branch_points_found) == count:
                return current

        for neighbor in G_shortest_leaf_to_leaf.neighbors(current):
            if neighbor not in visited:
                stack.append(neighbor)

    return None  # If not enough branch points are found


def get_angle_from_three_points(G, node1, node2):
    """
    Compute the angle ∠node1-node3-node2 in degrees, where node3 is the node in G 
    (excluding node1 and node2) that has the maximum perpendicular distance from the 
    straight line formed between node1 and node2.

    Note:
    - node[1] is the x-coordinate
    - node[0] is the y-coordinate
    """

    def point_line_distance(p, a, b):
        """
        Compute the perpendicular distance from point p to the line formed by points a and b.
        p, a, b are 2D points in (y, x) format.
        """
        # Convert to (x, y) for easier math
        px, py = p[1], p[0]
        ax, ay = a[1], a[0]
        bx, by = b[1], b[0]

        # Line equation: compute area of triangle and base length
        num = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax)
        den = math.hypot(bx - ax, by - ay)
        return num / den if den != 0 else 0

    # Find the node with the maximum distance from line node1-node2
    max_dist = -1
    node3 = None

    for node in G.nodes:
        if node == node1 or node == node2:
            continue
        dist = point_line_distance(node, node1, node2)
        if dist > max_dist:
            max_dist = dist
            node3 = node

    if node3 is None:
        return None  # Could not find a third node

    # Get the vectors node3->node1 and node3->node2
    def vector(a, b):
        return (b[1] - a[1], b[0] - a[0])  # (x2 - x1, y2 - y1)

    v1 = vector(node3, node1)
    v2 = vector(node3, node2)

    # Compute angle using dot product
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0 or mag2 == 0:
        return None  # Degenerate case

    cos_theta = dot / (mag1 * mag2)
    # Clamp value to avoid domain errors due to floating point imprecision
    cos_theta = max(-1, min(1, cos_theta))
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg, node3


def calculate_modulus_and_right_branch(G, required_nodes, extreme_edge_nodes):
    '''
    label_list = ["left", "right", "top", "bottom", "top_left", "bottom_right"]
    '''
    leaf_to_leaf_metric = []
    leaf_to_branch_metric = []
    third_point = []
    shortest_leaf_to_leaf_path = nx.shortest_path(G, required_nodes[0], required_nodes[1])
    G_shortest_leaf_to_leaf = G.subgraph(shortest_leaf_to_leaf_path)
    contour_leaf_to_leaf_length = calculate_contour_length(G_shortest_leaf_to_leaf)
    leaf_to_leaf_metric.append(contour_leaf_to_leaf_length)
    vertical_leaf_to_leaf_height = abs(required_nodes[0][0]-required_nodes[1][0])
    leaf_to_leaf_metric.append(vertical_leaf_to_leaf_height)
    horizontal_leaf_to_leaf_width = abs(required_nodes[0][1]-required_nodes[1][1])
    leaf_to_leaf_metric.append(horizontal_leaf_to_leaf_width)
    leaf_to_leaf_angle, node3 = get_angle_from_three_points(G_shortest_leaf_to_leaf, required_nodes[0], required_nodes[1])
    leaf_to_leaf_metric.append(leaf_to_leaf_angle)
    third_point.append(node3)

    right_node = extreme_edge_nodes[1]
    right_branch_node = None
    if right_node == required_nodes[1]:
        right_branch_node = find_branch_node(G, required_nodes[1], 1)
    else:
        height = abs(right_node[0]-required_nodes[1][0])
        height_threshold = math.floor(0.2*abs(extreme_edge_nodes[2][0] - extreme_edge_nodes[3][0]))
        #print(f"height {height} height_threshold {height_threshold}")
        if height < height_threshold:
            right_branch_node = find_branch_node(G, required_nodes[1], 2)
        else:
            right_branch_node = find_branch_node(G, required_nodes[1], 1)
    
    if right_branch_node is not None:
        shortest_leaf_to_branch_path = nx.shortest_path(G, required_nodes[0], right_branch_node)
        G_shortest_leaf_to_branch = G.subgraph(shortest_leaf_to_branch_path)
        contour_leaf_to_branch_length = calculate_contour_length(G_shortest_leaf_to_branch)
        leaf_to_branch_metric.append(contour_leaf_to_branch_length)
        vertical_leaf_to_branch_height = abs(required_nodes[0][0]-right_branch_node[0])
        leaf_to_branch_metric.append(vertical_leaf_to_branch_height)
        horizontal_leaf_to_branch_width = abs(required_nodes[0][1]-right_branch_node[1])
        leaf_to_branch_metric.append(horizontal_leaf_to_branch_width)
        leaf_to_branch_angle, node3 = get_angle_from_three_points(G_shortest_leaf_to_branch, required_nodes[0], right_branch_node)
        leaf_to_branch_metric.append(leaf_to_branch_angle)
        third_point.append(node3)

    return leaf_to_leaf_metric, leaf_to_branch_metric, third_point, right_branch_node


def find_nodes_at_fixed_length(G, required_nodes, right_branch_node):
    #lengths = [70, 120, 170]
    length_to_node = {}
    for length in range(10, 170, 1):
    #for length in lengths:
        shortest_path = nx.shortest_path(G, required_nodes[0], required_nodes[1])
        positions = {node: (node[0], node[1]) for node in G.nodes()}
        start_node = required_nodes[0]
        x_start = positions[start_node][1]
        prev_node = None
        for node in shortest_path:
            x_curr = positions[node][1]
            dx = x_curr - x_start

            if dx == length:
                length_to_node[length] = node
                break
            elif dx > length and prev_node is not None:
                y1, x1 = positions[prev_node]
                y2, x2 = positions[node]

                if x2 == x1:
                    continue

                ratio = (length - (x1 - x_start)) / (x2 - x1)
                y_new = y1 + ratio * (y2 - y1)
                x_new = x1 + ratio * (x2 - x1)

                new_node = f"interpolated_{length}"
                while new_node in G:
                    new_node += "_1"

                G.add_node(new_node, pos=(y_new, x_new))
                G.add_edge(prev_node, new_node)
                G.add_edge(new_node, node)
                if G.has_edge(prev_node, node):
                    G.remove_edge(prev_node, node)

                length_to_node[length] = new_node
                break

            prev_node = node
        else:
            print(f'appending None Why ?')
            print(f'The horizontal distance from end to end is {required_nodes[0][1]-required_nodes[1][1]}')
            length_to_node[length] = None
    return length_to_node


def plot_length_metrics(length_metrics):
    contour_lengths = [metric[0] for metric in length_metrics]
    vertical_heights = [metric[1] for metric in length_metrics]
    horizontal_widths = [metric[2] for metric in length_metrics]
    angles = [metric[3] for metric in length_metrics]
    frames = list(range(1, len(length_metrics) + 1))

    plt.figure(figsize=(12, 6))
    #plt.plot(frames, contour_lengths, label='Contour Length', marker='o')
    #plt.plot(frames, vertical_heights, label='Vertical Height', marker='s')
    #plt.plot(frames, horizontal_widths, label='Horizontal Width', marker='^')
    plt.plot(frames, angles, label='3-point Angle', marker='x')
    
    plt.xlabel('Frame Index')
    plt.ylabel('Length (pixels)')
    plt.title('Angle Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def write_leaf_to_fixed_node_metric(leaf_to_fixed_nodes_metrics, output_dir):
    contour_lengths = defaultdict(list)
    vertical_heights = defaultdict(list)
    angles = defaultdict(list)
    for metrics_dict in leaf_to_fixed_nodes_metrics:
        for length, length_metrics in metrics_dict.items():
            if not isinstance(length_metrics, list) or len(length_metrics) < 4:
                continue
            contour_lengths[length].append(length_metrics[0])
            vertical_heights[length].append(length_metrics[1])
            angles[length].append(length_metrics[3])
    output_path = os.path.join(output_dir, 'leaf_to_fixed_node_metric.csv')
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['length', 'contour_length', 'vertical_heights', 'angles'])
        for length in sorted(contour_lengths.keys()):
            cl_values = contour_lengths[length]
            vh_values = vertical_heights[length]
            angle_values = angles[length]
            for i in range(len(cl_values)):
                writer.writerow([
                    length,
                    cl_values[i],
                    vh_values[i],
                    angle_values[i]
                ])
    linear_fit_slopes = {}  
    for length, angle_list in angles.items():
        frames = np.arange(1, len(angle_list) + 1)
        angle_array = np.array(angle_list)
        
        if len(frames) >= 2:
            slope, intercept = np.polyfit(frames, angle_array, 1)
            linear_fit_slopes[length] = (slope, intercept)

    sorted_lengths = sorted(linear_fit_slopes.keys())
    with open(os.path.join(output_dir, 'linear_fitted_slope_vs_lengths.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['length', 'slopes'])

        for length in sorted_lengths:
            slope, _ = linear_fit_slopes[length]
            writer.writerow([length, slope])


def plot_leaf_to_fixed_node_metric(leaf_to_fixed_nodes_metrics, output_dir):    
    contour_lengths = defaultdict(list)
    vertical_heights = defaultdict(list)
    horizontal_widths = defaultdict(list)
    angles = defaultdict(list)

    for metrics_dict in leaf_to_fixed_nodes_metrics:
        for length, length_metrics in metrics_dict.items():
            if not isinstance(length_metrics, list) or len(length_metrics) < 4:
                continue
            contour_lengths[length].append(length_metrics[0])
            vertical_heights[length].append(length_metrics[1])
            horizontal_widths[length].append(length_metrics[2])
            angles[length].append(length_metrics[3])
    
    linear_fit_slopes = {}  
    for length, angle_list in angles.items():
        frames = np.arange(1, len(angle_list) + 1)
        angle_array = np.array(angle_list)
        
        if len(frames) >= 2:
            slope, intercept = np.polyfit(frames, angle_array, 1)
            linear_fit_slopes[length] = (slope, intercept)

    sorted_lengths = sorted(linear_fit_slopes.keys())
    slopes, intercepts = zip(*[linear_fit_slopes[length] for length in sorted_lengths])

    # Set up figure
    plt.figure(figsize=(12, 8))  # Increased size to reduce whitespace
    plt.plot(sorted_lengths, slopes, marker='o', label='Linear Fit Slope of Angle')

    # Axis labels with larger bold fonts
    plt.xlabel('Length', fontsize=16, fontweight='bold')
    plt.ylabel('Angle Change Rate (Slope)', fontsize=16, fontweight='bold')
    plt.title('Linear Fit of Angle vs Length', fontsize=18, fontweight='bold')

    # Tick label font size and boldness
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Grid and legend
    plt.grid(True)

    # Legend with bold, large font
    legend_font = FontProperties(weight='bold', size='large')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='black', prop=legend_font)

    plt.tight_layout()

    save_path = os.path.join(output_dir, f'Linear_fit_of_SDC_Angles.png')
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.close()
    
    plt.show()
    # for length, angle in angles.items():
    #     frames = list(range(1, len(angle) + 1))
    #     slope, intercept = linear_fit_slopes[length]
    #     p = np.poly1d([slope, intercept])

    #     plt.figure(figsize=(12, 6))  # Larger figure

    #     # Plot angle and its linear fit
    #     plt.plot(frames, angle, label='Angle', marker='o')
    #     plt.plot(frames, p(frames), label='Linear Fit', color='red', linewidth=5)
    #     # Axis labels and title with bold styling
    #     plt.xlabel('Frame Index', fontsize=16, fontweight='bold')
    #     plt.ylabel('Angle (degrees)', fontsize=16, fontweight='bold')
    #     plt.title(f'Angle Over Frames for Fixed Length {length}', fontsize=18, fontweight='bold')

    #     # Bold, large tick labels
    #     plt.xticks(fontsize=14, fontweight='bold')
    #     plt.yticks(fontsize=14, fontweight='bold')

    #     # Bold legend with styled box
    #     legend_font = FontProperties(weight='bold', size='large')
    #     plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='black', prop=legend_font)

    #     # Grid and layout
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()


def find_leaf_to_fixed_node_metric(G, length_to_fixed_node, required_nodes):
    leaf_to_fixed_node_metric = {}
    leaf_to_fixed_node_third_point = {}
    for length, node in length_to_fixed_node.items():
        leaf_to_leaf_metric = []
        third_point = []
        if node is None:
            leaf_to_fixed_node_metric[length] = leaf_to_leaf_metric
            continue
        shortest_leaf_to_leaf_path = nx.shortest_path(G, required_nodes[0], node)
        G_shortest_leaf_to_leaf = G.subgraph(shortest_leaf_to_leaf_path)
        contour_leaf_to_leaf_length = calculate_contour_length(G_shortest_leaf_to_leaf)
        leaf_to_leaf_metric.append(contour_leaf_to_leaf_length)
        vertical_leaf_to_leaf_height = abs(required_nodes[0][0]-node[0])
        leaf_to_leaf_metric.append(vertical_leaf_to_leaf_height)
        horizontal_leaf_to_leaf_width = abs(required_nodes[0][1]-node[1])
        leaf_to_leaf_metric.append(horizontal_leaf_to_leaf_width)
        leaf_to_leaf_angle, node3 = get_angle_from_three_points(G_shortest_leaf_to_leaf, required_nodes[0], node)
        leaf_to_leaf_metric.append(leaf_to_leaf_angle)
        third_point.append(node3)
        leaf_to_fixed_node_metric[length] = leaf_to_leaf_metric
        leaf_to_fixed_node_third_point[length] = third_point
    return leaf_to_fixed_node_metric, leaf_to_fixed_node_third_point



def process_tiff_file(skeleton_file, binarised_file, output_dir):
    skeleton_img = Image.open(skeleton_file)
    binary_img = Image.open(binarised_file)
    total_frames = skeleton_img.n_frames
    print(f'total_frames {total_frames}')
    leaf_to_leaf_metrics = []
    leaf_to_branch_metrics = []
    leaf_to_fixed_nodes_metrics = []
    for frame_idx in range(0, total_frames, 100):
        print(f"in frame_idx {frame_idx}")
        skeleton_img.seek(frame_idx)
        skeleton_frame = skeleton_img.convert('L')
        binary_img.seek(frame_idx)
        binary_frame = binary_img.convert('L')
        # Original processing pipeline
        G = create_graph_from_skeleton(skeleton_frame)
        detect_cycles(G)
        G_largest = get_largest_connected_component(G)
        leaf_nodes = find_leaf_nodes(G_largest)
        extreme_edge_nodes, required_nodes = find_edge_leaf_nodes(leaf_nodes)
        leaf_to_leaf_metric, leaf_to_branch_metric, third_point, right_branch_node = calculate_modulus_and_right_branch(G_largest, required_nodes, extreme_edge_nodes)
        length_to_fixed_node = find_nodes_at_fixed_length(G_largest, required_nodes, right_branch_node)
        leaf_to_fixed_node_metric, leaf_to_fixed_node_third_point = find_leaf_to_fixed_node_metric(G_largest, length_to_fixed_node, required_nodes)
        leaf_to_leaf_metrics.append(leaf_to_leaf_metric)
        leaf_to_branch_metrics.append(leaf_to_branch_metric)
        leaf_to_fixed_nodes_metrics.append(leaf_to_fixed_node_metric)
        #print("After find_edge_leaf_nodes")
        #visualize_graph_for_leaf(binary_frame, G_largest, extreme_edge_nodes, required_nodes, length_to_fixed_node, leaf_to_fixed_node_third_point, third_point, right_branch_node)
        
    #plot_length_metrics(leaf_to_leaf_metrics) 
    #plot_length_metrics(leaf_to_branch_metrics)
    write_leaf_to_fixed_node_metric(leaf_to_fixed_nodes_metrics, output_dir)
    plot_leaf_to_fixed_node_metric(leaf_to_fixed_nodes_metrics, output_dir)         


#Example Command - python track_sdc_angle.py <input skeleton file> <input confocal binarised file>
def main():
    if len(sys.argv) < 3:
        print("Usage: python track_sdc_angle.py <skeleton.tif> <binarised.tif>")
        sys.exit(1)
    skeleton_file = sys.argv[1]
    binarised_file = sys.argv[2]
    output_dir = "fiber_analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing skeleton file: {skeleton_file} binarised confocal file {binarised_file}")
    process_tiff_file(skeleton_file, binarised_file, output_dir)


if __name__ == "__main__":
    main()
