import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from collections import defaultdict
import colorsys
import random
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
import os
from PIL import Image
from collections import deque
import itertools
import math
import csv

def load_data(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    filtered_df = df[df['Value'] == 255]
    filtered_x = filtered_df['X'].values
    filtered_y = filtered_df['Y'].values
    return df, filtered_x, filtered_y


def add_interpolated_nodes(graph, max_edge_length=1.0):
    new_graph = nx.Graph()
    new_graph.add_nodes_from(graph.nodes())  
    for edge in graph.edges():
        node1, node2 = edge
        x1, y1 = node1
        x2, y2 = node2
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        num_interpolated_points = int(np.ceil(dist / max_edge_length))  
        for i in range(1, num_interpolated_points):
            t = i / num_interpolated_points
            interpolated_x = x1 + t * (x2 - x1)
            interpolated_y = y1 + t * (y2 - y1)
            new_node = (interpolated_x, interpolated_y)
            if not new_graph.has_node(new_node):
                new_graph.add_node(new_node)
            if i == 1:
                new_graph.add_edge(node1, new_node)  
            else:
                new_graph.add_edge(prev_node, new_node)  
            prev_node = new_node
        new_graph.add_edge(prev_node, node2)
    return new_graph


def extract_node_positions(graph):
    nodes = list(graph.nodes())
    x = np.array([node[1] for node in nodes])  
    y = np.array([node[0] for node in nodes])  
    return x, y


def calculate_gradient_curvature(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    gradient_curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    return gradient_curvature


def calculate_menger_curvature(x, y):
    def menger(a, b, c):
        area = 0.5 * np.abs(np.cross(b-a, c-a))
        return 2 * area / (np.linalg.norm(b-a) * np.linalg.norm(c-b) * np.linalg.norm(a-c))
    points = np.column_stack((x, y))
    menger_curvature = np.array([menger(points[i-1], points[i], points[i+1]) for i in range(1, len(points)-1)])
    return menger_curvature


def plot_original_vs_smoothed(filtered_x, filtered_y, G_smooth, node_colors):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(filtered_x, filtered_y, c='blue', marker='o')
    ax[0].set_title('Filtered Points (Original)')
    ax[0].invert_yaxis()
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    pos_smooth = {(x, y): (y, x) for (x, y) in G_smooth.nodes()}  
    colors = [node_colors.get(node, 'black') for node in G_smooth.nodes()]  
    nx.draw(G_smooth, pos_smooth, node_size=10, ax=ax[1], node_color=colors, with_labels=False)
    ax[1].set_title('Smoothed Graph (Color-Coded Nodes)')
    ax[1].invert_yaxis()
    plt.tight_layout()
    #plt.show()


def get_largest_connected_component(graph):
    components = list(nx.connected_components(graph))
    largest_component = max(components, key=len)
    return graph.subgraph(largest_component).copy()


def plot_graph_with_key_value_nodes(G_smooth, connected_neighbors):
    plt.figure(figsize=(6, 6))
    pos_smooth = {(x, y): (y, x) for (x, y) in G_smooth.nodes()}  
    nx.draw(G_smooth, pos_smooth, node_size=50, node_color='gray', with_labels=False)
    key_nodes = list(connected_neighbors.keys())  
    value_nodes = [val for sublist in connected_neighbors.values() for val in sublist]  
    nx.draw_networkx_nodes(
        G_smooth, 
        pos_smooth, 
        nodelist=key_nodes, 
        node_size=100,  
        node_color='red',  
        edgecolors='black'  
    )
    nx.draw_networkx_nodes(
        G_smooth, 
        pos_smooth, 
        nodelist=value_nodes, 
        node_size=50,  
        node_color='black'  
    )
    nx.draw_networkx_edges(G_smooth, pos_smooth)
    plt.tight_layout()
    plt.show()


def generate_distinct_colors(num_colors):
    hue_step = 1.0 / num_colors
    colors = []
    for i in range(num_colors):
        hue = i * hue_step
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(tuple(int(x * 255) for x in rgb))
    random.shuffle(colors)
    return colors


def get_random_color(index):
    distinct_colors = [
        "red",
        "blue",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "brown",
        "navy",
        "teal",
        "maroon",
        "olive",
        "lime",
        "indigo",
        "crimson",
        "gold",
        "turquoise",
        "violet",
        "salmon",
        "sienna",
        "orchid",
        "khaki",
        "plum",
        "coral",
        "peru",
        "tomato",
        "chocolate",
        "firebrick",
        "dodgerblue"
    ]
    color_index = index % len(distinct_colors)
    return distinct_colors[color_index]


def plot_graph_components(G_smooth_original, components, branch_node_colors):
    pos = {(row, col): (col, row) for row, col in G_smooth_original.nodes()}  
    for idx, component in components.items():
        plt.figure(figsize=(8, 8))
        nx.draw(G_smooth_original, pos, node_size=30, node_color='gray', with_labels=False)
        plt.title('Graph with Color-Coded Branch Components')
        color = 'yellow'
        if idx < len(branch_node_colors):
            color = branch_node_colors[idx]
        coordinates = [pos[node] for node in component if node in pos]  
        if coordinates:  
            x, y = zip(*coordinates)
            plt.scatter(x, y, label=f'Component {idx}', color=color, s=100)  
            plt.xlabel('X Coordinate (Swapped)')
            plt.ylabel('Y Coordinate (Swapped)')
            plt.legend()
            plt.grid()
            plt.axis('equal')  
            plt.gca().invert_yaxis()
            plt.show()


def color_graph_sections(G_smooth, G_smooth_original):
    leaf_nodes = [node for node in G_smooth.nodes if G_smooth.degree(node) == 1]
    branch_points = [node for node in G_smooth.nodes if G_smooth.degree(node) > 2]
    node_colors = {}
    components = {}
    color_index = 0
    colors = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 0.0, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.5, 0.0),  # orange
        (1.0, 0.0, 1.0),  # pink (magenta)
        (0.5, 0.0, 0.5),  # violet
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 0.0, 1.0),  # pink (magenta)
    ]
    
    def dfs_color_section(node, color):
        stack = [node]
        visited = set()
        component_nodes = []
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                node_colors[current_node] = color
                component_nodes.append(current_node)
                for neighbor in G_smooth.neighbors(current_node):
                    if neighbor not in visited:
                        if G_smooth.degree(neighbor) > 2: 
                            return component_nodes
                        stack.append(neighbor)
        return component_nodes
    for leaf in leaf_nodes:
        if leaf not in node_colors:
            color = colors[color_index % len(colors)]
            component = dfs_color_section(leaf, color)
            components[color_index] = component
            color_index += 1
    # for branch in branch_points:
    #     if branch not in node_colors:
    #         color = colors[color_index % len(colors)]
    #         component = dfs_color_section(branch, color)
    #         components[color_index] = component
    #         plot_graph_components(G_smooth, components, colors)
    #         color_index += 1
    branch_components = {}
    branch_color_index = 0
    branch_node_colors = {}
    
    
    def branch_dfs_color_section(node, color, component, visited):
        stack = [node]
        component_nodes = []
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                component_nodes.append(current_node)
                for neighbor in G_smooth_original.neighbors(current_node):
                    if neighbor in component:
                        if neighbor not in visited:
                            if G_smooth_original.degree(neighbor) > 2:
                                return component_nodes
                            stack.append(neighbor)
        return component_nodes
    branch_visited = set()
    for color_index, component in components.items():
        branch_nodes = []
        for node in component:
            if node in G_smooth_original.nodes():
                if G_smooth_original.degree(node) > 2:
                    branch_nodes.append(node)
        #print(f'lengths of branch node for this component - {color_index} is {len(branch_nodes)}')
        for branch_node in branch_nodes:
            color = get_random_color(branch_color_index)
            branch_node_colors[branch_color_index] = color
            branch_component = branch_dfs_color_section(branch_node, color, component, branch_visited)
            branch_components[branch_color_index] = branch_component
            branch_color_index += 1
    default_color = colors[-1]
    for node in G_smooth.nodes():
        if node not in node_colors:
            node_colors[node] = default_color
    return branch_node_colors, branch_components


def plot_colored_graph(G_smooth, node_colors):
    
    pos = {(row, col): (col, row) for row, col in G_smooth.nodes()}  
    colors = [node_colors.get(node, 'black') for node in G_smooth.nodes()]  
    plt.figure(figsize=(8, 8))
    nx.draw(G_smooth, pos, node_size=30, node_color=colors, with_labels=False)
    plt.title('Graph with Color-Coded Sections')
    plt.gca().invert_yaxis()  
    #plt.show()


def order_nodes(G):
    """
    Order nodes using BFS starting from the bottom-left node.
    """
    start_node = min(G.nodes(), key=lambda n: (n[1], -n[0]))
    ordered_nodes = []
    visited = set()
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            ordered_nodes.append(node)
            neighbors = sorted(G.neighbors(node), key=lambda n: (n[1], n[0]))
            queue.extend(n for n in neighbors if n not in visited)
    return ordered_nodes


def order_nodes_by_path(G):
    """
    Order nodes by following the path of the curve.
    """
    start_node = min(G.nodes(), key=lambda n: (n[1], -n[0]))
    ordered_nodes = [start_node]
    visited = set([start_node])
    current_node = start_node
    while len(visited) < len(G.nodes()):
        neighbors = [n for n in G.neighbors(current_node) if n not in visited]
        if not neighbors:
            unvisited = set(G.nodes()) - visited
            current_node = min(unvisited, key=lambda n: np.sqrt((n[0]-current_node[0])**2 + (n[1]-current_node[1])**2))
        else:
            current_node = min(neighbors, key=lambda n: n[1])  
        ordered_nodes.append(current_node)
        visited.add(current_node)
    return ordered_nodes


def calculate_curvatures(ordered_nodes):
    """
    Calculate gradient and Menger curvatures for ordered nodes.
    """
    x = np.array([node[1] for node in ordered_nodes])
    y = np.array([node[0] for node in ordered_nodes])
    gradient_curvature = calculate_gradient_curvature(x, y)
    menger_curvature = calculate_menger_curvature(x, y)
    return gradient_curvature, menger_curvature


def plot_graph_and_curvature(G_smooth, path_ordered_nodes, bfs_ordered_nodes, gradient_curvature, menger_curvature, node_colors):
    node_to_bfs_index = {node: i+1 for i, node in enumerate(bfs_ordered_nodes)}
    node_to_path_index = {node: i+1 for i, node in enumerate(path_ordered_nodes)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    curvature_data = {
        'Path Index': [],
        'BFS Index': [],
        'X': [],
        'Y': [],
        'Gradient Curvature': [],
    }
    top_20_indices = np.argsort(gradient_curvature)[-50:]
    indices_to_plot = set()
    for idx in top_20_indices:
        indices_to_plot.update([idx - 1, idx, idx + 1])
    indices_to_plot = sorted(list(indices_to_plot))
    pos = {node: (node[1], node[0]) for node in G_smooth.nodes()}
    colors = [node_colors.get(node, 'black') for node in G_smooth.nodes()]
    nx.draw(G_smooth, pos, node_size=50, ax=ax1, node_color=colors, with_labels=False)
    for idx in indices_to_plot:
        if 0 <= idx < len(path_ordered_nodes):
            node = path_ordered_nodes[idx]
            x, y = pos[node]
            path_index = node_to_path_index[node]
            ax1.text(x, y + 0.1, str(path_index), fontsize=9, ha='center', va='center', color='red', fontweight='bold')
    ax1.set_title('Graph of Smoothed Points (Path Ordering for Selected Nodes)')
    ax1.invert_yaxis()
    avg_gradient = np.mean(gradient_curvature)
    avg_menger = np.mean(menger_curvature)
    width = 0.35
    valid_indices = [idx for idx in indices_to_plot if 0 <= idx < len(path_ordered_nodes)]
    for plot_idx, idx in enumerate(valid_indices, start=1):
        node = path_ordered_nodes[idx]
        grad_curv = gradient_curvature[idx]
        color = node_colors.get(node, 'black')
        path_index = node_to_path_index[node]
        curvature_data['Path Index'].append(path_index)
        curvature_data['BFS Index'].append(node_to_bfs_index[node])
        curvature_data['X'].append(node[1])
        curvature_data['Y'].append(node[0])
        curvature_data['Gradient Curvature'].append(grad_curv)
        ax2.bar(plot_idx - width/2, grad_curv, width=width, color=color, label='Gradient Curvature' if plot_idx == 1 else "")
        ax2.text(plot_idx - width/2, grad_curv + 0.01, f'{grad_curv:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    ax2.axhline(avg_gradient, color='blue', linestyle='--', label=f'Avg Gradient Curvature: {avg_gradient:.6f}')
    ax2.set_title('Curvature Plot (Top 20 Gradient Curvature Nodes and Adjacent)')
    ax2.set_xlabel('Node Index (Path Order)')
    ax2.set_ylabel('Curvature')
    ax2.legend()
    ax2.set_xticks(range(1, len(valid_indices)+1))
    ax2.set_xticklabels([str(node_to_path_index[path_ordered_nodes[idx]]) for idx in valid_indices], rotation=45)
    plt.tight_layout()
    plt.show()
    curvature_df = pd.DataFrame(curvature_data)
    curvature_df.to_csv('curvature_data.csv', index=False)


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
                #print(f"node 1 {node1}, node 2 {node2}, node 3 {node3}")
                # Check if diagonal edges exist in the triangle
                if graph.has_edge(node1, node2) and is_diagonal_edge(node1, node2):
                    #print("node 1 and node 2 diagonal edge, removing edge")
                    graph.remove_edge(node1, node2)
                elif graph.has_edge(node2, node3) and is_diagonal_edge(node2, node3):
                    #print("node 2 and node 3 diagonal edge, removing edge")
                    graph.remove_edge(node2, node3)
                elif graph.has_edge(node1, node3) and is_diagonal_edge(node1, node3):
                    #print("node 1 and node 3 diagonal edge, removing edge")
                    graph.remove_edge(node1, node3)


def visualize_pca_steps(ordered_nodes, save_path=None):
    # Create a multi-panel figure
    plt.figure(figsize=(15, 10))
    plt.suptitle('PCA Analysis of Skeleton Structure', fontsize=16)
    # Step 1: Original Nodes
    plt.subplot(2, 3, 1)
    x = np.array([node[1] for node in ordered_nodes])
    y = np.array([node[0] for node in ordered_nodes])
    plt.scatter(x, y, alpha=0.5, color='blue')
    plt.title('Step 1: Original Nodes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis() 
    # Step 2: Spline Smoothing
    plt.subplot(2, 3, 2)
    tck, u = splprep([x, y], s=0.0, k=3)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_smooth, y_smooth = splev(u_new, tck)
    plt.scatter(x_smooth, y_smooth, alpha=0.5, color='green')
    plt.title('Step 2: Smoothed Curve')
    plt.xlabel('Smoothed X')
    plt.ylabel('Smoothed Y')
    plt.gca().invert_yaxis() 
    # Step 3: Centered Data
    plt.subplot(2, 3, 3)
    points = np.column_stack((x_smooth, y_smooth))
    points_centered = points - np.mean(points, axis=0)
    plt.scatter(points_centered[:, 0], points_centered[:, 1], alpha=0.5, color='red')
    plt.title('Step 3: Centered Data')
    plt.xlabel('Centered X')
    plt.ylabel('Centered Y')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.gca().invert_yaxis() 
    # Step 4: PCA Transformation
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(points_centered)
    pca1 = PCA(n_components=1)
    pca_points1 = pca1.fit_transform(points_centered)
    plt.subplot(2, 3, 4)
    plt.scatter(pca_points1, points_centered[:, 1], alpha=0.5, color='orange')
    plt.title('Step 4: PC1 Projection')
    plt.xlabel('First Principal Component')
    plt.ylabel('Projected Y')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.gca().invert_yaxis() 
    # Step 5: Variance Explanation
    plt.subplot(2, 3, 5)
    explained_variance = pca.explained_variance_ratio_
    plt.bar(['PC1', 'PC2'], explained_variance)
    plt.title('Step 5: Variance Explained')
    plt.ylabel('Proportion of Variance')
    plt.gca().invert_yaxis() 
    # Step 6: Principal Component Visualization
    plt.subplot(2, 3, 6)
    plt.scatter(points_centered[:, 0], points_centered[:, 1], alpha=0.5, color='gray')
    # Plot principal components
    for i, (component, variance) in enumerate(zip(pca.components_, pca.explained_variance_), 1):
        # Scale component by its variance
        scaled_component = component * np.sqrt(variance)
        plt.quiver(0, 0, scaled_component[0], scaled_component[1], 
                   angles='xy', scale_units='xy', scale=1, 
                   color=f'C{i}', label=f'PC{i}')
    plt.title('Step 6: Principal Components')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis() 
    plt.legend()
    principal_component = pca.components_[0]
    slope = principal_component[1] / principal_component[0] if principal_component[0] != 0 else np.inf
    plt.figtext(0.5, 0.01, 
                f"Global Slope: {slope:.4f}\n" +
                f"Variance Explained by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%\n" +
                f"Variance Explained by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%", 
                ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.gca().invert_yaxis() 
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    # Return additional details for potential further analysis
    # return {
    #     'pca': pca,
    #     'slope': slope,
    #     'points': points,
    #     'points_centered': points_centered,
    #     'pca_points': pca_points
    # }


def calculate_component_curvatures(G_smooth, components):
    component_curvatures = {}
    for component_id, nodes in components.items():
        ordered_nodes = order_nodes_by_path(G_smooth.subgraph(nodes))
        if len(ordered_nodes) <= 1:
            continue
        x = np.array([node[1] for node in ordered_nodes])
        y = np.array([node[0] for node in ordered_nodes])
        tck, u = splprep([x, y], s=0.0, k=3)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_smooth, y_smooth = splev(u_new, tck)
        dx_dt = np.gradient(x_smooth)
        dy_dt = np.gradient(y_smooth)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
        # Apply Savitzky-Golay filter to smooth the curvature
        curvature_smooth = savgol_filter(curvature, window_length=51, polyorder=3)
        # Calculate representative curvature (excluding extremes)
        curvature_sorted = np.sort(curvature_smooth)
        representative_curvature = np.average(curvature_sorted)  # Exclude top and bottom 10 values
        # Calculate component length
        distances = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
        component_length = np.sum(distances)
        # Use PCA for slope calculation
        points = np.column_stack((x_smooth, y_smooth))
        pca = PCA(n_components=2)
        pca.fit(points)
        principal_component = pca.components_[0] 
        slope = principal_component[1] / principal_component[0] if principal_component[0] != 0 else np.inf  # Slope from the orientation
        # Store results
        component_curvatures[component_id] = {
            'nodes': ordered_nodes,
            'x_smooth': x_smooth,
            'y_smooth': y_smooth,
            'curvature_smooth': curvature_smooth,
            'representative_curvature': representative_curvature,
            'component_length': component_length,
            'average_slope': slope
        }
    return component_curvatures


def plot_component_and_curvature(G_smooth, node_colors, components, component_curvatures):
    for component_id, data in component_curvatures.items():
        nodes = data['nodes']
        x_smooth = data['x_smooth']
        y_smooth = data['y_smooth']
        curvature_smooth = data['curvature_smooth']
        representative_curvature = data['representative_curvature']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        pos = {node: (node[1], node[0]) for node in G_smooth.nodes()}
        nx.draw_networkx_nodes(G_smooth, pos, node_color='lightgrey', node_size=20, ax=ax1)
        nx.draw_networkx_edges(G_smooth, pos, edge_color='lightgrey', width=1, ax=ax1)
        component_nodes = components[component_id]
        component_color = node_colors[component_id]  # Assume all nodes in component have same color
        nx.draw_networkx_nodes(G_smooth, pos, nodelist=component_nodes, node_color=component_color, node_size=50, ax=ax1)
        component_edges = G_smooth.edges(component_nodes)
        nx.draw_networkx_edges(G_smooth, pos, edgelist=component_edges, edge_color=component_color, width=2, ax=ax1)
        ax1.plot(x_smooth, y_smooth, color=component_color, linewidth=2, label='Smoothed Curve')
        ax1.set_title(f'Component {component_id} in Full Graph')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.invert_yaxis()
        ax1.legend()
        ax2.plot(range(len(curvature_smooth)), curvature_smooth, label='Smoothed Curvature', color = component_color)
        ax2.set_title(f'Smoothed Curvature for Component {component_id}')
        ax2.set_xlabel('Point Index')
        ax2.set_ylabel('Curvature')
        ax2.legend()
        stats_text = f'Representative Curvature: {representative_curvature:.6f}'
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.show()


def perturb_duplicates(points, epsilon=1e-6):
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(points)), unique_indices)
    for idx in duplicate_indices:
        points[idx] += np.random.uniform(-epsilon, epsilon, size=2)
    return points


def smooth_entire_graph_and_calculate_curvature(G_smooth, component_curvatures, distance_threshold=1e-2):
    ordered_nodes = order_nodes_by_path(G_smooth)
    x_entire = np.array([node[1] for node in ordered_nodes])
    y_entire = np.array([node[0] for node in ordered_nodes])
    entire_points = np.column_stack((x_entire, y_entire))
    entire_points = perturb_duplicates(entire_points)
    perturbed_nodes = [(entire_points[i, 1], entire_points[i, 0]) for i in range(len(entire_points))]
    component_ids = []
    for node in perturbed_nodes:
        for component_id, comp_nodes in components.items():
            if node in comp_nodes:
                component_ids.append(component_id)
                break
    graph_points = np.column_stack((x_entire, y_entire))
    kd_tree = KDTree(graph_points)
    # Smooth the entire curve again using spline interpolation
    tck, u = splprep([entire_points[:, 0], entire_points[:, 1]], s=0.0, k=3)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_smooth_entire, y_smooth_entire = splev(u_new, tck)
    u_original = np.linspace(0, 1, len(component_ids))  # Parametrize component_ids
    smoothed_component_ids = np.interp(u_new, u_original, component_ids)
    # Check distances from smoothed points to the nearest graph points
    smoothed_points = np.column_stack((x_smooth_entire, y_smooth_entire))
    distances, _ = kd_tree.query(smoothed_points)
    # Filter out points that are farther than the threshold distance
    valid_indices = distances <= distance_threshold
    x_smooth_filtered = x_smooth_entire[valid_indices]
    y_smooth_filtered = y_smooth_entire[valid_indices]
    smoothed_component_ids_filtered = smoothed_component_ids[valid_indices]
    # Calculate derivatives for the entire smooth curve
    dx_dt = np.gradient(x_smooth_filtered)
    dy_dt = np.gradient(y_smooth_filtered)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    # Calculate curvature for the filtered smooth graph
    curvature_filtered = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
    # Apply Savitzky-Golay filter to smooth the curvature
    curvature_smooth_filtered = savgol_filter(curvature_filtered, window_length=51, polyorder=3)
    # Calculate representative curvature (excluding extremes)
    curvature_sorted_filtered = np.sort(curvature_smooth_filtered)
    representative_curvature_filtered = np.mean(curvature_sorted_filtered[10:-10])  # Exclude top and bottom 10 values
    return x_smooth_filtered, y_smooth_filtered, curvature_smooth_filtered, representative_curvature_filtered, smoothed_component_ids_filtered


def plot_smoothed_graphs(x_smooth_entire, y_smooth_entire, curvature_smooth_entire):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(x_smooth_entire, y_smooth_entire, label="Smoothed Entire Graph", color='blue')
    axs[0].set_title("Smoothed Entire Graph")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].invert_yaxis()
    axs[0].legend()
    axs[1].plot(range(len(curvature_smooth_entire)), curvature_smooth_entire, label="Smoothed Curvature of Entire Graph", color='green')
    axs[1].set_title("Smoothed Curvature of Entire Graph")
    axs[1].set_xlabel("Point Index")
    axs[1].set_ylabel("Curvature")
    axs[1].legend()
    plt.tight_layout() 
    plt.show()


def remove_side_branches(G):
    G_pruned = G.copy()
    leaf_nodes = [node for node in G_pruned.nodes if G_pruned.degree(node) == 1]
    branch_points = [node for node in G_pruned.nodes if G_pruned.degree(node) > 2]
    if not branch_points:
        return G_pruned
    
    def dfs_to_branch(start_node):
        stack = [(start_node, [start_node])]
        while stack:
            node, path = stack.pop()
            if node in branch_points:
                return path[:-1]  
            for neighbor in G_pruned.neighbors(node):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return []  
    for leaf in leaf_nodes:
        side_branch = dfs_to_branch(leaf)
        if side_branch:
            G_pruned.remove_nodes_from(side_branch)
    return G_pruned


def plot_pruned_graph(G_pruned, G_smooth, frame_number, save_dir):
    pos_pruned = {(row, col): (col, row) for row, col in G_pruned.nodes()}  # Swap coordinates for plotting
    pos = {node: (node[1], node[0]) for node in G_smooth.nodes()}
    plt.figure(figsize=(8, 8))
    nx.draw(G_smooth, pos, node_size=10, node_color='gray', with_labels=False)
    nx.draw(G_pruned, pos_pruned, node_size=30, node_color='red', with_labels=False)
    plt.title('Pruned Graph overlayed on Original Graph')
    plt.gca().invert_yaxis()
    # save_path = os.path.join(save_dir, f'pruned_{frame_number}.png')
    # if save_path:
    #     plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # plt.close()  # Close the figure to free up memory
    plt.show()


def plot_overlay_background(ax, frame):
    ax.imshow(frame, cmap='gray', alpha=1.0)  
    #ax.invert_yaxis()


def plot_smoothed_graphs2(x_smooth_entire, y_smooth_entire, curvature_smooth_entire, smoothed_component_ids, node_colors, G_smooth, current_frame_number, overlay_frame, save_dir):
    fig = plt.figure(figsize=(20, 6))  
    gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.7)  
    ax1 = fig.add_subplot(gs[0, 0])
    pos = {node: (node[1], node[0]) for node in G_smooth.nodes()}
    plot_overlay_background(ax1, overlay_frame)
    distances = []
    for i in range(len(x_smooth_entire) - 1):
        dist = np.sqrt((x_smooth_entire[i+1] - x_smooth_entire[i])**2 + (y_smooth_entire[i+1] - y_smooth_entire[i])**2)
        distances.append(dist)
    max_jump_index = np.argmax(distances)
    print(f"Maximum Euclidean distance between points at index {max_jump_index} and {max_jump_index+1}")
    first_half_x = x_smooth_entire[:max_jump_index+1]
    first_half_y = y_smooth_entire[:max_jump_index+1]
    second_half_x = x_smooth_entire[max_jump_index+1:]
    second_half_y = y_smooth_entire[max_jump_index+1:]
    for i in range(len(first_half_x) - 1):
        color = node_colors[int(smoothed_component_ids[i])]
        ax1.plot(first_half_x[i:i+2], first_half_y[i:i+2], color=color, linewidth=4)
    for i in range(len(second_half_x) - 1):
        color = node_colors[int(smoothed_component_ids[max_jump_index + 1 + i])]
        ax1.plot(second_half_x[i:i+2], second_half_y[i:i+2], color=color, linewidth=4)
    ax1.set_title(f"Smoothed Entire Graph with Colored Components - Frame: {current_frame_number}", fontweight='bold')
    ax1.set_xlabel('X', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=10, labelcolor='black', width=2)
    for tick in ax1.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax1.get_yticklabels():
        tick.set_fontweight('bold')
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(len(curvature_smooth_entire) - 1):
        color = node_colors[int(smoothed_component_ids[i])]  # Use the same color for curvature
        ax2.plot([i, i+1], curvature_smooth_entire[i:i+2], color=color, linewidth=4)
    ax2.set_title(f"Smoothed Curvature of Entire Graph with Colored Components - Frame: {current_frame_number}", fontweight='bold')
    ax2.set_xlabel('X', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=10, labelcolor='black', width=2)
    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
    plt.tight_layout(rect=[0, 0, 1, 1]) 
    save_path = os.path.join(save_dir, f'plot{current_frame_number}.png')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #plt.show()

def create_graph_from_skeleton(frame):
    skeleton = np.array(frame) < 128
    rows, cols = skeleton.shape
    G = nx.Graph()
    # Add nodes and edges for each skeleton pixel
    for row in range(rows):
        for col in range(cols):
            if skeleton[row, col]:  # Check if it's a black pixel (part of the skeleton)
                G.add_node((row, col))
                # Add edges for the neighboring pixels
                for drow, dcol in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    new_row = row + drow
                    new_col = col + dcol
                    if 0 <= new_row < rows and 0 <= new_col < cols and skeleton[new_row, new_col]:
                        G.add_edge((row, col), (new_row, new_col))
    return G


def save_curvature_to_csv(curvature_rms_list, curvature_truncated_mean_list, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, "output.csv")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Curvature RMS", "Curvature Truncated Mean"])
        for rms, truncated_mean in zip(curvature_rms_list, curvature_truncated_mean_list):
            writer.writerow([rms, truncated_mean])
    print(f"Data saved successfully to {output_path}")


def find_longest_line_between_leaves(G):
    leaf_nodes = [node for node, degree in G.degree() if degree == 1]
    max_distance = 0
    longest_line = (None, None)
    longest_line_nodes = []
    for i, node1 in enumerate(leaf_nodes):
        for node2 in leaf_nodes[i + 1:]:
            distance = np.linalg.norm(np.array(node1) - np.array(node2))
            if distance > max_distance:
                max_distance = distance
                longest_line = (node1, node2)
                longest_line_nodes = [node1, node2]
    return longest_line, longest_line_nodes


def remove_side_branches2(G, longest_line_nodes):
    G_pruned = G.copy()
    leaf_nodes = [node for node in G_pruned.nodes if G_pruned.degree(node) == 1]
    branch_points = [node for node in G_pruned.nodes if G_pruned.degree(node) > 2]
    if not branch_points:
        return G_pruned
    
    def dfs_to_branch(start_node):
        stack = [(start_node, [start_node])]
        while stack:
            node, path = stack.pop()
            if node in branch_points:
                return path[:-1]  
            for neighbor in G_pruned.neighbors(node):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
        return []  
    for leaf in leaf_nodes:
        if leaf in longest_line_nodes:
            print(f"This is the node in longest_line_nodes {leaf}")
            continue
        side_branch = dfs_to_branch(leaf)
        if side_branch:
            G_pruned.remove_nodes_from(side_branch)
    return G_pruned


def find_branch_nodes(G_smooth):
    branch_nodes = [node for node, degree in G.degree() if degree > 2]
    return branch_nodes


def filter_branch_nodes(G_pruned_2, branch_nodes):
    filtered_branch_nodes = []
    graph_nodes = G_pruned_2.nodes
    for node in branch_nodes:
        if node in graph_nodes:
            filtered_branch_nodes.append(node)
    return filtered_branch_nodes


def plot_pruned_graph2(G_pruned, G_smooth, frame_number, save_dir, longest_line, filtered_branch_nodes):
    pos_pruned = {(row, col): (col, row) for row, col in G_pruned.nodes()}
    pos_smooth = {node: (node[1], node[0]) for node in G_smooth.nodes()}
    pos_filtered_branch_nodes = {node: (node[1], node[0]) for node in filtered_branch_nodes}
    plt.figure(figsize=(8, 8))
    nx.draw(G_smooth, pos=pos_smooth, node_size=5, node_color='gray', with_labels=False, label='Original Fiber Skeleton')
    nx.draw(G_pruned, pos=pos_pruned, node_size=15, node_color='red', with_labels=False, label='Fiber Skeleton with CenterLine')
    nx.draw_networkx_nodes(G_pruned, pos=pos_filtered_branch_nodes, nodelist=filtered_branch_nodes, node_color='blue', node_size=100, label='Filtered Branch Nodes')
    if longest_line[0] and longest_line[1]:
        x_values = [longest_line[0][1], longest_line[1][1]]  # col values
        y_values = [longest_line[0][0], longest_line[1][0]]  # row values
        plt.plot(x_values, y_values, color='green', linewidth=2, label='Longest Leaf-to-Leaf Line')
    plt.title(f'Pruned Graph with Longest Leaf-to-Leaf Line (Frame {frame_number})')
    plt.gca().invert_yaxis() 
    # Save or show the plot
    save_path = os.path.join(save_dir, f'pruned{frame_number}.png')
    # Uncomment the following line to save the image
    # plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #plt.legend()
    plt.show()
    plt.close()


SLOPE_TOLERANCE = 0.1


def is_diagonal_edge_by_slope(node1, node2, tolerance=SLOPE_TOLERANCE):
    x1, y1 = node1
    x2, y2 = node2
    if x1 == x2:  
        return False
    slope = (y2 - y1) / (x2 - x1)
    return abs(abs(slope) - 1) <= tolerance


def detect_cycles2(graph):
    cycles = list(nx.cycle_basis(graph))
    if not cycles:
        print("No cycles found.")
    else:
        for cycle in cycles:
            cycle_nodes = ", ".join([str(node) for node in cycle])
        for cycle in cycles:
            for i in range(len(cycle)):
                node1 = cycle[i]
                node2 = cycle[(i + 1) % len(cycle)]  
                if graph.has_edge(node1, node2) and is_diagonal_edge_by_slope(node1, node2):
                    graph.remove_edge(node1, node2)


def setup_csv(output_path="frame_metrics.csv"):
    headers = ['Frame', 'Component ID', 'Total Length', 'Average Slope', 'Average Curvature']
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)


def write_frame_to_csv(frame_number, component_curvatures, output_path="frame_metrics_2.csv"):
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for component_id, data in component_curvatures.items():
            row = [
                frame_number,
                component_id+1,
                data['component_length'],
                data['average_slope'],
                data['representative_curvature']
            ]
            writer.writerow(row)


def visualize_frame(frame, overlay_frame, current_frame_number):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 2)
    plt.title(f'Frame {current_frame_number}')
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.title(f'Overlay Frame {current_frame_number}')
    plt.imshow(overlay_frame, cmap='gray')
    plt.axis('off')
    # Optional: Adjust layout and save the figure
    plt.tight_layout()
    # plt.savefig(f'{save_directory}/frame_{current_frame_number}_comparison.png')
    # plt.close()
    plt.show()


def find_longest_leaf_path(G, previous_leaf_nodes):
    leaf_nodes = [node for node, degree in G.degree() if degree == 1]
    if len(leaf_nodes) < 2:
        return None
    leaf_pairs = list(itertools.combinations(leaf_nodes, 2))
    max_length = 0
    longest_line = (None, None)
    longest_line_nodes = []
    G_shortest = nx.Graph()
    for leaf1, leaf2 in leaf_pairs:
        try:
            shortest_path = nx.shortest_path(G, leaf1, leaf2)
        except nx.NetworkXNoPath:
            continue
        path_length = sum(
            math.sqrt((path_node[0] - shortest_path[i+1][0])**2 + 
                      (path_node[1] - shortest_path[i+1][1])**2) 
            for i, path_node in enumerate(shortest_path[:-1])
        )
        if path_length > max_length:
            max_length = path_length
            max_pair = [leaf1, leaf2]
            longest_line = (leaf1, leaf2)
            longest_line_nodes = [leaf1, leaf2]
            G_shortest = G.subgraph(shortest_path)
    return G_shortest, longest_line, longest_line_nodes


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python calculate_component_metrics.py <skeleton.tif> <confocal.tif>")
        sys.exit(1)

    tiff_path = sys.argv[1]
    overlay_tiff = sys.argv[2]
    img = Image.open(tiff_path)
    frames = []
    start_frame = 1
    end_frame = 2
    img_overlay = Image.open(overlay_tiff)
    save_directory = 'Plots'
    curvature_rms_list = []
    curvature_truncated_mean_list = []
    setup_csv()
    previous_leaf_nodes = [] 
    for i in range(start_frame-1, end_frame):
        print(f'Processing frame {i+1}/{end_frame}')
        current_frame_number = i+1
        img.seek(i)
        frame = img.convert('L')
        
        img_overlay.seek(i)
        overlay_frame = img_overlay.copy()
        

        visualize_frame(frame, overlay_frame, current_frame_number)

        G = create_graph_from_skeleton(frame)
        
        #plot_pruned_graph(G, G, frame_number=i + 1, save_dir=save_directory)
        
        detect_cycles(G)
        G_largest = get_largest_connected_component(G)

        plot_pruned_graph(G_largest, G, frame_number=i + 1, save_dir=save_directory)

        G_smooth = add_interpolated_nodes(G_largest, max_edge_length=0.05)

        branch_nodes = find_branch_nodes(G_smooth)
        

        G_pruned = remove_side_branches(G_smooth)

        
        #plot_pruned_graph(G_pruned, G_smooth, frame_number=i + 1, save_dir=save_directory)

        detect_cycles2(G_pruned)

        #plot_pruned_graph(G_pruned, G_smooth, frame_number=i + 1, save_dir=pruned_save_directory)

        branch_nodes = find_branch_nodes(G_pruned)

        G_smooth_original = G_smooth

        
        G_shortest, longest_line, longest_line_nodes = find_longest_leaf_path(G_pruned, previous_leaf_nodes)

        previous_leaf_nodes = longest_line_nodes

        G_pruned_2 = G_shortest

        detect_cycles2(G_pruned_2)

        filtered_branch_nodes = filter_branch_nodes(G_pruned_2, branch_nodes)

        plot_pruned_graph2(G_pruned_2, G_smooth, i + 1, save_directory, longest_line, filtered_branch_nodes)

        G_smooth = G_pruned_2

        bfs_ordered_nodes = order_nodes(G_smooth)

        path_ordered_nodes = order_nodes_by_path(G_smooth)

        node_colors, components = color_graph_sections(G_smooth, G_smooth_original)

        component_curvatures = calculate_component_curvatures(G_smooth, components)

        #plot_pruned_graph2(G_pruned_2, G_smooth, i + 1, pruned_save_directory, longest_line, filtered_branch_nodes)

        write_frame_to_csv(i+1, component_curvatures)
