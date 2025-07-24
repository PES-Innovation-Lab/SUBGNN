import hashlib
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.data import Data


def feature_to_label(vector: "np.ndarray") -> str:
    """
    Args:
        vector: A numpy array representing the node's feature vector.

    Returns:
        A SHA256 hash string representing the feature vector.
    """
    indices = np.where(vector == 1)[0]
    index_str = "_".join(map(str, indices))
    hash_str = hashlib.sha256(index_str.encode("utf-8")).hexdigest()
    return hash_str


def convert_graph_to_csv(
    data: Data,
    filename: str,
    global_id_to_label_feature: Dict[int, str],
    local_to_global_map: Dict[int, int],
):
    """
    Args:
        data: The PyTorch Geometric graph data object.
        filename: The path to the output CSV file.
        global_id_to_label_feature: Maps a global node ID to its feature hash.
        local_to_global_map: Maps a node's local index in `data` to its original global ID.
    """
    edge_data = data.edge_index.cpu().numpy().T
    edges_df = pd.DataFrame(edge_data)

    node_label_list = []
    for i in range(data.num_nodes):
        global_id = local_to_global_map[i]
        node_label_list.append([i, "", global_id_to_label_feature[global_id]])
    node_labels_df = pd.DataFrame(node_label_list)

    edges_df.to_csv(filename, header=False, index=False, mode="w")
    node_labels_df.to_csv(filename, header=False, index=False, mode="a")


def are_partitions_neighbors(
    G_nx: nx.Graph, nodes_a: List[int], nodes_b: List[int]
) -> bool:
    """
    Args:
        G_nx: The complete graph as a NetworkX object.
        nodes_a: A list of global node IDs in the first partition.
        nodes_b: A list of global node IDs in the second partition.

    Returns:
        True if an edge exists between the partitions, False otherwise.
    """
    node_set_b = set(nodes_b)
    for u in nodes_a:
        for v in G_nx.neighbors(u):
            if v in node_set_b:
                return True
    return False

def plot_unified_results(Gq, G_truth_vis, q_global_nodes, G_predicted, true_fine_indices, predicted_fine_idx, experiment_name, **kwargs):
    """
    Plots a unified visualization of the hierarchical graph search results.

    This function creates a 3-panel plot:
    1. The query graph.
    2. The ground truth graph(s), with the query nodes highlighted and drawn on top.
    3. The predicted graph, with the query nodes highlighted and drawn on top.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from torch_geometric.utils import to_networkx

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    is_correct = predicted_fine_idx in true_fine_indices
    fig.suptitle(f"Hierarchical Search Results: {experiment_name}", fontsize=16)

    # ===================================================================
    # Plot 1: Query Graph
    # ===================================================================
    Gq_nx = to_networkx(Gq, to_undirected=True)
    pos_q = nx.spring_layout(Gq_nx, seed=42)
    axes[0].set_title(f"Query Graph (Gq)\n{Gq.num_nodes} nodes", fontsize=12)
    nx.draw(Gq_nx, pos_q, ax=axes[0],
            node_color='#ff0000', edgecolors='black',
            node_size=90, with_labels=False, linewidths=1.8)

    # ===================================================================
    # Plot 2: Ground Truth
    # ===================================================================
    G_truth_nx = to_networkx(G_truth_vis, to_undirected=True)
    pos_truth = nx.spring_layout(G_truth_nx, seed=42)
    axes[1].set_title(f"Ground Truth Partition(s)\nFine Part(s): {sorted(true_fine_indices)}", fontsize=12)
    fine_part_nodes_map = kwargs['fine_part_nodes_map']

    if len(true_fine_indices) == 1:
        true_global_nodes = fine_part_nodes_map[true_fine_indices[0]]
        global_to_local_map = {node: i for i, node in enumerate(true_global_nodes)}
    else:
        all_global_nodes_in_stitched = sorted(list(set(node for idx in true_fine_indices for node in fine_part_nodes_map[idx])))
        global_to_local_map = {global_node: i for i, global_node in enumerate(all_global_nodes_in_stitched)}

    q_nodes_set = set(q_global_nodes)
    q_nodes_local = [global_to_local_map[g_node] for g_node in q_global_nodes if g_node in global_to_local_map]
    partition_nodes_local = [node for node in G_truth_nx.nodes() if node not in q_nodes_local]

    # --- Draw in layers: Edges -> Partition Nodes -> Query Nodes --- 
    nx.draw_networkx_edges(G_truth_nx, pos_truth, ax=axes[1], alpha=0.7)

    if len(true_fine_indices) == 1:
        nx.draw_networkx_nodes(G_truth_nx, pos_truth, ax=axes[1],
                                nodelist=partition_nodes_local,
                                node_color='#66c2a5',
                                node_size=90)
    else:
        node_to_part = {node: idx for idx in true_fine_indices for node in fine_part_nodes_map[idx]}
        cmap = plt.get_cmap('Set2')
        part_color_map = {idx: cmap(i % cmap.N) for i, idx in enumerate(sorted(true_fine_indices))}
        local_to_global_map = {v: k for k, v in global_to_local_map.items()}

        partition_node_colors = []
        for local_idx in partition_nodes_local:
            global_node = local_to_global_map[local_idx]
            part_idx = node_to_part.get(global_node)
            partition_node_colors.append(part_color_map.get(part_idx, '#cccccc'))

        nx.draw_networkx_nodes(G_truth_nx, pos_truth, ax=axes[1],
                                nodelist=partition_nodes_local,
                                node_color=partition_node_colors,
                                node_size=90)

        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              label='Query Node', markerfacecolor='#ff0000',
                              markeredgecolor='black', markersize=10)]
        for idx in sorted(true_fine_indices):
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      label=f'Part {idx}', markerfacecolor=part_color_map[idx],
                                      markeredgecolor='none', markersize=10))
        axes[1].legend(handles=handles, loc='best', fontsize=9)

    nx.draw_networkx_nodes(G_truth_nx, pos_truth, ax=axes[1],
                            nodelist=q_nodes_local,
                            node_color='#ff0000', edgecolors='black',
                            node_size=90, linewidths=1.8)

    query_edges = [(u, v) for u, v in G_truth_nx.edges() if u in q_nodes_local and v in q_nodes_local]
    nx.draw_networkx_edges(G_truth_nx, pos_truth, ax=axes[1],
                            edgelist=query_edges,
                            edge_color='black', width=2.5)

    # ===================================================================
    # Plot 3: Predicted Partition
    # ===================================================================
    G_pred_nx = to_networkx(G_predicted, to_undirected=True)
    pos_pred = nx.spring_layout(G_pred_nx, seed=42)
    pred_title = f"Predicted Partition (#{predicted_fine_idx})\n"
    pred_title += "(CORRECT)" if is_correct else "(INCORRECT)"
    axes[2].set_title(pred_title, fontsize=12)

    pred_global_nodes = fine_part_nodes_map[predicted_fine_idx]
    global_to_local_pred_map = {node: i for i, node in enumerate(pred_global_nodes)}

    q_nodes_in_pred_local = [global_to_local_pred_map[g_node] for g_node in q_global_nodes if g_node in global_to_local_pred_map]
    other_nodes_in_pred_local = [node for node in G_pred_nx.nodes() if node not in q_nodes_in_pred_local]
    base_color = '#228B22' if is_correct else '#FF6347'

    nx.draw_networkx_edges(G_pred_nx, pos_pred, ax=axes[2], alpha=0.7)

    nx.draw_networkx_nodes(G_pred_nx, pos_pred, ax=axes[2],
                            nodelist=other_nodes_in_pred_local,
                            node_color=base_color,
                            node_size=90)

    nx.draw_networkx_nodes(G_pred_nx, pos_pred, ax=axes[2],
                            nodelist=q_nodes_in_pred_local,
                            node_color='#ff0000', edgecolors='black',
                            node_size=90, linewidths=1.8)

    query_edges_pred = [(u, v) for u, v in G_pred_nx.edges() if u in q_nodes_in_pred_local and v in q_nodes_in_pred_local]
    nx.draw_networkx_edges(G_pred_nx, pos_pred, ax=axes[2],
                            edgelist=query_edges_pred,
                            edge_color='black', width=2.5)

    pred_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Query Node',
                   markerfacecolor='#ff0000', markeredgecolor='black', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Partition Node',
                   markerfacecolor=base_color, markersize=10)
    ]
    axes[2].legend(handles=pred_handles, loc='best', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
