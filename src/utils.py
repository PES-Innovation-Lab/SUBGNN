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
