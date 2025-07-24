from typing import Dict, List, Tuple

import networkx as nx
import pymetis
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def make_partitions(
    dataset: Data, num_parts: int, device: torch.device = None
) -> Tuple[List[Data], Dict[int, List[int]]]:
    """
    Args:
        dataset: The input graph.
        num_parts: The desired number of partitions.
        device: The torch device.

    Returns:
        - A list of PyG Data objects, each representing a partition.
        - A dictionary mapping partition ID to a list of original node IDs.
    """
    part_dict = {i: [] for i in range(num_parts)}
    nx_graph = to_networkx(dataset, to_undirected=True)

    adjacency_list = [
        list(nx_graph.neighbors(i)) for i in range(nx_graph.number_of_nodes())
    ]
    if not adjacency_list or nx_graph.number_of_edges() == 0:
        membership = [i % num_parts for i in range(nx_graph.number_of_nodes())]
    else:
        _, membership = pymetis.part_graph(num_parts, adjacency=adjacency_list)

    for orig_node_id, part_id in enumerate(membership):
        part_dict[part_id].append(orig_node_id)

    part_graphs = []
    for i in range(num_parts):
        nodes_in_part = torch.tensor(part_dict[i], dtype=torch.long, device=device)
        if len(nodes_in_part) > 0:
            part_graphs.append(dataset.subgraph(nodes_in_part))

    return part_graphs, part_dict


def build_recursive_hierarchy(
    data: Data, levels: Tuple[int, int], device: torch.device
) -> Tuple[List, Dict, List, Dict, Dict]:
    """
    Args:
        data: The full input graph.
        levels: A tuple (num_coarse_parts, num_fine_per_coarse).
        device: The torch device.

    Returns:
        - coarse_graphs: List of graphs for the coarse partitions.
        - coarse_part_nodes_map: Maps coarse partition index to global node IDs.
        - fine_graphs: List of graphs for all fine partitions.
        - fine_to_coarse_map: Maps a fine partition index to its parent coarse index.
        - fine_part_nodes_map: Maps fine partition index to global node IDs.
    """
    num_coarse_parts, num_fine_per_coarse = levels
    coarse_graphs, coarse_part_nodes_map = make_partitions(
        data, num_coarse_parts, device
    )

    fine_graphs, fine_to_coarse_map, fine_part_nodes_map = [], {}, {}
    fine_global_idx = 0

    for coarse_idx, coarse_graph in enumerate(coarse_graphs):
        original_coarse_nodes = coarse_part_nodes_map.get(coarse_idx, [])
        if not original_coarse_nodes:
            continue

        finer_partitions, finer_part_nodes_map_local = make_partitions(
            coarse_graph, num_fine_per_coarse, device
        )

        for fine_local_idx, fine_part in enumerate(finer_partitions):
            local_nodes_in_coarse = finer_part_nodes_map_local.get(fine_local_idx, [])
            global_nodes = [original_coarse_nodes[i] for i in local_nodes_in_coarse]

            fine_graphs.append(fine_part)
            fine_to_coarse_map[fine_global_idx] = coarse_idx
            fine_part_nodes_map[fine_global_idx] = global_nodes
            fine_global_idx += 1

    return (
        coarse_graphs,
        coarse_part_nodes_map,
        fine_graphs,
        fine_to_coarse_map,
        fine_part_nodes_map,
    )


def build_coarse_partition_graph(
    G_nx: nx.Graph, coarse_part_nodes_map: Dict
) -> nx.Graph:
    """
    Creates a "supergraph" where each node represents a coarse partition and an
    edge exists if the original partitions are connected.

    Args:
        G_nx: The full graph in NetworkX format.
        coarse_part_nodes_map: A map from coarse partition ID to its global node IDs.

    Returns:
        A NetworkX graph of the coarse partitions.
    """
    num_coarse_parts = len(coarse_part_nodes_map)
    node_to_coarse_part = {
        node: part_id
        for part_id, nodes in coarse_part_nodes_map.items()
        for node in nodes
    }
    C_nx = nx.Graph()
    C_nx.add_nodes_from(range(num_coarse_parts))

    for u, v in G_nx.edges():
        p1, p2 = node_to_coarse_part.get(u), node_to_coarse_part.get(v)
        if p1 is not None and p2 is not None and p1 != p2:
            C_nx.add_edge(p1, p2)
    return C_nx
