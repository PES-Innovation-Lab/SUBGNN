import itertools
import random
from collections import defaultdict
from typing import List, Optional, Set, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.config import MULTI_COARSE_CONFIGS
from src.utils import are_partitions_neighbors


def _extract_fragment(graph: Data, target_nodes: int) -> Optional[List[int]]:
    """
    Extracts a connected component fragment of a target size from a graph
    using a Breadth-First Search (BFS) starting from a random node.

    Args:
        graph: The graph to extract from.
        target_nodes: The desired number of nodes in the fragment.

    Returns:
        A list of local node indices for the fragment, or None if extraction fails.
    """
    if graph.num_nodes < 5 or target_nodes < 5:
        return None

    graph_nx = to_networkx(graph, to_undirected=True)
    if not nx.is_connected(graph_nx):
        if not list(graph_nx.nodes()):
            return None
        largest_cc = max(nx.connected_components(graph_nx), key=len)
        graph_nx = graph_nx.subgraph(largest_cc)

    if graph_nx.number_of_nodes() == 0:
        return None

    start_node = random.choice(list(graph_nx.nodes()))
    queue, visited, fragment = [start_node], {start_node}, [start_node]

    while queue and len(fragment) < target_nodes:
        current = queue.pop(0)
        neighbors = list(graph_nx.neighbors(current))
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                fragment.append(neighbor)
                if len(fragment) >= target_nodes:
                    break
    return fragment


def _finalize_query_from_nodes(
    original_data: Data, global_nodes: List[int], min_nodes: int, device: torch.device
) -> Tuple[Optional[Data], Optional[List[int]]]:
    """
    Takes a list of global node IDs, creates a subgraph, finds the largest
    connected component, and returns it if it's large enough.

    Args:
        original_data: The full source graph.
        global_nodes: A list of global node IDs to form the query.
        min_nodes: The minimum required size for the final query graph.
        device: The torch device.

    Returns:
        A tuple of (PyG Data object for the query, list of global node IDs),
        or (None, None) if the resulting component is too small.
    """
    if not global_nodes:
        return None, None

    unique_global_nodes = sorted(list(set(global_nodes)))
    temp_graph = original_data.subgraph(
        torch.tensor(unique_global_nodes, device=device)
    )
    temp_nx = to_networkx(temp_graph, to_undirected=True)

    if temp_nx.number_of_nodes() == 0:
        return None, None

    # Ensure the query is a single connected component
    largest_cc_nodes = (
        max(nx.connected_components(temp_nx), key=len)
        if not nx.is_connected(temp_nx)
        else list(temp_nx.nodes())
    )

    if len(largest_cc_nodes) < min_nodes:
        return None, None

    final_global_nodes = [unique_global_nodes[i] for i in largest_cc_nodes]
    Gq = original_data.subgraph(torch.tensor(final_global_nodes, device=device))
    return Gq, final_global_nodes


def generate_single_partition_query(
    **kwargs,
) -> Tuple[Data, Data, List[int], List[int], Set[int]]:
    """
    Generates a query that is entirely contained within a single fine-level partition.
    """
    fine_graphs, fine_to_coarse_map, device = (
        kwargs["fine_graphs"],
        kwargs["fine_to_coarse_map"],
        kwargs["device"],
    )
    min_nodes, max_nodes = kwargs["min_nodes"], kwargs["max_nodes"]

    Gq = None
    while Gq is None:
        true_fine_idx = random.randint(0, len(fine_graphs) - 1)
        anchor_partition = fine_graphs[true_fine_idx]
        local_nodes = _extract_fragment(anchor_partition, max_nodes)

        if local_nodes and len(local_nodes) >= min_nodes:
            Gq = anchor_partition.subgraph(torch.tensor(local_nodes, device=device))
            q_global_nodes = [
                kwargs["fine_part_nodes_map"][true_fine_idx][i] for i in local_nodes
            ]

    true_coarse_idx = fine_to_coarse_map[true_fine_idx]
    return Gq, anchor_partition, q_global_nodes, [true_fine_idx], {true_coarse_idx}


def generate_multi_fine_partition_query(
    **kwargs,
) -> Tuple[Data, Data, List[int], List[int], Set[int]]:
    """
    Generates a query by stitching together fragments from multiple neighboring
    fine-partitions that all belong to the *same* coarse partition.
    """
    G_nx, fine_graphs, fine_part_nodes_map, fine_to_coarse_map, device = (
        kwargs["G_nx"],
        kwargs["fine_graphs"],
        kwargs["fine_part_nodes_map"],
        kwargs["fine_to_coarse_map"],
        kwargs["device"],
    )
    num_frags, min_nodes, max_nodes = (
        kwargs["num_frags"],
        kwargs["min_nodes"],
        kwargs["max_nodes"],
    )

    for _ in range(100):  # Try up to 100 times
        start_fine_idx = random.choice(list(fine_part_nodes_map.keys()))
        true_coarse_idx = fine_to_coarse_map[start_fine_idx]

        # Find all "sibling" fine partitions within the same coarse one
        siblings = [
            idx for idx, c_idx in fine_to_coarse_map.items() if c_idx == true_coarse_idx
        ]

        # Find a connected set of sibling partitions
        q_fine_indices, queue, visited = (
            [start_fine_idx],
            [start_fine_idx],
            {start_fine_idx},
        )
        while queue and len(q_fine_indices) < num_frags:
            current_idx = queue.pop(0)
            random.shuffle(siblings)
            for neighbor_idx in siblings:
                if neighbor_idx not in visited and are_partitions_neighbors(
                    G_nx,
                    fine_part_nodes_map[current_idx],
                    fine_part_nodes_map[neighbor_idx],
                ):
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)
                    q_fine_indices.append(neighbor_idx)
                    if len(q_fine_indices) >= num_frags:
                        break

        if len(q_fine_indices) < num_frags:
            continue

        # Extract fragments from each selected fine partition
        nodes_per_frag = max_nodes // num_frags
        all_query_nodes = []
        for fine_idx in q_fine_indices:
            local_nodes = _extract_fragment(fine_graphs[fine_idx], nodes_per_frag)
            if local_nodes:
                all_query_nodes.extend(
                    [fine_part_nodes_map[fine_idx][i] for i in local_nodes]
                )

        Gq, q_global_nodes = _finalize_query_from_nodes(
            kwargs["original_data"], all_query_nodes, min_nodes, device
        )
        if Gq:
            # The "ground truth" graph for visualization is the combination of all selected fine partitions
            stitched_nodes = [
                node for idx in q_fine_indices for node in fine_part_nodes_map[idx]
            ]
            G_stitched = kwargs["original_data"].subgraph(
                torch.tensor(stitched_nodes, device=device)
            )
            return Gq, G_stitched, q_global_nodes, q_fine_indices, {true_coarse_idx}

    raise RuntimeError("Failed to generate multi-fine-partition query.")


def generate_multi_coarse_partition_query(
    **kwargs,
) -> Tuple[Data, Data, List[int], List[int], Set[int]]:
    """
    Generates the most complex query type, stitching fragments from fine partitions
    that span across multiple, neighboring coarse partitions.
    """
    (
        original_data,
        G_nx,
        coarse_part_graph,
        fine_graphs,
        fine_part_nodes_map,
        fine_to_coarse_map,
        device,
    ) = (
        kwargs["original_data"],
        kwargs["G_nx"],
        kwargs["coarse_part_graph"],
        kwargs["fine_graphs"],
        kwargs["fine_part_nodes_map"],
        kwargs["fine_to_coarse_map"],
        kwargs["device"],
    )
    min_nodes, max_nodes = kwargs["min_nodes"], kwargs["max_nodes"]

    if coarse_part_graph.number_of_edges() == 0:
        raise RuntimeError(
            "Coarse graph has no edges, cannot generate multi-coarse query."
        )

    # Create a map from coarse index to its constituent fine indices
    coarse_to_fine_map = defaultdict(list)
    for f_idx, c_idx in fine_to_coarse_map.items():
        coarse_to_fine_map[c_idx].append(f_idx)

    # Try different configurations (num_fragments, min_coarse_partitions) to find a valid query
    for num_frags, min_coarse_parts in MULTI_COARSE_CONFIGS:
        possible_start_edges = list(coarse_part_graph.edges())
        random.shuffle(possible_start_edges)

        for c_idx1, c_idx2 in possible_start_edges:
            fine_parts_in_c1 = coarse_to_fine_map.get(c_idx1, [])
            fine_parts_in_c2 = coarse_to_fine_map.get(c_idx2, [])
            all_fine_pairs = list(itertools.product(fine_parts_in_c1, fine_parts_in_c2))
            random.shuffle(all_fine_pairs)

            for f1, f2 in all_fine_pairs:
                if not are_partitions_neighbors(
                    G_nx, fine_part_nodes_map[f1], fine_part_nodes_map[f2]
                ):
                    continue

                # We found a valid "bridge" between two coarse partitions. Now expand from there.
                q_fine_indices, queue, visited = [f1, f2], [f1, f2], {f1, f2}
                while queue and len(q_fine_indices) < num_frags:
                    current_fine_idx = queue.pop(0)
                    current_c_idx = fine_to_coarse_map[current_fine_idx]

                    # Look for neighbors in adjacent *coarse* partitions
                    coarse_neighbors_and_self = list(
                        coarse_part_graph.neighbors(current_c_idx)
                    ) + [current_c_idx]
                    potential_fine_neighbors = [
                        fn
                        for c_idx in coarse_neighbors_and_self
                        for fn in coarse_to_fine_map.get(c_idx, [])
                    ]
                    random.shuffle(potential_fine_neighbors)

                    for neighbor_idx in potential_fine_neighbors:
                        if neighbor_idx not in visited and are_partitions_neighbors(
                            G_nx,
                            fine_part_nodes_map[current_fine_idx],
                            fine_part_nodes_map[neighbor_idx],
                        ):
                            visited.add(neighbor_idx)
                            queue.append(neighbor_idx)
                            q_fine_indices.append(neighbor_idx)
                            if len(q_fine_indices) >= num_frags:
                                break

                if len(q_fine_indices) < num_frags:
                    continue

                true_coarse_indices = {
                    fine_to_coarse_map[f_idx] for f_idx in q_fine_indices
                }
                if len(true_coarse_indices) < min_coarse_parts:
                    continue

                # We have enough fragments spanning enough coarse partitions. Finalize the query.
                nodes_per_frag = max_nodes // num_frags
                all_query_nodes = []
                for fine_idx in q_fine_indices:
                    local_nodes = _extract_fragment(
                        fine_graphs[fine_idx], nodes_per_frag
                    )
                    if local_nodes:
                        all_query_nodes.extend(
                            [fine_part_nodes_map[fine_idx][i] for i in local_nodes]
                        )

                Gq, q_global_nodes = _finalize_query_from_nodes(
                    original_data, all_query_nodes, min_nodes, device
                )
                if Gq:
                    stitched_node_list = [
                        node
                        for idx in q_fine_indices
                        for node in fine_part_nodes_map[idx]
                    ]
                    G_stitched = original_data.subgraph(
                        torch.tensor(stitched_node_list, device=device)
                    )
                    return (
                        Gq,
                        G_stitched,
                        q_global_nodes,
                        q_fine_indices,
                        true_coarse_indices,
                    )

    raise RuntimeError(
        "Failed to generate multi-coarse partition query after trying all configurations."
    )
