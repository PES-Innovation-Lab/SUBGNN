import os

import faiss
import torch
from torch_geometric.datasets import CoraFull
from torch_geometric.utils import to_networkx

from src.config import (
    DATA_ROOT,
    DEVICE,
    GIN_HIDDEN_NEURONS,
    GIN_OUTPUT_NEURONS,
    HIERARCHY_LEVELS,
    MODEL_PATH,
    MULTI_COARSE_QUERY_MAX_NODES,
    MULTI_COARSE_QUERY_MIN_NODES,
    MULTI_FINE_QUERY_FRAGMENTS,
    MULTI_FINE_QUERY_MAX_NODES,
    MULTI_FINE_QUERY_MIN_NODES,
    SINGLE_PARTITION_QUERY_MAX_NODES,
    SINGLE_PARTITION_QUERY_MIN_NODES,
)
from src.data_processing import build_coarse_partition_graph, build_recursive_hierarchy
from src.model import SubgraphEncoder, get_graph_embedding
from src.query_generator import (
    generate_multi_coarse_partition_query,
    generate_multi_fine_partition_query,
    generate_single_partition_query,
)
from src.search_executor import search
from src.utils import feature_to_label


def main():

    print(f"Using device: {DEVICE}")

    # --- 1. Load Data and Pre-trained Model ---
    print("\n--- Loading Data and Model ---")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        exit(1)

    cora = CoraFull(root=DATA_ROOT)[0].to(DEVICE)
    encoder = SubgraphEncoder(
        in_neurons=cora.num_features,
        hidden_neurons=GIN_HIDDEN_NEURONS,
        output_neurons=GIN_OUTPUT_NEURONS,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    encoder.eval()
    print("Data and model loaded successfully.")

    # --- 2. Pre-computation: Build Hierarchy and Context ---
    print("\n--- Starting Pre-computation ---")
    (
        coarse_graphs,
        coarse_part_nodes_map,
        fine_graphs,
        fine_to_coarse_map,
        fine_part_nodes_map,
    ) = build_recursive_hierarchy(cora, levels=HIERARCHY_LEVELS, device=DEVICE)

    cora_nx = to_networkx(cora, to_undirected=True)
    coarse_part_graph = build_coarse_partition_graph(cora_nx, coarse_part_nodes_map)

    global_id_to_label_feature = {
        i: feature_to_label(cora.x[i].cpu().numpy()) for i in range(cora.num_nodes)
    }

    # The 'context' dictionary holds all pre-computed data needed by other modules
    context = {
        "original_data": cora,
        "G_nx": cora_nx,
        "coarse_graphs": coarse_graphs,
        "fine_graphs": fine_graphs,
        "coarse_part_nodes_map": coarse_part_nodes_map,
        "fine_part_nodes_map": fine_part_nodes_map,
        "fine_to_coarse_map": fine_to_coarse_map,
        "coarse_part_graph": coarse_part_graph,
        "global_id_to_label_feature": global_id_to_label_feature,
        "device": DEVICE,
    }

    # --- 3. Pre-computation: Create Faiss Index for Coarse Search ---
    print("Creating Faiss index for coarse partitions...")
    faiss_coarse = faiss.IndexFlatL2(GIN_OUTPUT_NEURONS)
    valid_coarse_graphs = [
        cg for cg in coarse_graphs if cg is not None and cg.num_nodes > 0
    ]
    if not valid_coarse_graphs:
        print("Error: No valid coarse graphs were generated. Exiting.")
        exit(1)

    coarse_embeddings = torch.cat(
        [get_graph_embedding(cg, encoder, DEVICE) for cg in valid_coarse_graphs], dim=0
    )
    faiss_coarse.add(coarse_embeddings.cpu().numpy())
    context["faiss_coarse"] = faiss_coarse
    print("Pre-computation complete.")

    # --- 4. Run Experiments ---

    # Experiment 1: Single Partition Query
    single_query_params = {
        **context,
        "min_nodes": SINGLE_PARTITION_QUERY_MIN_NODES,
        "max_nodes": SINGLE_PARTITION_QUERY_MAX_NODES,
    }
    search(
        "Single Partition Query",
        generate_single_partition_query,
        single_query_params,
        context,
        encoder,
        DEVICE,
    )

    # Experiment 2: Multi-Fine Partition Query
    multi_fine_params = {
        **context,
        "num_frags": MULTI_FINE_QUERY_FRAGMENTS,
        "min_nodes": MULTI_FINE_QUERY_MIN_NODES,
        "max_nodes": MULTI_FINE_QUERY_MAX_NODES,
    }
    search(
        "Multi-Fine Partition Query",
        generate_multi_fine_partition_query,
        multi_fine_params,
        context,
        encoder,
        DEVICE,
    )

    # Experiment 3: Multi-Coarse Partition Query
    multi_coarse_params = {
        **context,
        "min_nodes": MULTI_COARSE_QUERY_MIN_NODES,
        "max_nodes": MULTI_COARSE_QUERY_MAX_NODES,
    }
    search(
        "Multi-Coarse Partition Query",
        generate_multi_coarse_partition_query,
        multi_coarse_params,
        context,
        encoder,
        DEVICE,
    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
