import os
import faiss
import torch
import pandas as pd
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
from src.data_processing import build_recursive_hierarchy, build_coarse_partition_graph
from src.model import SubgraphEncoder, get_graph_embedding
from src.query_generator import (
    generate_single_partition_query,
    generate_multi_fine_partition_query,
    generate_multi_coarse_partition_query,
)
from src.search_executor import search
from src.utils import feature_to_label
from tqdm import tqdm


def run_partitionwise_benchmark(
    encoder,
    device,
    context,
    query_generators,
    query_params_template,
    queries_per_type=50,
):
    import copy
    import pandas as pd
    from tqdm import tqdm
    from statistics import mean, median

    all_results = []
    num_coarse_parts = len(context["coarse_graphs"])

    for query_type, generator_fn in query_generators.items():
        print(f"\nRunning benchmark for query type: {query_type.upper()} (50 queries per coarse partition)")
        results = []

        for anchor_coarse_idx in range(num_coarse_parts):
            print(f"\n→ Coarse Partition {anchor_coarse_idx} / {num_coarse_parts - 1}")
            for i in tqdm(range(queries_per_type), desc=f"{query_type} | Partition {anchor_coarse_idx}"):
                params = copy.deepcopy(query_params_template[query_type])
                params["anchor_coarse_idx"] = anchor_coarse_idx
                params["query_id"] = f"{query_type}_{anchor_coarse_idx}_{i}"

                result = search(
                    name=params["query_id"],
                    query_generator=generator_fn,
                    query_params=params,
                    context=context,
                    encoder=encoder,
                    device=device,
                )
                result["query_type"] = query_type
                result["anchor_coarse_idx"] = anchor_coarse_idx
                results.append(result)
                all_results.append(result)

        # Partition-wise aggregation
        df = pd.DataFrame(results)
        df_success = df[df["success"] == True]

        def safe_mean(col): return df_success[col].dropna().mean() if col in df_success else -1
        def safe_median(col): return df_success[col].dropna().median() if col in df_success else -1

        print(f"\nSummary for '{query_type}' queries across all partitions:")
        print(f" - Total queries run:      {len(df)}")
        print(f" - Successful queries:     {len(df_success)}")
        print(f" - Coarse accuracy:        {df_success['coarse_correct'].mean() * 100:.2f}%")
        print(f" - Fine accuracy:          {df_success['fine_correct'].mean() * 100:.2f}%")
        print(f" - Perfect match rate:     {df_success['perfect_solution_found'].mean() * 100:.2f}%")
        print(f" - First solution accuracy:{safe_mean('first_solution_accuracy'):.2f}%")
        print(f" - Time to first solution: Mean={safe_mean('time_to_first_solution'):.2f}s | Median={safe_median('time_to_first_solution'):.2f}s")
        print(f" - Time to perfect match:  Mean={safe_mean('time_to_correct_solution'):.2f}s | Median={safe_median('time_to_correct_solution'):.2f}s")
        print(f" - Time to all solutions:  Mean={safe_mean('time_to_all_solutions'):.2f}s | Median={safe_median('time_to_all_solutions'):.2f}s")
        print(f" - Avg solutions found:    {safe_mean('solutions_found'):.2f}")

    print("\nCompleted benchmark for all query types and partitions.")
    return pd.DataFrame(all_results)



def main():
    print(f"Using device: {DEVICE}")

    # 1. Load model and data
    print("\n--- Loading CoraFull & Model ---")
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        exit(1)

    cora = CoraFull(root=DATA_ROOT)[0].to(DEVICE)
    encoder = SubgraphEncoder(
        in_neurons=cora.num_features,
        hidden_neurons=GIN_HIDDEN_NEURONS,
        output_neurons=GIN_OUTPUT_NEURONS,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    encoder.eval()

    # 2. Build hierarchy
    print("\n--- Building Partition Hierarchy ---")
    coarse_graphs, coarse_part_nodes_map, fine_graphs, fine_to_coarse_map, fine_part_nodes_map = build_recursive_hierarchy(
        cora, levels=HIERARCHY_LEVELS, device=DEVICE
    )
    cora_nx = to_networkx(cora, to_undirected=True)
    coarse_part_graph = build_coarse_partition_graph(cora_nx, coarse_part_nodes_map)

    global_id_to_label_feature = {
        i: feature_to_label(cora.x[i].cpu().numpy()) for i in range(cora.num_nodes)
    }

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

    print("Preprocessing complete.")

    # 3. Coarse partition FAISS index
    print("\n--- Building FAISS Index for Coarse Search ---")
    faiss_coarse = faiss.IndexFlatL2(GIN_OUTPUT_NEURONS)
    valid_coarse_graphs = [cg for cg in coarse_graphs if cg and cg.num_nodes > 0]
    coarse_embeddings = torch.cat(
        [get_graph_embedding(g, encoder, DEVICE) for g in valid_coarse_graphs], dim=0
    )
    faiss_coarse.add(coarse_embeddings.cpu().numpy())
    context["faiss_coarse"] = faiss_coarse

    # 4. Define query generators and parameters
    query_generators = {
        "single": generate_single_partition_query,
        "multi_fine": generate_multi_fine_partition_query,
        "multi_coarse": generate_multi_coarse_partition_query,
    }

    query_params_template = {
        "single": {
            **context,
            "min_nodes": SINGLE_PARTITION_QUERY_MIN_NODES,
            "max_nodes": SINGLE_PARTITION_QUERY_MAX_NODES,
        },
        "multi_fine": {
            **context,
            "num_frags": MULTI_FINE_QUERY_FRAGMENTS,
            "min_nodes": MULTI_FINE_QUERY_MIN_NODES,
            "max_nodes": MULTI_FINE_QUERY_MAX_NODES,
        },
        "multi_coarse": {
            **context,
            "min_nodes": MULTI_COARSE_QUERY_MIN_NODES,
            "max_nodes": MULTI_COARSE_QUERY_MAX_NODES,
        },
    }

    # 5. Run full benchmark
    df = run_partitionwise_benchmark(
        encoder=encoder,
        device=DEVICE,
        context=context,
        query_generators=query_generators,
        query_params_template=query_params_template,
        queries_per_type=50,  # per partition
    )

    # Save results
    df.to_csv("partitionwise_benchmark_results.csv", index=False)
    print("\nSaved partitionwise benchmark results to partitionwise_benchmark_results.csv")


if __name__ == "__main__":
    main()
