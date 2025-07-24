import random
import re
import subprocess
from typing import Any, Callable, Dict

import faiss
import torch
from torch_geometric.data import Data

from src.config import (
    COARSE_PARTITION_CSV_PATH,
    FAISS_TOP_K,
    QUERY_CSV_PATH,
    SOLVER_PATH,
)
from src.model import SubgraphEncoder, get_graph_embedding
from src.utils import convert_graph_to_csv


def _run_glasgow_solver(
    Gq,
    q_global_nodes: list,
    target_graph: Data,
    target_local_to_global_map: Dict[int, int],
    start_node_global_id: int,
    context: Dict[str, Any],
) -> bool:
    """
    Prepares CSV files and runs the external Glasgow Subgraph Solver,
    parsing its output to check for a correct mapping.
    """
    print("\n--- STEP 3: SUBGRAPH ISOMORPHISM SEARCH ---")

    # Create the query CSV
    query_local_to_global_map = {i: g_id for i, g_id in enumerate(q_global_nodes)}
    convert_graph_to_csv(
        Gq,
        QUERY_CSV_PATH,
        context["global_id_to_label_feature"],
        query_local_to_global_map,
    )

    # Create the target graph CSV
    convert_graph_to_csv(
        target_graph,
        COARSE_PARTITION_CSV_PATH,
        context["global_id_to_label_feature"],
        target_local_to_global_map,
    )

    target_global_to_local_map = {
        g_id: i for i, g_id in target_local_to_global_map.items()
    }
    start_node_local_id = target_global_to_local_map.get(start_node_global_id)

    if start_node_local_id is None:
        print(
            f"[Warning] Start hint (Global ID: {start_node_global_id}) not found in the current search space. Picking a random node."
        )
        if target_graph.num_nodes > 0:
            start_node_local_id = random.choice(range(target_graph.num_nodes))
        else:
            print("[ERROR] Target graph for solver is empty. Cannot run search.")
            return 0

    print(f"Search Space: ({target_graph.num_nodes} nodes).")
    # print(f"Start Node Hint (Global ID): {start_node_global_id}")
    # print(f"Start Node Hint (Local ID): {start_node_local_id}")

    cmd = [
        SOLVER_PATH,
        "--count-solutions",
        "--induced",
        "--print-all-solutions",
        # "--start-from-target",
        # str(start_node_local_id),
        QUERY_CSV_PATH,
        COARSE_PARTITION_CSV_PATH,
    ]

    mapping_pattern = re.compile(r"^mapping\s*=\s*(.*)$")
    solution_count_pattern = re.compile(r"^solution_count\s*=\s*(\d+)$")
    solution_found = False

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        for line in proc.stdout:
            line = line.strip()
            if mapping_match := mapping_pattern.match(line):
                mapping_data = mapping_match.group(1)
                pairs = re.findall(r"\((\d+)\s*->\s*(\d+)\)", mapping_data)

                # The solver maps query local ID -> target local ID.
                # Since target is the full graph, target local ID == global ID.
                glasgow_mapping = {
                    int(q_local): int(target_global) for q_local, target_global in pairs
                }

                correct_mappings = 0
                total_mappings = 0

                for q_local_id, predicted_local_id in glasgow_mapping.items():
                    if q_local_id < Gq.num_nodes:
                        total_mappings += 1
                        true_query_global_id = query_local_to_global_map.get(q_local_id)
                        true_target_global_id = target_local_to_global_map.get(
                            predicted_local_id
                        )
                        if true_query_global_id == true_target_global_id:
                            correct_mappings += 1

                if total_mappings > 0:
                    accuracy = (correct_mappings / total_mappings) * 100
                    print(f"Solver found a solution with accuracy: {accuracy:.2f}%")
                    if accuracy == 100.0:
                        print("(SUCCESS) Perfect solution found!")
                        solution_found = True
                        # proc.terminate()  # Stop searching after finding the first perfect match
                        # proc.wait()
                        # break

            elif solution_count_match := solution_count_pattern.match(line):
                solution_count = int(solution_count_match.group(1))
                print(f"\nTotal Solutions Found: {solution_count}\n")

    if not solution_found:
        print("  (FAILURE) Solver did not find the correct subgraph.")

    return solution_found


def search(
    name: str,
    query_generator: Callable,
    query_params: Dict[str, Any],
    context: Dict[str, Any],
    encoder: SubgraphEncoder,
    device: torch.device,
):
    """
    Executes a full search experiment:
    1. Generates a query graph.
    2. Performs a coarse-level search using FAISS.
    3. Performs a fine-level search within the predicted coarse partition.
    4. Uses the fine-level prediction as a hint for the subgraph isomorphism solver.
    """
    print("\n" + "#" * 70 + f"\n###   STARTING EXPERIMENT: {name}   ###\n" + "#" * 70)

    try:
        Gq, _, q_global_nodes, true_fine_indices, true_coarse_indices = query_generator(
            **query_params
        )
    except (RuntimeError, IndexError) as e:
        print(f"\n[ERROR] Could not generate query for '{name}': {e}")
        return

    print("\n--- GROUND TRUTH & QUERY DETAILS ---")
    print(f"Generated Query Graph (Gq):    {Gq.num_nodes} nodes, {Gq.num_edges} edges")
    print(f"Source Fine Partition(s):      {sorted(true_fine_indices)}")
    print(f"Source Coarse Container(s):      {sorted(list(true_coarse_indices))}")

    # Generate query embedding
    zq = get_graph_embedding(Gq, encoder, device)

    # --- STEP 1: COARSE-LEVEL SEARCH ---
    print("\n--- STEP 1: COARSE-LEVEL SEARCH ---")
    faiss_coarse = context["faiss_coarse"]
    D_coarse, I_coarse = faiss_coarse.search(zq.cpu().numpy(), FAISS_TOP_K)
    predicted_coarse_idx = I_coarse[0][0].item()

    print(f"Predicted Coarse Partition: Index {predicted_coarse_idx}")
    if predicted_coarse_idx in true_coarse_indices:
        print(
            f"  (CORRECT) Prediction is in the true set {sorted(list(true_coarse_indices))}. (Dist: {D_coarse[0][0]:.4f})"
        )
    else:
        print(
            f"  (INCORRECT) True containers were {sorted(list(true_coarse_indices))}."
        )

    # --- STEP 2: FINE-LEVEL SEARCH ---
    print("\n--- STEP 2: FINE-LEVEL SEARCH (for start node hint) ---")
    fine_graphs, fine_to_coarse_map = (
        context["fine_graphs"],
        context["fine_to_coarse_map"],
    )

    candidate_fine_graphs, candidate_global_indices = [], []
    for f_idx, c_idx in fine_to_coarse_map.items():
        if c_idx == predicted_coarse_idx:
            if f_idx < len(fine_graphs) and fine_graphs[f_idx] is not None:
                candidate_fine_graphs.append(fine_graphs[f_idx])
                candidate_global_indices.append(f_idx)

    if not candidate_fine_graphs:
        print(
            "[ERROR] No valid candidate fine partitions found in the predicted coarse partition. Stopping."
        )
        return

    faiss_fine = faiss.IndexFlatL2(encoder.lin.out_features)
    fine_embeddings = torch.cat(
        [get_graph_embedding(fg, encoder, device) for fg in candidate_fine_graphs],
        dim=0,
    )
    faiss_fine.add(fine_embeddings.cpu().numpy())
    D_fine, I_fine = faiss_fine.search(zq.cpu().numpy(), 1)

    predicted_fine_local_idx = I_fine[0][0].item()
    predicted_fine_global_idx = candidate_global_indices[predicted_fine_local_idx]

    print(f"Most Likely Fine Partition (Hint): Index {predicted_fine_global_idx}")
    if predicted_fine_global_idx in true_fine_indices:
        print(
            f"  (HINT IS CORRECT) The hint partition is one of the true sources. (Dist: {D_fine[0][0]:.4f})"
        )
    else:
        print("  (HINT IS INCORRECT) The hint partition is not a true source.")

    # --- STEP 3: SUBGRAPH ISOMORPHISM ---

    # --- Attempt 1: Search within the single predicted coarse partition ---
    print(
        f"\n[Attempt 1] Searching within predicted coarse partition index {predicted_coarse_idx}."
    )
    search_space_graph = context["coarse_graphs"][predicted_coarse_idx]
    search_space_map = {
        i: g_id
        for i, g_id in enumerate(context["coarse_part_nodes_map"][predicted_coarse_idx])
    }

    fine_partition_global_nodes = context["fine_part_nodes_map"].get(
        predicted_fine_global_idx, []
    )
    if not fine_partition_global_nodes:
        print(
            "[ERROR] Predicted fine partition has no nodes. Cannot select start hint. Stopping."
        )
        return

    # Select a random node from the predicted fine partition to use as a starting hint
    start_node_hint_global_id = random.choice(fine_partition_global_nodes)

    solution_found = _run_glasgow_solver(
        Gq,
        q_global_nodes,
        search_space_graph,
        search_space_map,
        start_node_hint_global_id,
        context,
    )

    # --- Attempt 2 (Fallback): If no solution, expand search to neighbors ---
    if not solution_found:
        print(
            f"\n[Attempt 2] Perfect solution not found. Expanding search space to include neighbors of coarse partition index {predicted_coarse_idx}."
        )

        coarse_part_graph = context["coarse_part_graph"]
        neighbors = list(coarse_part_graph.neighbors(predicted_coarse_idx))
        partitions_to_merge_indices = [predicted_coarse_idx] + neighbors

        print(f"  Merging partitions: {partitions_to_merge_indices}")

        all_nodes_global = set()
        for c_idx in partitions_to_merge_indices:
            all_nodes_global.update(context["coarse_part_nodes_map"][c_idx])

        sorted_nodes_global = sorted(list(all_nodes_global))

        # Create the new, larger search space graph
        merged_graph = context["original_data"].subgraph(
            torch.tensor(sorted_nodes_global, device=device)
        )
        merged_local_to_global_map = {
            i: g_id for i, g_id in enumerate(sorted_nodes_global)
        }

        print(f"  New search space has {merged_graph.num_nodes} nodes.")

        _run_glasgow_solver(
            Gq,
            q_global_nodes,
            merged_graph,
            merged_local_to_global_map,
            start_node_hint_global_id,
            context,
        )
    else:
        print("\nSolution found in the first attempt. No expansion needed.")
