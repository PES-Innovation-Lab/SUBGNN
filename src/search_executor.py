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


import time
import re
import subprocess
from typing import Any, Dict, Tuple
from torch_geometric.data import Data
from src.utils import convert_graph_to_csv

def _run_glasgow_solver(
    Gq,
    q_global_nodes: list,
    target_graph: Data,
    target_local_to_global_map: Dict[int, int],
    start_node_global_id: int,
    context: Dict[str, Any],
) -> Tuple[bool, int, float, float, float, float]:
    """
    Runs the Glasgow solver, tracks timing and accuracy metrics.
    Returns:
        - solution_found (bool)
        - total_solutions_found (int)
        - first_solution_accuracy (float)
        - time_to_first_solution (float)
        - time_to_correct_solution (float)
        - total_time_to_all_solutions (float)
    """
    print("\n--- STEP 3: SUBGRAPH ISOMORPHISM SEARCH ---")

    query_local_to_global_map = {i: g_id for i, g_id in enumerate(q_global_nodes)}
    convert_graph_to_csv(
        Gq, context["query_csv_path"],
        context["global_id_to_label_feature"],
        query_local_to_global_map,
    )
    convert_graph_to_csv(
        target_graph, context["target_csv_path"],
        context["global_id_to_label_feature"],
        target_local_to_global_map,
    )

    target_global_to_local_map = {
        g_id: i for i, g_id in target_local_to_global_map.items()
    }
    start_node_local_id = target_global_to_local_map.get(start_node_global_id)

    if start_node_local_id is None:
        print(f"[Warning] Start node {start_node_global_id} not found in target. Using random fallback.")
        if target_graph.num_nodes > 0:
            start_node_local_id = random.choice(range(target_graph.num_nodes))
        else:
            print("[ERROR] Target graph is empty.")
            return False, 0, -1, -1, -1, -1

    cmd = [
        context["solver_path"],
        "--count-solutions",
        "--induced",
        "--print-all-solutions",
        context["query_csv_path"],
        context["target_csv_path"],
    ]

    mapping_pattern = re.compile(r"^mapping\s*=\s*(.*)$")
    solution_count_pattern = re.compile(r"^solution_count\s*=\s*(\d+)$")

    solution_found = False
    solution_count = 0
    first_accuracy = -1
    time_to_first_solution = -1
    time_to_correct_solution = -1
    total_time_to_all = -1

    start_time = time.time()

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            line = line.strip()

            if mapping_match := mapping_pattern.match(line):
                now = time.time()
                if time_to_first_solution < 0:
                    time_to_first_solution = now - start_time

                pairs = re.findall(r"\((\d+)\s*->\s*(\d+)\)", mapping_match.group(1))
                glasgow_mapping = {int(q): int(t) for q, t in pairs}

                correct, total = 0, 0
                for q_local, t_local in glasgow_mapping.items():
                    total += 1
                    q_gid = query_local_to_global_map.get(q_local)
                    t_gid = target_local_to_global_map.get(t_local)
                    if q_gid == t_gid:
                        correct += 1
                accuracy = (correct / total) * 100 if total else 0
                if first_accuracy < 0:
                    first_accuracy = accuracy

                print(f"Solver found a solution with accuracy: {accuracy:.2f}%")
                if accuracy == 100.0 and not solution_found:
                    print("(SUCCESS) Perfect solution found!")
                    time_to_correct_solution = now - start_time
                    solution_found = True

            elif count_match := solution_count_pattern.match(line):
                solution_count = int(count_match.group(1))
                total_time_to_all = time.time() - start_time
                print(f"\nTotal Solutions Found: {solution_count}")

    if not solution_found:
        print("  (FAILURE) No perfect solution found.")

    return (
        solution_found,
        solution_count,
        first_accuracy,
        time_to_first_solution,
        time_to_correct_solution,
        total_time_to_all,
    )


def search(
    name: str,
    query_generator: Callable,
    query_params: Dict[str, Any],
    context: Dict[str, Any],
    encoder: SubgraphEncoder,
    device: torch.device,
) -> Dict[str, Any]:
    print("\n" + "#" * 70 + f"\n###   STARTING EXPERIMENT: {name}   ###\n" + "#" * 70)

    try:
        Gq, G_truth_vis, q_global_nodes, true_fine_indices, true_coarse_indices = query_generator(
            **query_params
        )
    except (RuntimeError, IndexError) as e:
        print(f"\n[ERROR] Could not generate query for '{name}': {e}")
        return {"error": str(e), "success": False}

    print("\n--- GROUND TRUTH & QUERY DETAILS ---")
    print(f"Query: {Gq.num_nodes} nodes, {Gq.num_edges} edges")
    print(f"Fine Source Partitions: {sorted(true_fine_indices)}")
    print(f"Coarse Containers: {sorted(list(true_coarse_indices))}")

    zq = get_graph_embedding(Gq, encoder, device)

    print("\n--- STEP 1: COARSE-LEVEL SEARCH ---")
    D_coarse, I_coarse = context["faiss_coarse"].search(zq.cpu().numpy(), FAISS_TOP_K)
    predicted_coarse_idx = I_coarse[0][0].item()
    correct_coarse = predicted_coarse_idx in true_coarse_indices
    print(f"Predicted Coarse Partition: {predicted_coarse_idx}")
    print(
        "(CORRECT)" if correct_coarse else "(INCORRECT)",
        f"True: {sorted(list(true_coarse_indices))}, Dist: {D_coarse[0][0]:.4f}"
    )

    print("\n--- STEP 2: FINE-LEVEL SEARCH ---")
    fine_graphs = context["fine_graphs"]
    fine_to_coarse = context["fine_to_coarse_map"]

    candidate_fines = [
        (i, fine_graphs[i])
        for i, c in fine_to_coarse.items()
        if c == predicted_coarse_idx and fine_graphs[i] is not None
    ]
    if not candidate_fines:
        print("[ERROR] No fine candidates for predicted coarse.")
        return {"error": "No fine candidates", "success": False}

    fine_embeds = torch.cat(
        [get_graph_embedding(fg, encoder, device) for _, fg in candidate_fines], dim=0
    )
    faiss_fine = faiss.IndexFlatL2(fine_embeds.shape[1])
    faiss_fine.add(fine_embeds.cpu().numpy())
    _, I_fine = faiss_fine.search(zq.cpu().numpy(), 1)

    predicted_fine_global_idx = candidate_fines[I_fine[0][0].item()][0]
    correct_fine = predicted_fine_global_idx in true_fine_indices
    print(f"Predicted Fine Partition (Hint): {predicted_fine_global_idx}")
    print(
        "(HINT CORRECT)"
        if correct_fine
        else "(HINT INCORRECT)"
    )

    # STEP 3 – ATTEMPT 1
    print(f"\n[Attempt 1] Searching within Coarse Partition {predicted_coarse_idx}")
    search_graph = context["coarse_graphs"][predicted_coarse_idx]
    search_map = {
        i: g for i, g in enumerate(context["coarse_part_nodes_map"][predicted_coarse_idx])
    }

    start_hint_gid = random.choice(context["fine_part_nodes_map"][predicted_fine_global_idx])

    results = _run_glasgow_solver(
        Gq,
        q_global_nodes,
        search_graph,
        search_map,
        start_hint_gid,
        {
            **context,
            "query_csv_path": QUERY_CSV_PATH,
            "target_csv_path": COARSE_PARTITION_CSV_PATH,
            "solver_path": SOLVER_PATH,
        },
    )

    (
        found,
        count,
        acc1,
        t_first,
        t_correct,
        t_total,
    ) = results

    if not found:
        print(f"\n[Attempt 2] Expanding to neighbors of {predicted_coarse_idx}")
        coarse_neighbors = list(context["coarse_part_graph"].neighbors(predicted_coarse_idx))
        merge_indices = [predicted_coarse_idx] + coarse_neighbors
        all_nodes = set()
        for idx in merge_indices:
            all_nodes.update(context["coarse_part_nodes_map"][idx])
        sorted_nodes = sorted(all_nodes)

        merged_graph = context["original_data"].subgraph(
            torch.tensor(sorted_nodes, device=device)
        )
        merged_map = {i: g for i, g in enumerate(sorted_nodes)}

        results = _run_glasgow_solver(
            Gq,
            q_global_nodes,
            merged_graph,
            merged_map,
            start_hint_gid,
            {
                **context,
                "query_csv_path": QUERY_CSV_PATH,
                "target_csv_path": COARSE_PARTITION_CSV_PATH,
                "solver_path": SOLVER_PATH,
            },
        )
        (
            found,
            count,
            acc1,
            t_first,
            t_correct,
            t_total,
        ) = results

    print("\n--- FINAL REPORT ---")
    print(f"Solutions Found: {count}")
    if acc1 >= 0:
        print(f"First Solution Accuracy: {acc1:.2f}%")
    if t_first >= 0:
        print(f"Time to First Solution: {t_first:.2f}s")
    if t_correct >= 0:
        print(f"Time to First Perfect Solution: {t_correct:.2f}s")
    else:
        print("No perfect solution found.")
    if t_total >= 0:
        print(f"Time to All Solutions: {t_total:.2f}s")

    return {
        "success": True,
        "experiment_name": name,
        "coarse_correct": correct_coarse,
        "fine_correct": correct_fine,
        "solutions_found": count,
        "first_solution_accuracy": acc1,
        "time_to_first_solution": t_first,
        "time_to_correct_solution": t_correct,
        "time_to_all_solutions": t_total,
        "perfect_solution_found": found,
        "predicted_coarse_idx": predicted_coarse_idx,
        "predicted_fine_idx": predicted_fine_global_idx,
        "true_fine_indices": true_fine_indices,
        "true_coarse_indices": list(true_coarse_indices),
        "Gq": Gq,
        "G_truth_vis": G_truth_vis,
        "G_predicted": context["fine_graphs"][predicted_fine_global_idx],
        "q_global_nodes": q_global_nodes,
    }