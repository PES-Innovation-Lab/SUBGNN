from torch_geometric.datasets import TUDataset
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph, subgraph
import torch
import torch.nn.functional as F
import time

dataset = TUDataset(root="/tmp/MUTAG", name="MUTAG")

print(f"Number of graphs: {len(dataset)}")

def generate_subgraph(data, k_hops=2, anchor_node=0):
    subset, edge_index_sub, _, _ = k_hop_subgraph(anchor_node, k_hops, data.edge_index, relabel_nodes=True)
    Gq = data.subgraph(subset)
    return Gq

for i in range(len(dataset)):
    data = dataset[i]
    try:
        Gq = generate_subgraph(data)
        data, Gq = to_networkx(data), to_networkx(Gq)

        start_time = time.time()
        M = nx.algorithms.isomorphism.GraphMatcher(data, Gq)
        is_subgraph = M.subgraph_is_isomorphic()
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"Graph {i}: {is_subgraph}\tElapsed Time: {elapsed}")

    except TypeError:
        print(f"Error at graph {i}")
