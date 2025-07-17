# TODO:
# Add VF2/3 to the pipeline
# Clean up functions

# Imports

import random
import sys
import time

import faiss
import networkx as nx
import pymetis
import torch
import torch.nn.functional as F
from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential
from torch_geometric.datasets import CoraFull
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import degree, k_hop_subgraph, subgraph, to_networkx, from_networkx

import vf3py

import matplotlib.pyplot as plt

# Load dataset

DATASET = CoraFull(root="/tmp/Cora")[0]

# print("Cora Dataset")
# print("=" * 80)
# print(f"Number of nodes: {DATASET.num_nodes}")
# print(f"Number of edges: {DATASET.num_edges}")
# print(f"Number of features: {DATASET.num_node_features}")
# print("=" * 80)


# Triplet Loss
# max(0, || zq - zpos || ^ 2 - || zq - zneg || ^ 2 + alpha)
#
# Change alpha to increase margin between positive samples and negative samples
def triplet_loss(zq, zpos, zneg, alpha=1.0):
    dist_pos = F.pairwise_distance(zq, zpos, p=2)
    dist_neg = F.pairwise_distance(zq, zneg, p=2)
    loss = F.relu(dist_pos - dist_neg + alpha)
    return loss.mean()


# Generate triplets
# 1. Gq, which is sampled from Gpos
# 2. Gpos, from which Gq is sampled
# 3. Gneg, which is not a match for Gq
#
# k: k-hops to generate Gpos and Gneg
# q_size: max number of nodes in Gq
def generate_triplets(data, k=6, q_size=80):
    # Check if data has original_node_ids attribute
    if hasattr(data, 'original_node_ids'):
        # For partitioned data, work with the available nodes
        available_nodes = list(range(data.num_nodes))
        anchor_node = random.choice(available_nodes)
        
        # Create Gpos using k-hop subgraph
        subset_pos, _, _, _ = k_hop_subgraph(
            anchor_node, k, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
        )
        
        # Create Gpos 
        edge_index_pos, _ = subgraph(subset_pos, data.edge_index, relabel_nodes=True)
        Gpos = data.__class__()
        Gpos.x = data.x[subset_pos]
        Gpos.edge_index = edge_index_pos
        Gpos.num_nodes = len(subset_pos)
        
        # Sample nodes for Gq from subset_pos
        q_nodes = subset_pos[torch.randperm(len(subset_pos))][:q_size]
        
        # Create Gq
        edge_index_q, _ = subgraph(q_nodes, data.edge_index, relabel_nodes=True)
        Gq = data.__class__()
        Gq.x = data.x[q_nodes]
        Gq.edge_index = edge_index_q
        Gq.num_nodes = len(q_nodes)
        
        # Store original node IDs for Gq
        Gq.original_node_ids = data.original_node_ids[q_nodes]
        
        # Create a simple negative example by sampling different nodes
        neg_nodes = torch.randperm(data.num_nodes)[:min(len(subset_pos), data.num_nodes)]
        edge_index_neg, _ = subgraph(neg_nodes, data.edge_index, relabel_nodes=True)
        Gneg = data.__class__()
        Gneg.x = data.x[neg_nodes]
        Gneg.edge_index = edge_index_neg
        Gneg.num_nodes = len(neg_nodes)
        
    else:
        # Original logic for full dataset
        anchor_node = torch.randint(0, data.num_nodes, (1,)).item()
        subset_pos, _, _, _ = k_hop_subgraph(
            anchor_node, k, data.edge_index, relabel_nodes=True
        )

        Gpos = data.subgraph(subset_pos)

        q_nodes = subset_pos[torch.randperm(len(subset_pos))][:q_size]
        Gq = data.subgraph(q_nodes)

        neg_anchor = torch.randint(0, data.num_nodes, (1,)).item()
        attempts = 0
        while neg_anchor == anchor_node and attempts < 5:
            neg_anchor = torch.randint(0, data.num_nodes, (1,)).item()
            attempts += 1

        subset_neg, _, _, _ = k_hop_subgraph(
            neg_anchor, k, data.edge_index, relabel_nodes=True
        )
        Gneg = data.subgraph(subset_neg)

    return Gq, Gpos, Gneg


# Model
class SubgraphEncoder(torch.nn.Module):
    def __init__(self, in_neurons, hidden_neurons, output_neurons):
        super().__init__()
        nn1 = Sequential(
            Linear(in_neurons, hidden_neurons),
            Dropout(0.1),
            LeakyReLU(1.5),
            Linear(hidden_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
        )
        self.conv1 = GINConv(nn1)
        self.lin = Linear(hidden_neurons, output_neurons)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = global_mean_pool(h, batch)
        return F.normalize(self.lin(h), dim=1)


device = torch.device("cpu")

encoder = SubgraphEncoder(
    in_neurons=DATASET.num_features, hidden_neurons=64, output_neurons=16
)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

encoder = encoder.to(device)
DATASET = DATASET.to(device)

epochs = 280
steps_per_epoch = 512



# Training
def train(encoder, dataset, epochs, steps_per_epoch, device):

    for epoch in range(epochs):
        total_loss = 0

        for _ in range(steps_per_epoch):

            result = generate_triplets(dataset)
            if not result:
                continue

            Gq, Gpos, Gneg = result

            Gq.batch = torch.zeros(Gq.num_nodes, dtype=torch.long, device=device)
            Gpos.batch = torch.zeros(Gpos.num_nodes, dtype=torch.long, device=device)
            Gneg.batch = torch.zeros(Gneg.num_nodes, dtype=torch.long, device=device)

            zq = encoder(Gq.x, Gq.edge_index, Gq.batch)
            zpos = encoder(Gpos.x, Gpos.edge_index, Gpos.batch)
            zneg = encoder(Gneg.x, Gneg.edge_index, Gneg.batch)

            loss = triplet_loss(
                zq.unsqueeze(0), zpos.unsqueeze(0), zneg.unsqueeze(0), alpha=1.0
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Triplet Loss = {total_loss:.4f}")

    torch.save(encoder.state_dict(), "cora-model.pth")


# Uncomment to re-train
# train(encoder, DATASET, epochs, steps_per_epoch, device)


# Load saved model
encoder.load_state_dict(torch.load("cora-model.pth", map_location=device))
encoder.eval()


# Generate embedding for a single graph
def get_graph_embedding(dataset):
    with torch.no_grad():
        dataset.batch = torch.zeros(dataset.num_nodes, dtype=torch.long, device=device)
        z = encoder(dataset.x, dataset.edge_index, dataset.batch)

    return z


# Partition a graph using METIS
def make_partitions(dataset, num_parts):
    part_dict = {}
    for i in range(num_parts):
        part_dict[i] = []

    part_graphs = []

    nx_graph = to_networkx(dataset)
    adjacency_list = {i: list(nx_graph.neighbors(i)) for i in nx_graph.nodes}
    _, membership = pymetis.part_graph(num_parts, adjacency_list)

    for i in range(len(membership)):
        part_dict[membership[i]].append(i)

    for i in range(num_parts):
        # Use subgraph with relabel_nodes=True for proper indexing in the model
        node_indices = torch.tensor(part_dict[i]).to(device)
        edge_index, _ = subgraph(node_indices, dataset.edge_index, relabel_nodes=True)
        
        # Create subgraph data object
        subgraph_data = dataset.__class__()
        subgraph_data.x = dataset.x[node_indices]
        subgraph_data.edge_index = edge_index
        subgraph_data.num_nodes = len(node_indices)
        
        # Store original node mapping for later use
        subgraph_data.original_node_ids = node_indices
        
        part_graphs.append(subgraph_data)
        # print(f"Partition {i}:", len(part_dict[i]))

    return part_graphs


# Helper function to create NetworkX graph with original node IDs
def to_networkx_with_original_ids(data):
    """Convert PyTorch Geometric data to NetworkX graph using original node IDs"""
    G = to_networkx(data)
    
    # If original_node_ids exist, create a mapping from relabeled to original
    if hasattr(data, 'original_node_ids'):
        # Create mapping from relabeled (0,1,2...) to original node IDs
        id_mapping = {i: int(data.original_node_ids[i].item()) for i in range(len(data.original_node_ids))}
        
        # Relabel nodes in the NetworkX graph
        G = nx.relabel_nodes(G, id_mapping)
    
    return G


def main():

    # args = sys.argv[1:]
    # partition_num = 0
    #
    # if len(args) == 1:
    #     partition_num = int(args[0])
    # elif len(args) > 1:
    #     print("Wrong Usage: python3 cora_pipeline.py <Partition Number>")

    # Build vector index using FAISS
    index = faiss.IndexFlatL2(16)

    # Make partitions
    k = 15
    partition_list = make_partitions(DATASET, k)

    # Generate embeddings for each partition
    partition_embeddings = []

    for i in range(len(partition_list)):
        emb = get_graph_embedding(partition_list[i])
        partition_embeddings.append(emb)
        index.add(emb.cpu())
                
    # Sample query
    # Generate a query graph from Partition "partition_num", so we know it will always be an exact match
    Gq, Gpos, Gneg = generate_triplets(partition_list[0])
    while(Gq.num_nodes < 50):
        Gq, Gpos, Gneg = generate_triplets(partition_list[0])
    # Generate query embedding
    # model_start_time = time.time()
    zq = get_graph_embedding(Gq)
    # model_end_time = time.time(
    # Search for query embedding within FAISS index
    # faiss_start_time = time.time()
    D, I = index.search(zq.cpu(), k)
    # faiss_end_time = time.time(
    # model_time = (model_end_time - model_start_time) * 1000
    # faiss_time = (faiss_end_time - faiss_start_time) * 100
    # print("Probable Partitions:", I)
    # print("Distances:", D[0])
    most_prob=I[0][0]

    # Create NetworkX graphs with original node IDs
    Gt = to_networkx_with_original_ids(partition_list[most_prob])
    Gq = to_networkx_with_original_ids(Gq)
    
    # Display node IDs
    # print("Gq node IDs (using original IDs):", list(Gq.nodes()))
    # print("Gt node IDs (using original IDs):", list(Gt.nodes()))
    
    # For VF3, we'll use the NetworkX graphs with original node IDs

    # print("\n\nRunning Model + FAISS...")
    # print("=" * 80)
    #
    # print("Number of query nodes:", query_nodes
    # print("Most Probable Partition:", most_prob
    # print("Probable Partitions:", I)
    # print("Distances:", D[0]
    # print(f"Input Partition: {partition_num}\t\tMost Probable: {most_prob}\t\tSecond: {second_prob}\t\tDistances: {D[0][0]}\t\tNodes: {query_nodes}"
    # prinf"Errors: {count}"
        # total_time = model_time + faiss_time
        #
        # print("Model Time:", model_time)
        # print("FAISS Time:", faiss_time)
        # print("Total Time:", total_time)


    # vf3_start_time = time.time()
    # vf3_is_subgraph = vf3py.has_subgraph(Gq, Gt)
    # vf3_is_subgraph = vf3py.has_subgraph(Gq, d)

    # Choose a random node from Gq for start_target_node_id
    random_gq_node = int(random.choice(list(Gq.nodes())))
    print(f"random Gq node = start_target_node_id: {random_gq_node}")
    
    vf3_isomorphisms = vf3py.get_subgraph_isomorphisms(Gq, Gt, start_target_node_id=random_gq_node)
    # vf3_end_time = time.time()
    # vf3_time = (vf3_end_time - vf3_start_time) * 1000
    # print(f"VF3 Result: {vf3_is_subgraph}")
    # print(f"VF3 Time: {vf3_time}")
    
    print(f"VF3 Mapping: {vf3_isomorphisms[0]}")



if __name__ == "__main__":
    main()
