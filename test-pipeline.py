import sys
import tempfile
import time

import faiss
import pymetis
import torch
import torch.nn.functional as F
from torch.nn import Dropout, LeakyReLU, Linear, ReLU, Sequential
from torch_geometric.datasets import CoraFull
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.utils import k_hop_subgraph, to_networkx

sys.path.append("./modified_glasgow/")
from glasgow_subgraph_solver import GlasgowSubgraphSolver

DATASET = CoraFull(root="/tmp/Cora")[0]
DATASET.y = torch.arange(DATASET.num_nodes)


def triplet_loss(zq, zpos, zneg, alpha=1.0):
    dist_pos = F.pairwise_distance(zq, zpos, p=2)
    dist_neg = F.pairwise_distance(zq, zneg, p=2)
    loss = F.relu(dist_pos - dist_neg + alpha)
    return loss.mean()


# --- GNN Model and Helper Functions ---
class SubgraphEncoder(torch.nn.Module):
    def __init__(self, in_neurons, hidden_neurons, output_neurons):
        super().__init__()

        # Layer 1
        nn1 = Sequential(
            Linear(in_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_neurons)

        # Layer 2
        nn2 = Sequential(
            Linear(hidden_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_neurons)

        # Layer 3
        nn3 = Sequential(
            Linear(hidden_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_neurons)

        # The final linear layer takes the concatenated embeddings from all layers
        self.lin = Linear(hidden_neurons * 3, output_neurons)

    def forward(self, x, edge_index, batch):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))

        # Hierarchical Readout: Pool the output of each layer and concatenate
        h_final = torch.cat(
            [
                global_mean_pool(h1, batch),
                global_mean_pool(h2, batch),
                global_mean_pool(h3, batch),
            ],
            dim=1,
        )

        return F.normalize(self.lin(h_final), dim=1)


def generate_triplets(data, k=6, q_size=80):
    anchor_node = torch.randint(0, data.num_nodes, (1,)).item()
    subset_pos, _, _, _ = k_hop_subgraph(
        anchor_node, k, data.edge_index, relabel_nodes=False
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
        neg_anchor, k, data.edge_index, relabel_nodes=False
    )
    Gneg = data.subgraph(subset_neg)

    return Gq, Gpos, Gneg


device = torch.device("cpu")

encoder = SubgraphEncoder(
    in_neurons=DATASET.num_features, hidden_neurons=64, output_neurons=16
)
optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

encoder = encoder.to(device)
DATASET = DATASET.to(device)

epochs = 280
steps_per_epoch = 512


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

    torch.save(encoder.state_dict(), "cora-model-jigsaw.pth")


# Uncomment to re-train
# train(encoder, DATASET, epochs, steps_per_epoch, device)

# Load saved model
encoder.load_state_dict(torch.load("cora-model-jigsaw.pth", map_location=device))
encoder.eval()


def get_graph_embedding(dataset):
    with torch.no_grad():
        dataset.batch = torch.zeros(dataset.num_nodes, dtype=torch.long, device=device)
        z = encoder(dataset.x, dataset.edge_index, dataset.batch)

    return z


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
        part_graphs.append(dataset.subgraph(torch.tensor(part_dict[i]).to(device)))

    return part_graphs, part_dict


def write_graph_to_csv_format(G, filename):
    """
    Write a NetworkX graph to CSV format for Glasgow solver.
    CSV format: edge_list with two columns (source, target)
    """
    import csv

    # Create a mapping from node IDs to sequential integers starting from 0
    nodes = sorted(G.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        # writer.writerow(["source", "target"])

        # Write edges using mapped indices
        for edge in G.edges():
            source_idx = node_to_index[edge[0]]
            target_idx = node_to_index[edge[1]]
            writer.writerow([source_idx, target_idx])

    return node_to_index  # Return mapping for later use


def write_graph_to_lad_format(G, filename):
    """
    Write a NetworkX graph to LAD format for Glasgow solver.
    LAD format:
    - First line: number of vertices
    - Following lines: adjacency list for each vertex
    """
    with open(filename, "w") as f:
        # Number of vertices
        f.write(f"{G.number_of_nodes()}\n")

        # Create a mapping from node IDs to sequential integers starting from 0
        nodes = sorted(G.nodes())
        node_to_index = {node: i for i, node in enumerate(nodes)}

        # Write adjacency list for each vertex
        for node in nodes:
            neighbors = sorted(
                [node_to_index[neighbor] for neighbor in G.neighbors(node)]
            )
            f.write(f"{len(neighbors)}")
            for neighbor in neighbors:
                f.write(f" {neighbor}")
            f.write("\n")

    return node_to_index  # Return mapping for later use


if __name__ == "__main__":

    args = sys.argv[1:]
    partition_num = 0

    if len(args) == 1:
        partition_num = int(args[0])
    elif len(args) > 1:
        print("Wrong Usage: python3 cora_pipeline.py <Partition Number>")

    graph_index = faiss.IndexFlatL2(16)
    partition_node_index = faiss.IndexFlatL2(8710)
    query_node_index = faiss.IndexFlatL2(8710)

    num_parts = 20
    partition_list, partition_dict = make_partitions(DATASET, num_parts)

    partition_embeddings = []

    for i in range(len(partition_list)):
        emb = get_graph_embedding(partition_list[i])
        partition_embeddings.append(emb)
        graph_index.add(emb.cpu())

    Gq, Gpos, Gneg = generate_triplets(partition_list[partition_num])
    while Gq.num_nodes < 50:
        Gq, Gpos, Gneg = generate_triplets(partition_list[partition_num])

    zq = get_graph_embedding(Gq)
    D_part, I_part = graph_index.search(zq.cpu(), num_parts)

    most_prob_partition = I_part[0][0]

    print("\n\nRunning Model...")
    print("=" * 80)
    print(f"Most Probable Partition: {most_prob_partition}")

    for partition_node_emb in partition_list[most_prob_partition].x:
        partition_node_index.add(partition_node_emb.unsqueeze(0).cpu())

    D_node, I_node = partition_node_index.search(Gq.x[0].unsqueeze(0), 10)

    most_prob_node = I_node[0][0]

    G_part = to_networkx(partition_list[most_prob_partition], node_attrs=["x", "y"])
    G_query = to_networkx(Gq, node_attrs=["x", "y"])
    G_target = to_networkx(DATASET, node_attrs=["x", "y"])

    print(f"Most Probable Node: {most_prob_node}")
    print(I_node)
    print(D_node)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lad", delete=False
    ) as query_file:
        query_mapping = write_graph_to_lad_format(G_query, query_file.name)
        query_filename = query_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lad", delete=False
    ) as target_file:
        target_mapping = write_graph_to_lad_format(G_target, target_file.name)
        target_filename = target_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lad", delete=False
    ) as part_file:
        part_mapping = write_graph_to_lad_format(G_part, part_file.name)
        part_filename = part_file.name

    #################################################################################################################

    print(
        f"\n\nRunning Glasgow Subgraph Solver on Partition {most_prob_partition} with start node {most_prob_node}..."
    )
    print("=" * 80)

    try:
        # Convert the CSV files and use custom function
        solver = GlasgowSubgraphSolver()

        # Run to find the match using CSV format files
        glasgow_start_time = time.time()
        result = solver.solve(
            query_filename, part_filename, start_from_vertex=most_prob_node
        )
        glasgow_end_time = time.time()
        glasgow_time = (glasgow_end_time - glasgow_start_time) * 1000

        print(f"Glasgow Result: {result.get('success', False)}")
        print(f"Glasgow Time: {glasgow_time:.2f} ms")

        if result.get("success", False) and result.get("mappings"):
            glasgow_mapping = result["mappings"][0] if result["mappings"] else None
            if glasgow_mapping:
                # Convert CSV mapping back to original node IDs
                reverse_query_mapping = {v: k for k, v in query_mapping.items()}
                reverse_target_mapping = {v: k for k, v in target_mapping.items()}

                original_mapping = {}
                for query_csv_idx, target_csv_idx in glasgow_mapping.items():
                    query_original = reverse_query_mapping[int(query_csv_idx)]
                    target_original = reverse_target_mapping[int(target_csv_idx)]
                    original_mapping[query_original] = target_original

                print(f"Glasgow Mapping (Original node IDs): {original_mapping}")

    except Exception as e:
        print(f"Glasgow solver error: {e}")

    #################################################################################################################

    print(
        f"\n\nRunning Glasgow Subgraph Solver on Target Graph with start node {most_prob_node}..."
    )
    print("=" * 80)

    try:
        # Convert the CSV files and use custom function
        solver = GlasgowSubgraphSolver()

        # Run to find the match using CSV format files
        glasgow_start_time = time.time()
        result = solver.solve(
            query_filename, target_filename, start_from_vertex=most_prob_node
        )
        glasgow_end_time = time.time()
        glasgow_time = (glasgow_end_time - glasgow_start_time) * 1000

        print(f"Glasgow Result: {result.get('success', False)}")
        print(f"Glasgow Time: {glasgow_time:.2f} ms")

        if result.get("success", False) and result.get("mappings"):
            glasgow_mapping = result["mappings"][0] if result["mappings"] else None
            if glasgow_mapping:
                # Convert CSV mapping back to original node IDs
                reverse_query_mapping = {v: k for k, v in query_mapping.items()}
                reverse_target_mapping = {v: k for k, v in target_mapping.items()}

                original_mapping = {}
                for query_csv_idx, target_csv_idx in glasgow_mapping.items():
                    query_original = reverse_query_mapping[int(query_csv_idx)]
                    target_original = reverse_target_mapping[int(target_csv_idx)]
                    original_mapping[query_original] = target_original

                print(f"Glasgow Mapping (Original node IDs): {original_mapping}")

    except Exception as e:
        print(f"Glasgow solver error: {e}")

    #################################################################################################################

    print(
        "\n\nRunning vanilla Glasgow Subgraph Solver on Target Graph without start node..."
    )
    print("=" * 80)

    try:
        # Convert the CSV files and use custom function
        solver = GlasgowSubgraphSolver()

        # Run to find the match using CSV format files
        glasgow_start_time = time.time()
        result = solver.solve(query_filename, target_filename)
        glasgow_end_time = time.time()
        glasgow_time = (glasgow_end_time - glasgow_start_time) * 1000

        print(f"Glasgow Result: {result.get('success', False)}")
        print(f"Glasgow Time: {glasgow_time:.2f} ms")

        if result.get("success", False) and result.get("mappings"):
            glasgow_mapping = result["mappings"][0] if result["mappings"] else None
            if glasgow_mapping:
                # Convert CSV mapping back to original node IDs
                reverse_query_mapping = {v: k for k, v in query_mapping.items()}
                reverse_target_mapping = {v: k for k, v in target_mapping.items()}

                original_mapping = {}
                for query_csv_idx, target_csv_idx in glasgow_mapping.items():
                    query_original = reverse_query_mapping[int(query_csv_idx)]
                    target_original = reverse_target_mapping[int(target_csv_idx)]
                    original_mapping[query_original] = target_original

                print(f"Glasgow Mapping (Original node IDs): {original_mapping}")

    except Exception as e:
        print(f"Glasgow solver error: {e}")

    #################################################################################################################

    print(
        f"\n\nRunning vanilla Glasgow Subgraph Solver on Partition {most_prob_partition} without start node..."
    )
    print("=" * 80)

    try:
        # Convert the CSV files and use custom function
        solver = GlasgowSubgraphSolver()

        # Run to find the match using CSV format files
        glasgow_start_time = time.time()
        result = solver.solve(query_filename, part_filename)
        glasgow_end_time = time.time()
        glasgow_time = (glasgow_end_time - glasgow_start_time) * 1000

        print(f"Glasgow Result: {result.get('success', False)}")
        print(f"Glasgow Time: {glasgow_time:.2f} ms")

        if result.get("success", False) and result.get("mappings"):
            glasgow_mapping = result["mappings"][0] if result["mappings"] else None
            if glasgow_mapping:
                # Convert CSV mapping back to original node IDs
                reverse_query_mapping = {v: k for k, v in query_mapping.items()}
                reverse_target_mapping = {v: k for k, v in target_mapping.items()}

                original_mapping = {}
                for query_csv_idx, target_csv_idx in glasgow_mapping.items():
                    query_original = reverse_query_mapping[int(query_csv_idx)]
                    target_original = reverse_target_mapping[int(target_csv_idx)]
                    original_mapping[query_original] = target_original

                print(f"Glasgow Mapping (Original node IDs): {original_mapping}")

    except Exception as e:
        print(f"Glasgow solver error: {e}")
