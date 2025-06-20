"""
General plan
    1.  Generate a synthetic graph
    2.	Use a GNN to extract embeddings for each node
    3.	Cluster nodes in embedding space (using FAISS or similar)
    4.	Linearize the node order based on proximity in that space
    5.	Use a learned index (like an RMI) to predict node positions on this new 1D
layout
"""

# make sure to have these modules
# pip install -q torch-geometric
# pip install python-louvain

# using networkx to generate graphs
import networkx as nx
import numpy as np

# used to convert to gnn library format
import torch
import torch.nn as nn

# for GCN
import torch.nn.functional as F
import torch_geometric

# for clustering
from sklearn.cluster import KMeans

# for linearisation using PCA
from sklearn.decomposition import PCA

# to train the RMI
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# using the Amazon dataset
from torch_geometric.datasets import Amazon
from torch_geometric.nn import GCNConv

# torch_geometric still needs CUDA for larger graphs
from torch_geometric.utils import from_networkx

# 1)defining real world data

# use this if needed
# from torch_geometric.datasets import FacebookPagePage
# from torch_geometric.transforms import NormalizeFeatures
# dataset = FacebookPagePage(root='./data/Facebook', transform=NormalizeFeatures())
# data1 = dataset[0]

# print("Dataset: Facebook Page-Page Network (from torch_geometric.datasets.FacebookPagePage)")
# print(f"  - Nodes: {data1.num_nodes}")
# print(f"  - Edges: {data1.num_edges}")
# print(f"  - Features: {data1.num_features}")
# print(f"  - Classes: {len(data1.y.unique())}")  # Shows number of unique classes
# print("-" * 30)

dataset = Amazon(root="./data/Amazon", name="Photo")
data = dataset[0]


# 2)use a GCN to extract embeddings for each node
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = data.to(device)
embed_dim = 128


# improved loss, uses the loss used in graphsage
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    sim = torch.mm(z1, z2.t()) / temperature
    pos_sim = torch.exp(torch.diag(sim))
    neg_sim = torch.sum(torch.exp(sim), dim=1)
    loss = -torch.log(pos_sim / neg_sim)
    return loss.mean()


# will be used later
# def community_loss(embeddings, labels):
#     loss = 0.0
#     num_comms = labels.max().item() + 1
#     for comm in range(num_comms):
#         idx = (labels == comm).nonzero(as_tuple=True)[0]
#         if idx.size(0) <= 1:
#             continue
#         comm_embeds = embeddings[idx]
#         sim = F.cosine_similarity(comm_embeds.unsqueeze(1), comm_embeds.unsqueeze(0), dim=2)
#         loss += 1 - sim.mean()  # higher similarity = lower loss
#     return loss / num_comms

# import community.community_louvain as community_louvain  # pip install python-louvain
from torch_geometric.utils import dropout_adj, to_networkx

# will be used later
# G_nx = to_networkx(data, to_undirected=True)
# partition = community_louvain.best_partition(G_nx)
# # This returns a dict: node_id → community_id

# # Convert to tensor aligned with data.x order
# community_labels = torch.tensor([partition[i] for i in range(data.num_nodes)], device=device)

# import torch.optim as optim


model = GCN(in_feats=data.num_features, hidden_dim=128, out_dim=embed_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

alpha = 0.5
model.train()
for epoch in range(201):
    optimizer.zero_grad()
    edge_index_1, _ = dropout_adj(data.edge_index, p=0.3, force_undirected=True)
    edge_index_2, _ = dropout_adj(data.edge_index, p=0.4, force_undirected=True)

    z1 = model(data.x, edge_index_1)
    z2 = model(data.x, edge_index_2)

    loss = contrastive_loss(z1, z2)  # + alpha * community_loss(z1, community_labels)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# 3)cluster nodes in embedding space, using k means in this case
import math

model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)
    embeddings_np = embeddings.cpu().numpy()

# 4)linearize the node order based on proximity in that space, using UMAP in this case
import umap

# embeddings_np is your [num_nodes, embedding_dim] array from the GNN
reducer = umap.UMAP(n_components=1, random_state=42)
embeddings_1d = reducer.fit_transform(embeddings_np).squeeze()

# sort the nodes based on their 1D embedding
node_ids = list(range(len(embeddings_1d)))
sorted_nodes = [x for _, x in sorted(zip(embeddings_1d, node_ids))]


# 5) Use a learned index (like an RMI) to predict node positions on this new 1D layout
# so now that we have the linear order of the nodes of the graph, we can now train a learned index to predict the
# position of a node in this new sorted 1d layout

from sklearn.linear_model import LinearRegression

# assume: X = embeddings_1d (N, 1), y = node positions in sorted list
node_id_to_index = {node_id: i for i, node_id in enumerate(sorted_nodes)}
X = embeddings_1d.reshape(-1, 1)
y = np.array([node_id_to_index[i] for i in range(len(X))])

# root model (layer 0)
root_model = LinearRegression()
root_model.fit(X, y / len(y))  # normalize to [0, 1]

# layer 1 (leaf models)
num_leaf_models = 100
leaf_models = [None for _ in range(num_leaf_models)]
leaf_buckets = [[] for _ in range(num_leaf_models)]

# assign points to leaf buckets using root model
for i in range(len(X)):
    pred_norm = root_model.predict(X[i : i + 1])[0]
    bucket_id = int(pred_norm * num_leaf_models)
    bucket_id = max(0, min(bucket_id, num_leaf_models - 1))
    leaf_buckets[bucket_id].append((X[i], y[i]))

# train each leaf model
for i, bucket in enumerate(leaf_buckets):
    if not bucket:
        continue
    Xb, yb = zip(*bucket)
    Xb = np.array(Xb)
    yb = np.array(yb)
    model = LinearRegression().fit(Xb, yb)
    leaf_models[i] = model


# predict with this RMI heirarcy with linear models
def rmi_predict(x_input_1d):
    x_input_1d = np.array(x_input_1d).reshape(1, -1)  # shape: [1, 1]
    root_pred = root_model.predict(x_input_1d)[0]
    bucket_id = int(root_pred * num_leaf_models)
    bucket_id = max(0, min(bucket_id, num_leaf_models - 1))

    model = leaf_models[bucket_id]
    if model is None:
        return None
    return model.predict(x_input_1d)[0]


total_error = 0
errors = []
for i in range(len(X)):
    pred = rmi_predict(X[i])
    real = y[i]
    error = abs(pred - real)
    errors.append(error)
    print(f"real: {real}, pred: {pred}, error: {error}")
    total_error += error

print(f"Mean Absolute Error: {total_error / len(X):.4f}")

max_search_error = max(errors)
print(max_search_error)
