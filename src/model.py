import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool


class SubgraphEncoder(torch.nn.Module):

    def __init__(self, in_neurons: int, hidden_neurons: int, output_neurons: int):
        super().__init__()

        nn1 = Sequential(
            Linear(in_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_neurons)

        nn2 = Sequential(
            Linear(hidden_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_neurons)

        nn3 = Sequential(
            Linear(hidden_neurons, hidden_neurons),
            ReLU(),
            Linear(hidden_neurons, hidden_neurons),
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_neurons)

        self.lin = Linear(hidden_neurons * 3, output_neurons)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Defines the forward pass of the encoder.

        Args:
            x: Node features.
            edge_index: Graph connectivity.
            batch: Batch vector for pooling.

        Returns:
            A normalized graph-level embedding vector.
        """
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))

        h_final = torch.cat(
            [
                global_mean_pool(h1, batch),
                global_mean_pool(h2, batch),
                global_mean_pool(h3, batch),
            ],
            dim=1,
        )

        return F.normalize(self.lin(h_final), dim=1)


def get_graph_embedding(
    dataset: Data, encoder: SubgraphEncoder, device: torch.device
) -> torch.Tensor:
    """
    Args:
        dataset: The PyTorch Geometric graph data object.
        encoder: The trained SubgraphEncoder model.
        device: The torch device to run inference on.

    Returns:
        The graph's embedding tensor.
    """
    with torch.no_grad():
        batch = torch.zeros(dataset.num_nodes, dtype=torch.long, device=device)
        z = encoder(dataset.x, dataset.edge_index, batch)
    return z
