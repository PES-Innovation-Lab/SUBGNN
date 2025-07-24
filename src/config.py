import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = "./data/cora-model-jigsaw.pth"
DATA_ROOT = "/tmp/Cora"
QUERY_CSV_PATH = "./query.csv"
COARSE_PARTITION_CSV_PATH = "./partition.csv"
TARGET_CSV_PATH = "./target.csv"
SOLVER_PATH = "./build/glasgow_subgraph_solver"


GIN_HIDDEN_NEURONS = 64
GIN_OUTPUT_NEURONS = 16


# (num_coarse_partitions, num_fine_partitions_per_coarse)
HIERARCHY_LEVELS = (20, 5)


FAISS_TOP_K = 5


SINGLE_PARTITION_QUERY_MIN_NODES = 150
SINGLE_PARTITION_QUERY_MAX_NODES = 200

MULTI_FINE_QUERY_FRAGMENTS = 3
MULTI_FINE_QUERY_MIN_NODES = 150
MULTI_FINE_QUERY_MAX_NODES = 200

MULTI_COARSE_QUERY_MIN_NODES = 150
MULTI_COARSE_QUERY_MAX_NODES = 200
MULTI_COARSE_CONFIGS = [(4, 4), (4, 3), (3, 3), (4, 2), (3, 2), (2, 2)]
