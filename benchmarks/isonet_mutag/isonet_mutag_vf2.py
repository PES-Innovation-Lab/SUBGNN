import pickle
import networkx as nx
from torch_geometric.utils import from_networkx
import time

with open("mutag240k_corpus_subgraphs.pkl", "rb") as f:
    corpus = pickle.load(f)

with open("val_mutag240k_query_subgraphs.pkl", "rb") as f:
    val = pickle.load(f)


for query_no in range(len(val)):
    for corpus_no in range(len(corpus)):
        start_time = time.time()
        M = nx.algorithms.isomorphism.GraphMatcher(corpus[corpus_no], val[query_no])
        is_subgraph = M.subgraph_is_isomorphic()
        end_time = time.time()
        if is_subgraph:
            print(f"Query: {query_no}\tCorpus: {corpus_no}\tElapsed: {int(round((end_time - start_time) * 1000000))}")
