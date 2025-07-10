import vf3py
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import random

def nx_is_monomorphic(subg, g):
    gm = GraphMatcher(g, subg)
    return gm.subgraph_is_monomorphic()

# Hardcoded sanity check test case.
def main():
    for num_nodes in range(10, 20, 5):
        for ita in range(2, 5):
            num_edges = int(num_nodes * num_nodes * ita / 10.0)
            print(f"Testing with {num_nodes} nodes and {num_edges} edges.")

            g1 = nx.gnm_random_graph(num_nodes, num_edges)
            g2 = nx.Graph()

            for e in nx.edges(g1):
                if random.random() > 0.5:
                    g2.add_edge(e[0], e[1])
            
            print(f"G1: {len(g1.nodes())}-{len(g1.edges())}, G2: {len(g2.nodes())}-{len(g2.edges())}")
            for variant in ('B', 'P', 'L'):
                print(f"Running VF3{variant}...")
                if not vf3py.has_monomorphic_subgraph(g2, g1, variant=variant,
                                                      num_threads=vf3py.MAX_THREADS if variant == 'P' else 1):
                    # VF3 Wrong?
                    print("Running nx.VF2...")
                    assert not nx_is_monomorphic(g2, g1)
                    

    print("Random monomorphism test complete.")

if __name__ == "__main__":
    main()