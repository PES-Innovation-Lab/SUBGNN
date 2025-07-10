import vf3py
import networkx as nx

# Hardcoded sanity check test case.
def main():
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])

    g2 = nx.Graph()
    g2.add_edges_from([('A','B'), ('B', 'C')])

    assert not vf3py.has_subgraph(g2, g1)
    assert vf3py.has_monomorphic_subgraph(g2, g1)

    sols = vf3py.get_subgraph_monomorphisms(g2, g1)
    # sols = vf3py.main_vf3_caller(g2, g1, induced=False)
    assert len(sols) == 24

    print("Monomorphism test passed")

if __name__ == "__main__":
    main()

