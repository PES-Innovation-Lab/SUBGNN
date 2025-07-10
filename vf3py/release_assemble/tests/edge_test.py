import sys
import vf3py
import networkx as nx
from networkx.algorithms import isomorphism


def same_edge(n1_attrib: dict, n2_attrib: dict) -> bool:
    return n1_attrib['type'] == n2_attrib['type']


def main(verbose=False):
    graph = nx.DiGraph()
    graph.add_edges_from([
        ('A',  'C', {'type': 1}),
        ('A',  'B', {'type': 1}),
        ('A',  'B`',{'type': 1}),
        ('B',  'C', {'type': 1}),
        ('B`', 'C', {'type': 2}),
    ])

    if verbose:
        for edge in graph.edges(data=True):
            print(repr(edge))

    GM = isomorphism.GraphMatcher(graph, graph, edge_match=same_edge)
    reference_answer = set(
        item
        for item in GM.isomorphisms_iter()
    )
    
    if verbose:
        print(f"REF = {repr(reference_answer)}")
    result = set(
        item
        for item in vf3py.get_automorphisms(graph, edge_match=same_edge, verbose=verbose)
    )
    assert result == reference_answer, \
        f"Failed for test graph : correct={repr(reference_answer)},\ngot={repr(result)}"
    # raise Exception('jrghk')
    print('Edge test is passed')


if __name__ == "__main__":
    VERBOSE = '-v' in sys.argv
    main(verbose=VERBOSE)
