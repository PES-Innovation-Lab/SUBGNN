import sys
import vf3py
import networkx as nx
from networkx.algorithms import isomorphism

TEST_GRAPHS = {
    'gnp_random_graph': lambda: nx.gnp_random_graph(5, 0.3),
    'cycle_graph': lambda: nx.cycle_graph(4),
    'star_graph': lambda: nx.star_graph(5),
    'complete_graph': lambda: nx.complete_graph(4),
    'barbell_graph': lambda: nx.barbell_graph(5, 0),
    'path_graph': lambda: nx.path_graph(4),
    'random_regular_graph': lambda: nx.random_regular_graph(3, 4),
    'watts_strogatz_graph': lambda: nx.watts_strogatz_graph(4, 3, 0.1),
    # 'erdos_renyi_graph': lambda: nx.erdos_renyi_graph(4, 0.2),
}

CRASH_GRAPHS = {
    'scale_free_graph': {
        'get_graph': lambda: nx.scale_free_graph(4),
        'exception': vf3py.ApplicabilityScopeError,
    }
}

def networkx_isomorphisms(G):
    GM = isomorphism.GraphMatcher(G, G)
    isomorphisms = set()
    atoms = list(G.nodes)
    for isom in GM.isomorphisms_iter():
        isomorphisms.add(tuple(isom[i] for i in atoms))
    return isomorphisms


def main(verbose=False):
    for graph_name, get_graph in TEST_GRAPHS.items():
        graph = get_graph()
        print(f"Processing test case '{graph_name}'")
        reference_answer = networkx_isomorphisms(graph)
        atoms = list(graph.nodes)
        if verbose:
            print([node for node in graph.nodes])
            print([edge for edge in graph.edges])
        if graph.number_of_edges() == 0:
            continue
        result = set(
            tuple(
                isom[i]
                for i in atoms
            )
            for isom in vf3py.get_automorphisms(graph, verbose=verbose)
        )
        assert result == reference_answer, \
            f"Failed for test graph '{graph_name}': correct={repr(reference_answer)},\ngot={repr(result)}"
    
    for graph_name, testcase_data in CRASH_GRAPHS.items():
        print(f"Processing test case '{graph_name}'")
        graph = testcase_data['get_graph']()
        expected_exception_type = testcase_data['exception']

        error = None
        try:
            vf3py.get_automorphisms(graph, verbose=verbose)
        except Exception as e:
            error = type(e)
        
        assert error == expected_exception_type, \
            f"Expected to get an error '{expected_exception_type}' but got '{error}'"
    
    print('Basic graph tests passed')


if __name__ == "__main__":
    VERBOSE = '-v' in sys.argv
    main(verbose=VERBOSE)
