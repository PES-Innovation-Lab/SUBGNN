import sys
import networkx as nx
from networkx.algorithms import isomorphism

import vf3py

def main(verbose=False):
    nx_graph = nx.Graph()
    
    nx_graph.add_edge(1, 2, color="red")
    nx_graph.add_edge(2, 3, color="blue")
    nx_graph.add_edge(3, 4, color="green")
    nx_graph.add_edge(4, 1, color="yellow")
    nx_graph.add_edge(4, 2, color="blue")
    nx_graph.add_edge(3, 1, color="yellow")

    nx_graph.add_nodes_from([3, 4], color='red')
    nx_graph.add_nodes_from([1], color='blue')
    nx_graph.add_nodes_from([2], color='green')
 
    nx_graph_sub = nx.Graph()
    nx_graph_sub.add_edge('A', 'B', color="red")
    nx_graph_sub.add_nodes_from('A', color='blue')
    nx_graph_sub.add_nodes_from('B', color='green')

    def node_match(subgraph_dict, graph_dict):
        return graph_dict['color'] == subgraph_dict['color']
    
    result = vf3py.get_subgraph_isomorphisms(
        subgraph=nx_graph_sub,
        graph=nx_graph,
        node_match=node_match
    )
    assert result == [{'A': 1, 'B': 2}]


if __name__ == "__main__":
    VERBOSE = '-v' in sys.argv
    main(verbose=VERBOSE)
