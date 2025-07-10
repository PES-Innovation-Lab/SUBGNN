#!/usr/bin/env python3

import networkx as nx
import vf3py

def test_complex_graph():
    """Test with a more complex graph structure."""
    
    # Create a target graph: a cycle with additional nodes
    target_graph = nx.Graph()
    target_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 5)])
    # Graph: 0-1-2-3-0 (cycle) with additional nodes 4 and 5
    
    # Create a subgraph: simple triangle
    subgraph = nx.Graph()
    subgraph.add_edges_from([(0, 1), (1, 2)])
    # Pattern: 0-1-2 (path of length 2)
    
    print("Testing get_subgraph_isomorphisms with complex graph...")
    print("Target graph: 0-1-2-3-0 (cycle) with nodes 4, 5 attached")
    print("Subgraph: 0-1-2 (path of length 2)")
    print()
    
    # Test without start_target_node_id
    print("1. Test without start_target_node_id:")
    try:
        result_default = vf3py.get_subgraph_isomorphisms(
            subgraph=subgraph,
            graph=target_graph,
            variant='B'
        )
        print(f"   Found {len(result_default)} matches:")
        for i, match in enumerate(result_default):
            print(f"   Match {i+1}: {match}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Test with start_target_node_id = 2
    print("2. Test with start_target_node_id = 2:")
    try:
        result_start_2 = vf3py.get_subgraph_isomorphisms(
            subgraph=subgraph,
            graph=target_graph,
            start_target_node_id=2
        )
        print(f"   Found {len(result_start_2)} matches:")
        for i, match in enumerate(result_start_2):
            print(f"   Match {i+1}: {match}")
    except Exception as e:
        print(f"   Error: {e}")
    print()


if __name__ == "__main__":
    test_complex_graph()
