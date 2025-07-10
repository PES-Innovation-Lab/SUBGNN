#!/usr/bin/env python3

import networkx as nx
import vf3py

def simple_test():
    """Simple test with a basic graph structure."""
    
    # Create a simple target graph: triangle with one extra node
    target_graph = nx.Graph()
    target_graph.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3)])
    # Graph: 0-1-2-0 (triangle) with node 3 attached to node 1
    
    # Create a simple pattern: edge
    pattern = nx.Graph()
    pattern.add_edges_from([(0, 1),(1,2)])
    # Pattern: 0-1 (simple edge)
    
    print("Simple test with NetworkX graphs")
    print("Target: 0-1-2-0 (triangle) with node 3 attached to 1")
    print("Pattern: 0-1-2 (edge)")
    print()
    
    # Test without start_target_node_id
    print("1. Without start_target_node_id:")
    try:
        result_default = vf3py.get_subgraph_isomorphisms(
            subgraph=pattern,
            graph=target_graph
        )
        print(f"   Found {len(result_default)} matches:")
        for i, match in enumerate(result_default):
            print(f"   Match {i+1}: {match}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Test with start_target_node_id = 2
    print("2. With start_target_node_id = 3:")
    try:
        result_start_2 = vf3py.get_subgraph_isomorphisms(
            subgraph=pattern,
            graph=target_graph,
            start_target_node_id=3
        )
        print(f"   Found {len(result_start_2)} matches:")
        for i, match in enumerate(result_start_2):
            print(f"   Match {i+1}: {match}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Test with start_target_node_id = 1
    print("3. With start_target_node_id = 1:")
    try:
        result_start_1 = vf3py.get_subgraph_isomorphisms(
            subgraph=pattern,
            graph=target_graph,
            start_target_node_id=1
        )
        print(f"   Found {len(result_start_1)} matches:")
        for i, match in enumerate(result_start_1):
            print(f"   Match {i+1}: {match}")
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    simple_test()
