#!/usr/bin/env python3
"""
Example usage of the Glasgow Subgraph Solver Python wrapper.
"""

from glasgow_subgraph_solver import GlasgowSubgraphSolver
import json

def main():
    # Initialize the solver
    solver = GlasgowSubgraphSolver()
    
    # Example 1: Basic subgraph isomorphism
    print("=== Example 1: Basic subgraph isomorphism ===")
    result = solver.find_subgraph_isomorphism(
        pattern_file="pattern1",
        target_file="target1"
    )
    print(f"Success: {result['success']}")
    if result['success']:
        print("Output:")
        print(result['stdout'])
    else:
        print("Error:")
        print(result['stderr'])
    
    # Example 2: Using start_from_vertex parameter
    print("\n=== Example 2: Using start_from_vertex parameter ===")
    result = solver.find_subgraph_isomorphism(
        pattern_file="pattern1",
        target_file="target1",
        start_from_vertex=0  # Start search from vertex 0 in the target graph
    )
    print(f"Success: {result['success']}")
    if result['success']:
        print("Output:")
        print(result['stdout'])
    else:
        print("Error:")
        print(result['stderr'])
    
    # Example 3: Count all solutions
    print("\n=== Example 3: Count all solutions ===")
    result = solver.count_subgraph_isomorphisms(
        pattern_file="pattern1",
        target_file="target1"
    )
    print(f"Success: {result['success']}")
    if result['success']:
        print("Output:")
        print(result['stdout'])
    
    # Example 4: Find all solutions with start vertex
    print("\n=== Example 4: Find all solutions with start vertex ===")
    result = solver.find_all_subgraph_isomorphisms(
        pattern_file="pattern1",
        target_file="target1",
        start_from_vertex=1,  # Start from vertex 1
        solution_limit=10     # Limit to 10 solutions
    )
    print(f"Success: {result['success']}")
    if result['success']:
        print("Output:")
        print(result['stdout'])
    
    # Example 5: Using advanced options
    print("\n=== Example 5: Advanced options ===")
    result = solver.solve(
        pattern_file="pattern1",
        target_file="target1",
        start_from_vertex=0,
        timeout=30,           # 30 second timeout
        parallel=True,        # Use parallel search
        induced=True,         # Find induced mappings
        value_ordering="degree",  # Use degree-based value ordering
        restarts="luby"       # Use Luby restarts
    )
    print(f"Success: {result['success']}")
    if result['success']:
        print("Output:")
        print(result['stdout'])
    
    # Example 6: Working with CSV format files
    print("\n=== Example 6: CSV format files ===")
    result = solver.solve(
        pattern_file="test-instances/c3.csv",
        target_file="test-instances/c3c2.csv",
        format="csv",
        start_from_vertex=0
    )
    print(f"Success: {result['success']}")
    if result['success']:
        print("Output:")
        print(result['stdout'])
    else:
        print("Error:")
        print(result['stderr'])


if __name__ == "__main__":
    main()
