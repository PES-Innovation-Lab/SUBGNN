#!/usr/bin/env python3
"""
Example usage of glasgow_subgraph_solver as a library.
"""

# Import the library
from glasgow_subgraph_solver import (
    GlasgowSubgraphSolver,
    solve_subgraph_isomorphism,
    count_subgraph_isomorphisms,
    find_all_subgraph_isomorphisms,
    has_subgraph_isomorphism,
    get_first_mapping
)

def main():
    print("Glasgow Subgraph Solver - Library Usage Examples")
    print("=" * 50)
    
    # Example 1: Using convenience functions (recommended for most use cases)
    print("\n1. Using convenience functions:")
    
    # Check if a subgraph isomorphism exists
    exists = has_subgraph_isomorphism("test-instances/c3.csv", "test-instances/c3c2.csv")
    print(f"   Subgraph isomorphism exists: {exists}")
    
    # Get the first mapping
    mapping = get_first_mapping("test-instances/c3.csv", "test-instances/c3c2.csv")
    print(f"   First mapping: {mapping}")
    
    # Count all solutions
    count = count_subgraph_isomorphisms("test-instances/c3.csv", "test-instances/c3c2.csv")
    print(f"   Total solutions: {count}")
    
    # Find all mappings (limited to 5)
    all_mappings = find_all_subgraph_isomorphisms(
        "test-instances/c3.csv", 
        "test-instances/c3c2.csv", 
        limit=5
    )
    print(f"   First 5 mappings: {all_mappings}")
    
    # Example 2: Using start_from_vertex parameter
    print("\n2. Using start_from_vertex parameter:")
    
    # Find mapping starting from vertex 0
    mapping_from_0 = get_first_mapping(
        "test-instances/c3.csv", 
        "test-instances/c3c2.csv",
        start_from_vertex=0
    )
    print(f"   Mapping starting from vertex 0: {mapping_from_0}")
    
    # Find mapping starting from vertex 5
    mapping_from_5 = get_first_mapping(
        "test-instances/c3.csv", 
        "test-instances/c3c2.csv",
        start_from_vertex=5
    )
    print(f"   Mapping starting from vertex 5: {mapping_from_5}")
    
    # Example 3: Using the class directly (for advanced usage)
    print("\n3. Using GlasgowSubgraphSolver class directly:")
    
    solver = GlasgowSubgraphSolver()
    
    # Get full result with statistics
    result = solver.solve(
        "test-instances/c3.csv", 
        "test-instances/c3c2.csv",
        start_from_vertex=0,
        timeout=10
    )
    
    print(f"   Success: {result['success']}")
    print(f"   Mappings found: {len(result['mappings'])}")
    print(f"   Solution count: {result['solution_count']}")
    print(f"   Statistics: {result['stats']}")
    
    # Example 4: Advanced options
    print("\n4. Advanced options:")
    
    # Induced subgraph isomorphism
    induced_mapping = get_first_mapping(
        "test-instances/c3.csv", 
        "test-instances/c3c2.csv",
        induced=True,
        start_from_vertex=0
    )
    print(f"   Induced mapping: {induced_mapping}")
    
    # With timeout and parallel processing
    result_advanced = solve_subgraph_isomorphism(
        "test-instances/c3.csv", 
        "test-instances/c3c2.csv",
        timeout=5,
        parallel=True,
        start_from_vertex=0
    )
    print(f"   Advanced result success: {result_advanced['success']}")
    print(f"   Advanced mappings: {result_advanced['mappings']}")
    
    # Example 5: Error handling
    print("\n5. Error handling:")
    
    try:
        # Try with non-existent files
        mapping = get_first_mapping("nonexistent1.csv", "nonexistent2.csv")
        print(f"   Unexpected success: {mapping}")
    except Exception as e:
        print(f"   Expected error handled: {type(e).__name__}: {e}")
    
    # Try with files that might not have solutions
    mapping = get_first_mapping("pattern1", "target1")
    if mapping:
        print(f"   Pattern1->Target1 mapping: {mapping}")
    else:
        print("   No mapping found for pattern1->target1")
    
    print("\n" + "=" * 50)
    print("Library usage examples completed!")

if __name__ == "__main__":
    main()
