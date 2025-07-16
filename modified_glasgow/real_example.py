#!/usr/bin/env python3
"""
Real example usage of the Glasgow Subgraph Solver Python wrapper
using actual test instances from the project.
"""

from glasgow_subgraph_solver import GlasgowSubgraphSolver
import os
import json

def main():
    print("Glasgow Subgraph Solver - Real Example Usage")
    print("=" * 50)
    
    # Initialize the solver
    try:
        solver = GlasgowSubgraphSolver()
        print("✓ Solver initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize solver: {e}")
        return
    
    # Check if test instances exist
    test_instances = [("pattern1", "target1")]
    
    for pattern_file, target_file in test_instances:
        if os.path.exists(pattern_file) and os.path.exists(target_file):
            print(f"\n--- Testing with {pattern_file} -> {target_file} ---")
            
            # Test 1: Basic subgraph isomorphism
            print("1. Basic subgraph isomorphism:")
            result = solver.find_subgraph_isomorphism(pattern_file, target_file)
            print(f"   Success: {result['success']}")
            if result['success']:
                print(f"   Solution found!")
                print(f"   Output: {result['stdout'][:200]}...")
            else:
                print(f"   No solution or error: {result['stderr'][:100]}...")
            
            # Test 2: With start_from_vertex parameter
            print("2. With start_from_vertex=0:")
            result = solver.find_subgraph_isomorphism(
                pattern_file, target_file, 
                start_from_vertex=1032
            )
            print(f"   Success: {result['success']}")
            print(f"   Command used: {result['command']}")
            
            # Test 3: Count solutions
            print("3. Count all solutions:")
            result = solver.count_subgraph_isomorphisms(pattern_file, target_file)
            print(f"   Success: {result['success']}")
            if result['success']:
                print(f"   Output: {result['stdout']}")
            
            # Test 5: With timeout
            print("5. With 5 second timeout:")
            result = solver.solve(
                pattern_file, target_file,
                timeout=5,
                start_from_vertex=1032
            )
            print(f"   Success: {result['success']}")
            print(f"   Command: {result['command']}")
            
            break  # Only test the first valid pair
    else:
        print("No valid test instance pairs found. Creating simple test...")
        
        # Create a simple test case
        pattern_content = """3
0 1
1 2
2 0
"""
        target_content = """4
0 1
1 2
2 3
3 0
0 2
"""
        
        with open("simple_pattern.csv", "w") as f:
            f.write(pattern_content)
        with open("simple_target.csv", "w") as f:
            f.write(target_content)
        
        print("Testing with simple 3-cycle in 4-cycle...")
        result = solver.solve(
            "simple_pattern.csv", "simple_target.csv",
            format="csv",
            start_from_vertex=0
        )
        print(f"Success: {result['success']}")
        print(f"Command: {result['command']}")
        if result['stdout']:
            print(f"Output: {result['stdout']}")
        if result['stderr']:
            print(f"Error: {result['stderr']}")
    
    print("\n" + "=" * 50)
    print("Example completed!")

if __name__ == "__main__":
    main()
