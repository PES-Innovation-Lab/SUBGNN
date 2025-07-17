#!/usr/bin/env python3
"""
Test script for the Glasgow Subgraph Solver Python wrapper.
"""

import sys
import os
import tempfile
from glasgow_subgraph_solver import GlasgowSubgraphSolver

def create_test_graph(filename, content):
    """Create a test graph file."""
    with open(filename, 'w') as f:
        f.write(content)

def test_basic_functionality():
    """Test basic functionality of the wrapper."""
    print("Testing basic functionality...")
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as pattern_file:
        pattern_file.write("3\n0 1\n1 2\n2 0\n")  # Triangle (C3)
        pattern_path = pattern_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as target_file:
        target_file.write("4\n0 1\n1 2\n2 3\n3 0\n0 2\n")  # Square with diagonal
        target_path = target_file.name
    
    try:
        # Initialize solver
        solver = GlasgowSubgraphSolver()
        
        # Test 1: Basic solve
        print("Test 1: Basic solve")
        result = solver.solve(pattern_path, target_path, format="csv")
        print(f"  Success: {result['success']}")
        print(f"  Command: {result['command']}")
        
        # Test 2: With start_from_vertex
        print("Test 2: With start_from_vertex")
        result = solver.solve(pattern_path, target_path, format="csv", start_from_vertex=0)
        print(f"  Success: {result['success']}")
        print(f"  Command: {result['command']}")
        if "--start-from-target 0" in result['command']:
            print("  ✓ start_from_vertex parameter correctly mapped to --start-from-target")
        else:
            print("  ✗ start_from_vertex parameter not found in command")
        
        # Test 3: Count solutions
        print("Test 3: Count solutions")
        result = solver.count_subgraph_isomorphisms(pattern_path, target_path, format="csv")
        print(f"  Success: {result['success']}")
        print(f"  Command: {result['command']}")
        
        # Test 4: Find all solutions
        print("Test 4: Find all solutions")
        result = solver.find_all_subgraph_isomorphisms(
            pattern_path, target_path, 
            format="csv", 
            start_from_vertex=1,
            solution_limit=5
        )
        print(f"  Success: {result['success']}")
        print(f"  Command: {result['command']}")
        
        print("\nAll tests completed!")
        
    finally:
        # Clean up temporary files
        os.unlink(pattern_path)
        os.unlink(target_path)

def test_command_line_interface():
    """Test the command line interface."""
    print("\nTesting command line interface...")
    
    # Create temporary test files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as pattern_file:
        pattern_file.write("3\n0 1\n1 2\n2 0\n")  # Triangle (C3)
        pattern_path = pattern_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as target_file:
        target_file.write("4\n0 1\n1 2\n2 3\n3 0\n0 2\n")  # Square with diagonal
        target_path = target_file.name
    
    try:
        import subprocess
        
        # Test command line interface
        cmd = [
            sys.executable, "glasgow_subgraph_solver.py",
            "--start-from-vertex", "0",
            "--format", "csv",
            "--json",
            pattern_path,
            target_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Command line test - Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✓ Command line interface works correctly")
            # Parse JSON output
            try:
                import json
                output = json.loads(result.stdout)
                print(f"  Success: {output['success']}")
                print(f"  Command: {output['command']}")
            except:
                print("  JSON output parsing failed")
        else:
            print("✗ Command line interface failed")
            print(f"  Error: {result.stderr}")
            
    finally:
        # Clean up temporary files
        os.unlink(pattern_path)
        os.unlink(target_path)

def main():
    """Run all tests."""
    print("Glasgow Subgraph Solver Python Wrapper Tests")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_command_line_interface()
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
