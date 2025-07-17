#!/usr/bin/env python3
from glasgow_subgraph_solver import get_first_mapping, GlasgowSubgraphSolver
import time

def main():
    print("Getting Mappings with start_from_vertex")
    print("=" * 40)
    
    pattern_file = "../query/query_part_1.csv"
    target_file = "../target/target_part_1.csv"

    # pattern_file = "pattern1"
    # target_file = "target1"

    print("\nMethod 2: Using GlasgowSubgraphSolver class")
    solver = GlasgowSubgraphSolver()

    test_vertices= [909]

    # test_vertices= [1000,1032,1582]

    for start_vertex in test_vertices:
        
        start_time = time.time()
        result = solver.solve(pattern_file, target_file, start_from_vertex=start_vertex)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if result['success'] and result['mappings']:
            mapping = result['mappings'][0]  # Get first mapping
            print(f"Found mapping starting from vertex {start_vertex}:\n")
            print(f"Mapping: {mapping}\n")
            print(f"Time: {elapsed_time:.4f} seconds")
            print("=" * 40)
        else:
            print(f"No mapping found starting from vertex {start_vertex}\n")
            print(f"Time: {elapsed_time:.4f} seconds")

if __name__ == "__main__":
    main()
