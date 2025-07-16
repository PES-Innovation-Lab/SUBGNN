"""
Glasgow Subgraph Solver Python Package

This package provides a Python interface to the Glasgow Subgraph Solver,
allowing you to find subgraph isomorphisms between pattern and target graphs.

Basic usage:
    from glasgow_subgraph_solver import get_first_mapping, has_subgraph_isomorphism
    
    # Check if isomorphism exists
    exists = has_subgraph_isomorphism("pattern.csv", "target.csv")
    
    # Get first mapping
    mapping = get_first_mapping("pattern.csv", "target.csv", start_from_vertex=0)
    
    # Mapping is a dict: {pattern_vertex: target_vertex}
    print(f"Pattern vertex 0 maps to target vertex {mapping[0]}")
"""

from .glasgow_subgraph_solver import (
    GlasgowSubgraphSolver,
    solve_subgraph_isomorphism,
    count_subgraph_isomorphisms,
    find_all_subgraph_isomorphisms,
    has_subgraph_isomorphism,
    get_first_mapping
)

__version__ = "1.0.0"
__author__ = "Glasgow Subgraph Solver Python Wrapper"

__all__ = [
    'GlasgowSubgraphSolver',
    'solve_subgraph_isomorphism',
    'count_subgraph_isomorphisms',
    'find_all_subgraph_isomorphisms',
    'has_subgraph_isomorphism',
    'get_first_mapping'
]
