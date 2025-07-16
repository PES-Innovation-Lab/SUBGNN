#!/usr/bin/env python3
"""
Python wrapper for the Glasgow Subgraph Solver.

This module provides a convenient Python interface to the Glasgow Subgraph Solver,
allowing you to find subgraph isomorphisms between pattern and target graphs.
"""

import subprocess
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import json
import re


class GlasgowSubgraphSolver:
    """
    Python wrapper for the Glasgow Subgraph Solver.
    
    This class provides a convenient interface to run the Glasgow Subgraph Solver
    with various options and parameters.
    """
    
    def __init__(self, solver_path: Optional[str] = None):
        """
        Initialize the Glasgow Subgraph Solver wrapper.
        
        Args:
            solver_path: Path to the glasgow_subgraph_solver executable.
                        If None, will look for it in common locations.
        """
        if solver_path is None:
            # Try to find the solver in common locations
            possible_paths = [
                "./build/glasgow_subgraph_solver",
                "./glasgow_subgraph_solver",
                "glasgow_subgraph_solver",
                str(Path(__file__).parent / "build" / "glasgow_subgraph_solver")
            ]
            
            for path in possible_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    solver_path = path
                    break
            
            if solver_path is None:
                raise FileNotFoundError(
                    "Could not find glasgow_subgraph_solver executable. "
                    "Please specify the path explicitly."
                )
        
        self.solver_path = solver_path
        
        # Verify the solver exists and is executable
        if not os.path.isfile(self.solver_path):
            raise FileNotFoundError(f"Solver not found at: {self.solver_path}")
        if not os.access(self.solver_path, os.X_OK):
            raise PermissionError(f"Solver is not executable: {self.solver_path}")
    
    def _parse_solution_mapping(self, output: str) -> List[Dict[str, str]]:
        """
        Parse solution mappings from solver output.
        
        Args:
            output: The stdout from the solver
            
        Returns:
            List of dictionaries where each dict maps pattern vertex -> target vertex
        """
        mappings = []
        
        # Look for solution lines in the output
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('mapping = '):
                # Extract the mapping part
                mapping_str = line.replace('mapping = ', '')
                mapping = {}
                
                # Parse mappings like "(a1 -> DEF) (a2 -> F2) (a3 -> F3)"
                # Use regex to find all (pattern -> target) pairs
                import re
                pairs = re.findall(r'\(([^)]+)\s*->\s*([^)]+)\)', mapping_str)
                
                for pattern_vertex, target_vertex in pairs:
                    mapping[pattern_vertex.strip()] = target_vertex.strip()
                
                if mapping:
                    mappings.append(mapping)
            elif line.startswith('solution = '):
                # Handle alternative format if it exists
                mapping_str = line.replace('solution = ', '')
                mapping = {}
                
                # Try to parse numeric format "0->1 1->2 2->0"
                pairs = mapping_str.split()
                for pair in pairs:
                    if '->' in pair:
                        try:
                            pattern_vertex, target_vertex = pair.split('->')
                            mapping[pattern_vertex.strip()] = target_vertex.strip()
                        except ValueError:
                            continue
                
                if mapping:
                    mappings.append(mapping)
        
        return mappings
    
    def _parse_solution_count(self, output: str) -> Optional[int]:
        """
        Parse solution count from solver output.
        
        Args:
            output: The stdout from the solver
            
        Returns:
            Number of solutions found, or None if not found
        """
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('solution_count = '):
                try:
                    return int(line.replace('solution_count = ', ''))
                except ValueError:
                    pass
        return None
    
    def _parse_solver_stats(self, output: str) -> Dict[str, Any]:
        """
        Parse solver statistics from output.
        
        Args:
            output: The stdout from the solver
            
        Returns:
            Dictionary containing solver statistics
        """
        stats = {}
        lines = output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ' = ' in line:
                key, value = line.split(' = ', 1)
                
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        stats[key] = float(value)
                    else:
                        stats[key] = int(value)
                except ValueError:
                    # Keep as string if conversion fails
                    stats[key] = value
        
        return stats
    
    def solve(self,
              pattern_file: str,
              target_file: str,
              timeout: Optional[int] = None,
              parallel: bool = False,
              noninjective: bool = False,
              locally_injective: bool = False,
              induced: bool = False,
              count_solutions: bool = False,
              print_all_solutions: bool = False,
              solution_limit: Optional[int] = None,
              format: Optional[str] = None,
              pattern_format: Optional[str] = None,
              target_format: Optional[str] = None,
              restarts: Optional[str] = None,
              geometric_multiplier: Optional[float] = None,
              geometric_constant: Optional[float] = None,
              restart_interval: Optional[int] = None,
              restart_minimum: Optional[int] = None,
              luby_constant: Optional[float] = None,
              value_ordering: Optional[str] = None,
              pattern_symmetries: bool = False,
              target_symmetries: bool = False,
              no_clique_detection: bool = False,
              no_supplementals: bool = False,
              no_nds: bool = False,
              threads: Optional[int] = None,
              triggered_restarts: bool = False,
              delay_thread_creation: bool = False,
              pattern_less_than: Optional[List[str]] = None,
              pattern_automorphism_group_size: Optional[int] = None,
              target_occurs_less_than: Optional[List[str]] = None,
              target_automorphism_group_size: Optional[int] = None,
              send_to_lackey: Optional[str] = None,
              receive_from_lackey: Optional[str] = None,
              send_partials_to_lackey: bool = False,
              propagate_using_lackey: Optional[str] = None,
              start_from_target: Optional[int] = None,
              start_from_vertex: Optional[int] = None,  # Alias for start_from_target
              prove: Optional[str] = None,
              verbose_proofs: bool = False,
              recover_proof_encoding: bool = False) -> Dict[str, Any]:
        """
        Run the Glasgow Subgraph Solver with the specified parameters.
        
        Args:
            pattern_file: Path to the pattern graph file
            target_file: Path to the target graph file
            timeout: Abort after this many seconds
            parallel: Use auto-configured parallel search
            noninjective: Drop the injectivity requirement
            locally_injective: Require only local injectivity
            induced: Find an induced mapping
            count_solutions: Count the number of solutions
            print_all_solutions: Print out every solution
            solution_limit: Stop after finding this many solutions
            format: Specify input file format (auto, lad, vertexlabelledlad, labelledlad, dimacs)
            pattern_format: Specify input file format just for the pattern graph
            target_format: Specify input file format just for the target graph
            restarts: Specify restart policy (luby / geometric / timed / none)
            geometric_multiplier: Specify multiplier for geometric restarts
            geometric_constant: Specify starting constant for geometric restarts
            restart_interval: Specify the restart interval in milliseconds for timed restarts
            restart_minimum: Specify a minimum number of backtracks before a timed restart can trigger
            luby_constant: Specify the starting constant / multiplier for Luby restarts
            value_ordering: Specify value-ordering heuristic (biased / degree / antidegree / random / none)
            pattern_symmetries: Eliminate pattern symmetries (requires Gap)
            target_symmetries: Eliminate target symmetries (requires Gap)
            no_clique_detection: Disable clique / independent set detection
            no_supplementals: Do not use supplemental graphs
            no_nds: Do not use neighbourhood degree sequences
            threads: Use threaded search, with this many threads (0 to auto-detect)
            triggered_restarts: Have one thread trigger restarts
            delay_thread_creation: Do not create threads until after the first restart
            pattern_less_than: Specify pattern less than constraints, in the form v<w
            pattern_automorphism_group_size: Specify the size of the pattern graph automorphism group
            target_occurs_less_than: Specify target occurs less than constraints, in the form v<w
            target_automorphism_group_size: Specify the size of the target graph automorphism group
            send_to_lackey: Send candidate solutions to an external solver over this named pipe
            receive_from_lackey: Receive responses from external solver over this named pipe
            send_partials_to_lackey: Send partial solutions to the lackey
            propagate_using_lackey: Propagate using lackey (never / root / root-and-backjump / always)
            start_from_target: Start search from this target vertex (improves search in nearby locality)
            start_from_vertex: Alias for start_from_target parameter
            prove: Write unsat proofs to this filename (suffixed with .opb and .pbp)
            verbose_proofs: Write lots of comments to the proof, for tracing
            recover_proof_encoding: Recover the proof encoding, to work with verified encoders
            
        Returns:
            A dictionary containing the solver output and metadata
        """
        # Build command line arguments
        cmd = [self.solver_path]
        
        # Program options
        if timeout is not None:
            cmd.extend(["--timeout", str(timeout)])
        if parallel:
            cmd.append("--parallel")
        
        # Problem options
        if noninjective:
            cmd.append("--noninjective")
        if locally_injective:
            cmd.append("--locally-injective")
        if induced:
            cmd.append("--induced")
        if count_solutions:
            cmd.append("--count-solutions")
        if print_all_solutions:
            cmd.append("--print-all-solutions")
        if solution_limit is not None:
            cmd.extend(["--solution-limit", str(solution_limit)])
        
        # Input file options
        if format is not None:
            cmd.extend(["--format", format])
        if pattern_format is not None:
            cmd.extend(["--pattern-format", pattern_format])
        if target_format is not None:
            cmd.extend(["--target-format", target_format])
        
        # Advanced search configuration options
        if restarts is not None:
            cmd.extend(["--restarts", restarts])
        if geometric_multiplier is not None:
            cmd.extend(["--geometric-multiplier", str(geometric_multiplier)])
        if geometric_constant is not None:
            cmd.extend(["--geometric-constant", str(geometric_constant)])
        if restart_interval is not None:
            cmd.extend(["--restart-interval", str(restart_interval)])
        if restart_minimum is not None:
            cmd.extend(["--restart-minimum", str(restart_minimum)])
        if luby_constant is not None:
            cmd.extend(["--luby-constant", str(luby_constant)])
        if value_ordering is not None:
            cmd.extend(["--value-ordering", value_ordering])
        if pattern_symmetries:
            cmd.append("--pattern-symmetries")
        if target_symmetries:
            cmd.append("--target-symmetries")
        
        # Advanced input processing options
        if no_clique_detection:
            cmd.append("--no-clique-detection")
        if no_supplementals:
            cmd.append("--no-supplementals")
        if no_nds:
            cmd.append("--no-nds")
        
        # Advanced parallelism options
        if threads is not None:
            cmd.extend(["--threads", str(threads)])
        if triggered_restarts:
            cmd.append("--triggered-restarts")
        if delay_thread_creation:
            cmd.append("--delay-thread-creation")
        
        # Manual symmetry options
        if pattern_less_than is not None:
            for constraint in pattern_less_than:
                cmd.extend(["--pattern-less-than", constraint])
        if pattern_automorphism_group_size is not None:
            cmd.extend(["--pattern-automorphism-group-size", str(pattern_automorphism_group_size)])
        if target_occurs_less_than is not None:
            for constraint in target_occurs_less_than:
                cmd.extend(["--target-occurs-less-than", constraint])
        if target_automorphism_group_size is not None:
            cmd.extend(["--target-automorphism-group-size", str(target_automorphism_group_size)])
        
        # External constraint solver options
        if send_to_lackey is not None:
            cmd.extend(["--send-to-lackey", send_to_lackey])
        if receive_from_lackey is not None:
            cmd.extend(["--receive-from-lackey", receive_from_lackey])
        if send_partials_to_lackey:
            cmd.append("--send-partials-to-lackey")
        if propagate_using_lackey is not None:
            cmd.extend(["--propagate-using-lackey", propagate_using_lackey])
        
        # Starting vertex options
        # Handle both parameter names (start_from_target and start_from_vertex as alias)
        start_vertex = start_from_target if start_from_target is not None else start_from_vertex
        if start_vertex is not None:
            cmd.extend(["--start-from-target", str(start_vertex)])
        
        # Proof logging options
        if prove is not None:
            cmd.extend(["--prove", prove])
        if verbose_proofs:
            cmd.append("--verbose-proofs")
        if recover_proof_encoding:
            cmd.append("--recover-proof-encoding")
        
        # Add pattern and target files
        cmd.extend([pattern_file, target_file])
        
        # Run the solver
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout if timeout else None
            )
            
            # Parse the output for mappings and statistics
            mappings = self._parse_solution_mapping(result.stdout)
            solution_count = self._parse_solution_count(result.stdout)
            stats = self._parse_solver_stats(result.stdout)
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
                "mappings": mappings,
                "solution_count": solution_count,
                "stats": stats
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Solver timed out",
                "command": " ".join(cmd),
                "mappings": [],
                "solution_count": None,
                "stats": {}
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd),
                "mappings": [],
                "solution_count": None,
                "stats": {}
            }
    
    def find_subgraph_isomorphism(self,
                                  pattern_file: str,
                                  target_file: str,
                                  start_from_vertex: Optional[int] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Simplified interface to find a subgraph isomorphism.
        
        Args:
            pattern_file: Path to the pattern graph file
            target_file: Path to the target graph file
            start_from_vertex: Start search from this target vertex
            **kwargs: Additional parameters passed to solve()
            
        Returns:
            A dictionary containing the solver output and metadata
        """
        return self.solve(
            pattern_file=pattern_file,
            target_file=target_file,
            start_from_vertex=start_from_vertex,
            **kwargs
        )
    
    def count_subgraph_isomorphisms(self,
                                    pattern_file: str,
                                    target_file: str,
                                    start_from_vertex: Optional[int] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Count the number of subgraph isomorphisms.
        
        Args:
            pattern_file: Path to the pattern graph file
            target_file: Path to the target graph file
            start_from_vertex: Start search from this target vertex
            **kwargs: Additional parameters passed to solve()
            
        Returns:
            A dictionary containing the solver output and metadata
        """
        return self.solve(
            pattern_file=pattern_file,
            target_file=target_file,
            count_solutions=True,
            start_from_vertex=start_from_vertex,
            **kwargs
        )
    
    def find_all_subgraph_isomorphisms(self,
                                       pattern_file: str,
                                       target_file: str,
                                       start_from_vertex: Optional[int] = None,
                                       solution_limit: Optional[int] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Find all subgraph isomorphisms.
        
        Args:
            pattern_file: Path to the pattern graph file
            target_file: Path to the target graph file
            start_from_vertex: Start search from this target vertex
            solution_limit: Stop after finding this many solutions
            **kwargs: Additional parameters passed to solve()
            
        Returns:
            A dictionary containing the solver output and metadata
        """
        return self.solve(
            pattern_file=pattern_file,
            target_file=target_file,
            print_all_solutions=True,
            solution_limit=solution_limit,
            start_from_vertex=start_from_vertex,
            **kwargs
        )
    
    def get_first_mapping(self, pattern_file: str, target_file: str, **kwargs) -> Optional[Dict[str, str]]:
        """
        Get the first solution mapping if it exists.
        
        Args:
            pattern_file: Path to the pattern graph file
            target_file: Path to the target graph file
            **kwargs: Additional parameters passed to solve()
            
        Returns:
            Dictionary mapping pattern vertices to target vertices, or None if no solution
        """
        result = self.solve(pattern_file, target_file, **kwargs)
        if result["success"] and result["mappings"]:
            return result["mappings"][0]
        return None
    
    def has_subgraph_isomorphism(self, pattern_file: str, target_file: str, **kwargs) -> bool:
        """
        Check if a subgraph isomorphism exists.
        
        Args:
            pattern_file: Path to the pattern graph file
            target_file: Path to the target graph file
            **kwargs: Additional parameters passed to solve()
            
        Returns:
            True if a subgraph isomorphism exists, False otherwise
        """
        result = self.solve(pattern_file, target_file, **kwargs)
        return result["success"] and (result["mappings"] or result["solution_count"] is not None and result["solution_count"] > 0)
    
    def get_all_mappings(self, pattern_file: str, target_file: str, limit: Optional[int] = None, **kwargs) -> List[Dict[str, str]]:
        """
        Get all solution mappings.
        
        Args:
            pattern_file: Path to the pattern graph file
            target_file: Path to the target graph file
            limit: Maximum number of solutions to return
            **kwargs: Additional parameters passed to solve()
            
        Returns:
            List of dictionaries mapping pattern vertices to target vertices
        """
        kwargs["print_all_solutions"] = True
        if limit is not None:
            kwargs["solution_limit"] = limit
            
        result = self.solve(pattern_file, target_file, **kwargs)
        if result["success"]:
            return result["mappings"]
        return []


# Library-friendly exports
__all__ = [
    'GlasgowSubgraphSolver',
    'solve_subgraph_isomorphism',
    'count_subgraph_isomorphisms', 
    'find_all_subgraph_isomorphisms',
    'has_subgraph_isomorphism',
    'get_first_mapping'
]

# Global solver instance for convenience functions
_global_solver = None

def _get_global_solver():
    """Get or create the global solver instance."""
    global _global_solver
    if _global_solver is None:
        _global_solver = GlasgowSubgraphSolver()
    return _global_solver

def solve_subgraph_isomorphism(pattern_file: str, target_file: str, **kwargs) -> Dict[str, Any]:
    """
    Find a subgraph isomorphism (convenience function).
    
    Args:
        pattern_file: Path to the pattern graph file
        target_file: Path to the target graph file
        **kwargs: Additional parameters (including start_from_vertex)
        
    Returns:
        Dictionary containing solver results with mappings
    """
    return _get_global_solver().solve(pattern_file, target_file, **kwargs)

def count_subgraph_isomorphisms(pattern_file: str, target_file: str, **kwargs) -> int:
    """
    Count the number of subgraph isomorphisms (convenience function).
    
    Args:
        pattern_file: Path to the pattern graph file
        target_file: Path to the target graph file
        **kwargs: Additional parameters (including start_from_vertex)
        
    Returns:
        Number of solutions found
    """
    result = _get_global_solver().count_subgraph_isomorphisms(pattern_file, target_file, **kwargs)
    return result.get("solution_count", 0) or 0

def find_all_subgraph_isomorphisms(pattern_file: str, target_file: str, limit: Optional[int] = None, **kwargs) -> List[Dict[str, str]]:
    """
    Find all subgraph isomorphisms (convenience function).
    
    Args:
        pattern_file: Path to the pattern graph file
        target_file: Path to the target graph file
        limit: Maximum number of solutions to return
        **kwargs: Additional parameters (including start_from_vertex)
        
    Returns:
        List of mappings (pattern vertex -> target vertex)
    """
    return _get_global_solver().get_all_mappings(pattern_file, target_file, limit=limit, **kwargs)

def has_subgraph_isomorphism(pattern_file: str, target_file: str, **kwargs) -> bool:
    """
    Check if a subgraph isomorphism exists (convenience function).
    
    Args:
        pattern_file: Path to the pattern graph file
        target_file: Path to the target graph file
        **kwargs: Additional parameters (including start_from_vertex)
        
    Returns:
        True if a subgraph isomorphism exists
    """
    return _get_global_solver().has_subgraph_isomorphism(pattern_file, target_file, **kwargs)

def get_first_mapping(pattern_file: str, target_file: str, **kwargs) -> Optional[Dict[str, str]]:
    """
    Get the first solution mapping (convenience function).
    
    Args:
        pattern_file: Path to the pattern graph file
        target_file: Path to the target graph file
        **kwargs: Additional parameters (including start_from_vertex)
        
    Returns:
        Dictionary mapping pattern vertices to target vertices, or None
    """
    return _get_global_solver().get_first_mapping(pattern_file, target_file, **kwargs)


def main():
    """
    Command line interface for the Python wrapper.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Python wrapper for the Glasgow Subgraph Solver"
    )
    
    # Required arguments
    parser.add_argument("pattern", help="Pattern graph file")
    parser.add_argument("target", help="Target graph file")
    
    # Optional arguments
    parser.add_argument("--solver-path", help="Path to glasgow_subgraph_solver executable")
    parser.add_argument("--start-from-vertex", type=int, help="Start search from this target vertex")
    parser.add_argument("--timeout", type=int, help="Abort after this many seconds")
    parser.add_argument("--parallel", action="store_true", help="Use auto-configured parallel search")
    parser.add_argument("--induced", action="store_true", help="Find an induced mapping")
    parser.add_argument("--count-solutions", action="store_true", help="Count the number of solutions")
    parser.add_argument("--print-all-solutions", action="store_true", help="Print out every solution")
    parser.add_argument("--solution-limit", type=int, help="Stop after finding this many solutions")
    parser.add_argument("--format", help="Specify input file format")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    try:
        solver = GlasgowSubgraphSolver(args.solver_path)
        
        # Prepare keyword arguments
        kwargs = {}
        if args.start_from_vertex is not None:
            kwargs["start_from_vertex"] = args.start_from_vertex
        if args.timeout is not None:
            kwargs["timeout"] = args.timeout
        if args.parallel:
            kwargs["parallel"] = True
        if args.induced:
            kwargs["induced"] = True
        if args.count_solutions:
            kwargs["count_solutions"] = True
        if args.print_all_solutions:
            kwargs["print_all_solutions"] = True
        if args.solution_limit is not None:
            kwargs["solution_limit"] = args.solution_limit
        if args.format is not None:
            kwargs["format"] = args.format
        
        result = solver.solve(args.pattern, args.target, **kwargs)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["success"]:
                print("SUCCESS")
                print(result["stdout"])
            else:
                print("FAILED")
                print(f"Return code: {result['returncode']}")
                print(f"Error: {result['stderr']}")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
