The Glasgow Subgraph Solver
===========================

This is a solver for subgraph isomorphism (induced and non-induced) problems, based upon a series of
papers by subsets of Blair Archibald, Ciaran McCreesh, Patrick Prosser and James Trimble at the
University of Glasgow, and Fraser Dunlop and Ruth Hoffmann at the University of St Andrews. A clique
decision / maximum clique solver is also included.

Steps:
1. Compiling
2. Make Executable

Compiling
---------

The build file is omitted from the repository, so you will need to generate it yourself. The
`CMakeLists.txt` file is set up to build the Glasgow Subgraph Solver as a C++ executable named
`glasgow_subgraph_solver`. It also includes a Python wrapper for convenience

To build, you will need a C++20 compiler, such as GCC 10.3, as well as Boost (use
``libboost-all-dev`` on Ubuntu).

```shell
cmake -S . -B build
cmake --build build
```

# Glasgow Subgraph Solver Python Wrapper

This Python wrapper provides a convenient interface to the Glasgow Subgraph Solver, including full support for the `--start-from-target` parameter (exposed as `start_from_vertex` for convenience) and automatic parsing of solution mappings.

## Installation

No installation required - just ensure the `glasgow_subgraph_solver` executable is built and accessible.
```bash
chmod +x glasgow_subgraph_solver.py
```

### Basic Usage

Check `start_from_vertex_example.py` and `example_usage.py`
