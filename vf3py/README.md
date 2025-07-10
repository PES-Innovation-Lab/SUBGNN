# VF3py

VF3py is a Python library for fast and efficient subgraph isomorphism detection, based on the VF3 algorithm. 
We modified VF3 to do regional subgraph matching.

## Building the Library

To build the VF3py library, follow these steps:

1. Open a terminal and navigate to the `release_assemble` directory:
   ```bash
   cd release_assemble
   ```
2. Run the following command to build the library:
   ```bash
   python release_assemble.py
   ```

This will assemble and build the library for use.

## Using VF3py

Use parameter ```python start_target_node```

For Example:
```python
        result = vf3py.get_subgraph_isomorphisms(
            subgraph=pattern,
            graph=target_graph,
            start_target_node_id=3
        )
```