The Glasgow Subgraph Solver
===========================

This is a solver for subgraph isomorphism (induced and non-induced) problems, based upon a series of
papers by subsets of Blair Archibald, Ciaran McCreesh, Patrick Prosser and James Trimble at the
University of Glasgow, and Fraser Dunlop and Ruth Hoffmann at the University of St Andrews. A clique
decision / maximum clique solver is also included.

If you use this software for research, please cite [icgt/McCreeshP020]. If you use this solver in a
non-research setting, please get in touch if you can. This software is an output of taxpayer funded
research, and it is very helpful for us if we can demonstrate real-world impact when we write grant
applications.

Please contact [Ciaran McCreesh](mailto:ciaran.mccreesh@glasgow.ac.uk) with any queries.

Compiling
---------

To build, you will need a C++20 compiler, such as GCC 10.3, as well as Boost (use
``libboost-all-dev`` on Ubuntu).

```shell
cmake -S . -B build
cmake --build build
```

Execution
---------

To execute the test suite, which evaluates 75 subgraphs against 800 target graphs, run the following command:

```shell
./run-tests.bash
```

This script will perform comprehensive benchmarking across the provided graph datasets and store the output in the result directory.