
#README

#VF2 and VF3
============


- mutag_vf2.py and mutag_vf3.py use the MUTAG dataset from TUDatasets

- isonet_mutag_vf2.py and isonet_mutag_vf3.py use the same dataset as ISONET++
    - this is the preferred approach
    - loads the dataset from the pickle files
    - run : python isonet_mutag_vf2.py > bench_vf2.txt
    
The Glasgow Subgraph Solver
===========================

This is a solver for subgraph isomorphism (induced and non-induced) problems, based upon a series of
papers by subsets of Blair Archibald, Ciaran McCreesh, Patrick Prosser and James Trimble at the
University of Glasgow, and Fraser Dunlop and Ruth Hoffmann at the University of St Andrews.

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


RESULT
======
The file result_corpus800_test75.csv contains the result between "mutag240k_corpus_subgraphs.pk" and "test_mutag240k_query_subgraphs.pkl"
