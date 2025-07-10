# VF3Py

Interfacing Python and NetworkX with VF3 – the fastest algorithm for graph/subgraph isomorphism calculation.

## Setup

Make sure that your OS is Linux and Python version is >= 3.8.

VF3Py can be installed using this command:

```
pip install vf3py
```
Test your installation:

```
$ python
>>> import vf3py.test
>>> vf3py.test.run_tests()
(...lots of output...)
OK
>>> 
```

## Documentation

[![pipeline status](https://gitlab.com/knvvv/vf3py/badges/master/pipeline.svg)](https://gitlab.com/knvvv/vf3py/-/commits/master)

Available [here](https://knvvv.gitlab.io/vf3py)


## Other projects that use VF3Py

For now, all projects are mine:

* [algebra_repr](https://gitlab.com/knvvv/algebra_repr) (TODO: come up with a better name) -- uses SageMath to construct finite algebraic structures (groups, rings, fields), then represents them as graphs. Graph isomorphism allows to check if any two algebraic structures are isomorphic or not.

* [PyXYZ](https://gitlab.com/knvvv/pyxyz) -- A Python Library for Molecular Geometry Manipulation. TODO: Use VF3Py to generate automorphisms groups for molecular graphs.


## Links

[Gitlab home page](https://gitlab.com/knvvv/vf3py)

[VF3Py PyPi page](https://pypi.org/project/vf3py)

[Original VF3 implementation](https://github.com/MiviaLab/vf3lib)


## References

1. Challenging the time complexity of exact subgraph isomorphism for huge and dense graphs with VF3 - Carletti V., Foggia P., Saggese A., Vento M. - IEEE transactions on pattern analysis and machine intelligence - 2018
2. Introducing VF3: A new algorithm for subgraph isomorphism - Carletti V., Foggia P., Saggese A., Vento M. - International Workshop on Graph-Based Representations in Pattern Recognition - 2017
3. Comparing performance of graph matching algorithms on huge graphs - Carletti V., Foggia P., Saggese A., Vento M. - Pattern Recognition Letters - 2018
4. A Parallel Algorithm for Subgraph Isomorphism - V. Carletti, P. Foggia, P. Ritrovato, M. Vento, V. Vigilante - International Workshop on Graph-Based Representations in Pattern Recognition - 2019
