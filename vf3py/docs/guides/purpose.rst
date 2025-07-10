Purpose of the project
======================

Graph and subgraph isomorphism problems frequently arise in practical computational tasks. These may involve identifying isomorphic graph pairs or determining a complete set of possible mappings from one graph to another. For more information, see `Wikipedia <https://en.wikipedia.org/wiki/Graph_isomorphism_problem>`_.

To solve such tasks in Python, a popular NetworkX library implements ``GraphMatcher`` class:

.. code-block:: python

    import networkx as nx
    from networkx.algorithms import isomorphism

    # Create some simple graph
    G = nx.cycle_graph(4)

    # The GraphMatcher object implements graph isomorphism algorithm
    GM = isomorphism.GraphMatcher(G, G)

    # Iterate over all possible G automorphisms (isomorphisms onto itself)
    for isom in GM.isomorphisms_iter():
        # Print each mapping as {'old node': 'new node'} dict
        print(repr(isom))

    # Result is 4*2=8 square symmetries:
    # {0: 0, 1: 1, 2: 2, 3: 3}
    # {0: 0, 3: 1, 2: 2, 1: 3}
    # {1: 0, 0: 1, 3: 2, 2: 3}
    # {1: 0, 2: 1, 3: 2, 0: 3}
    # {2: 0, 1: 1, 0: 2, 3: 3}
    # {2: 0, 3: 1, 0: 2, 1: 3}
    # {3: 0, 0: 1, 1: 2, 2: 3}
    # {3: 0, 2: 1, 1: 2, 0: 3}


Unfortunately, current graph isomorphism algorithms suffer from scalability issues with larger graphs. Moreover, NetworkX is implemented entirely in Python giving it a slowdown of at least x10.

To solve this problem, there exists a VF3 algorithm for graph and subgraph isomorphism calculation, which extremely efficient in time and memory. Unlike NetworkX, `VF3lib <https://github.com/MiviaLab/vf3lib>`_ (the implementation of VF3) is made completely in C++, which is itself a huge boost. However, the original VF3lib is inconvenient to work with from larger projects, since it requires file-based input and returns results through the command line.

The ``VF3Py`` library makes transition from NetworkX's ``GraphMatcher`` to a more efficient VF3 as easy a possible. All graphs are still constructed in NetworkX, but the calls to ``GraphMatcher`` are seamlessly replaced with calls to ``VF3Py``:

.. code-block:: python

    import networkx as nx
    import vf3py

    # Create some simple graph
    G = nx.cycle_graph(4)

    # The GraphMatcher object implements graph isomorphism algorithm
    automorphisms = vf3py.get_automorphisms(G)

    # Iterate over all possible G automorphisms (isomorphisms onto itself)
    for isom in automorphisms:
        print(repr(isom))

    # This results in the same 8 square symmetries ('old label' -> 'new label'):
    # {0: 0, 1: 1, 2: 2, 3: 3}
    # {0: 0, 1: 3, 2: 2, 3: 1}
    # {0: 1, 1: 0, 2: 3, 3: 2}
    # {0: 1, 1: 2, 2: 3, 3: 0}
    # {0: 2, 1: 1, 2: 0, 3: 3}
    # {0: 2, 1: 3, 2: 0, 3: 1}
    # {0: 3, 1: 0, 2: 1, 3: 2}
    # {0: 3, 1: 2, 2: 1, 3: 0}
