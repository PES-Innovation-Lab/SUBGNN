.. _installation-label:

Installation
============

Make sure that your OS is Linux and Python is 3.8.* or newer.

VF3Py can be installed using this command:

.. code-block:: bash

    pip install --upgrade vf3py

Test your installation:

.. code-block:: python

    >>> import vf3py.test
    >>> vf3py.test.run_tests()
    (...lots of output...)
    OK
    >>>

If you prefer to check that everything works manually, run a simple test code:

.. code-block:: python

    import vf3py
    import networkx as nx

    source_graph = nx.Graph()
    source_graph.add_edges_from([[0, 1]])
    
    target_graph = nx.Graph()
    target_graph.add_edges_from([['A', 'B']])

    x = vf3py.get_exact_isomorphisms(source_graph, target_graph)
    print(x)
    # The output should be [{0: 'A', 1: 'B'}, {0: 'B', 1: 'A'}]
