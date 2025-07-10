Usage
===========

VF3Py cheat sheet
-----------------

.. list-table::
   :widths: 40 25 30
   :header-rows: 1
   :stub-columns: 1

   * - 
     - Check
     - Enumerate
   * - Graph → Graph (same size)
     - :py:meth:`vf3py.are_isomorphic`
     - :py:meth:`vf3py.get_exact_isomorphisms`
   * - Automorphisms (onto itself)
     - —
     - :py:meth:`vf3py.get_automorphisms`
   * - Subgraph isomorphism
     - :py:meth:`vf3py.has_subgraph`
     - :py:meth:`vf3py.get_subgraph_isomorphisms`
   * - Subgraph monomorphism
     - :py:meth:`vf3py.has_monomorphic_subgraph`
     - :py:meth:`vf3py.get_subgraph_monomorphisms`


Here, *Check* refers to giving a binary answer ``True`` / ``False`` - graphs are isomorphic or not
(search terminates as soon as first solution is found, this solution can be accessed with ``get_mapping`` kwarg).
Conversely, *Enumerate* implies construction of exhaustive list of all solutions.

There is a difference between subgraph mono- and isomorphisms. Isomorphism ``G -> H`` is a stronger condition requiring pairs of nodes in ``H`` to be connected iff their images are connected in ``G``. In case of monomorphism, some edges of ``G`` may be missing in ``H``. For example, there is monomorphism from graph ``A B-C`` to ``A-B-C``, but there is no isomorphism.

Basic usage example:

.. code-block:: python

    >>> import vf3py
    >>> import networkx as nx
    >>> vf3py.are_isomorphic(nx.complete_graph(3), nx.cycle_graph(3))
    True
    >>> vf3py.are_isomorphic(nx.complete_graph(4), nx.cycle_graph(4))
    False
    >>> vf3py.get_subgraph_isomorphisms(subgraph=nx.path_graph(3), graph=nx.cycle_graph(4))
    [{0: 1, 1: 0, 2: 3}, {0: 3, 1: 0, 2: 1}, {0: 0, 1: 1, 2: 2}, {0: 2, 1: 1, 2: 0},
     {0: 1, 1: 2, 2: 3}, {0: 3, 1: 2, 2: 1}, {0: 0, 1: 3, 2: 2}, {0: 2, 1: 3, 2: 0}]
    >>> vf3py.get_subgraph_isomorphisms(subgraph=nx.path_graph(4), graph=nx.cycle_graph(4))
    [] # <- i.e. no isomorphisms


Supported types of graphs
-------------------------

VF3Py supports `nx.Graph` and `nx.DiGraph`, and does not support `nx.MultiGraph` and `nx.MultiDiGraph` (for details see `manual <https://networkx.org/documentation/stable/reference/classes/index.html>`_):

+----------------+------------+--------------------+------------------------+---------------------+
| NetworkX Class | Type       | Self-loops allowed | Parallel edges allowed | Supported by VF3Py? |
+================+============+====================+========================+=====================+
| Graph          | undirected | Yes                | No                     | **Yes**             |
+----------------+------------+--------------------+------------------------+---------------------+
| DiGraph        | directed   | Yes                | No                     | **Yes**             |
+----------------+------------+--------------------+------------------------+---------------------+
| MultiGraph     | undirected | Yes                | Yes                    | **No**              |
+----------------+------------+--------------------+------------------------+---------------------+
| MultiDiGraph   | directed   | Yes                | Yes                    | **No**              |
+----------------+------------+--------------------+------------------------+---------------------+

In other words, VF3Py is able to treat any undirected and directed graphs if nodes are allowed to be connected by one edge at most (i.e. no parallel edges).

In practice, graph objects are simply passed into any VF3Py function, and the appropriate treatment is automatically performed depending on whether it's a ``nx.Graph`` and ``nx.DiGraph``. If either ``nx.MultiGraph`` or ``nx.MultiDiGraph`` is passed to VF3Py, then an exception will be thrown:

.. code-block:: python

    import vf3py
    import networkx as nx

    two_path = nx.DiGraph()
    two_path.add_edges_from([['A', 'B'], ['B', 'C']])

    square_vortex = nx.DiGraph()
    square_vortex.add_edges_from([[0, 1], [1, 2], [2, 3], [3, 0]])

    # Automatically figures out when dealing with DiGraphs
    vf3py.get_subgraph_isomorphisms(
        subgraph=two_path, # Directed
        graph=square_vortex # Directed
    )
    # [{'A': 3, 'B': 0, 'C': 1}, {'A': 0, 'B': 1, 'C': 2},
    #  {'A': 1, 'B': 2, 'C': 3}, {'A': 2, 'B': 3, 'C': 0}]
    
    # Automatically figures out when dealing with undirected nx.Graphs => results in more isomorphisms
    vf3py.get_subgraph_isomorphisms(
        subgraph=two_path.to_undirected(), # Undirected
        graph=square_vortex.to_undirected() # Undirected
    )
    # [{'A': 1, 'B': 0, 'C': 3}, {'A': 3, 'B': 0, 'C': 1}, {'A': 0, 'B': 1, 'C': 2}, {'A': 2, 'B': 1, 'C': 0},
    #  {'A': 1, 'B': 2, 'C': 3}, {'A': 3, 'B': 2, 'C': 1}, {'A': 0, 'B': 3, 'C': 2}, {'A': 2, 'B': 3, 'C': 0}]

    vf3py.get_subgraph_isomorphisms(
        subgraph=two_path, # Directed
        graph=square_vortex.to_undirected() # Undirected
    )
    # AssertionError: Both graphs must be either directed or undirected

    test_multigraph = nx.MultiGraph()
    edges = [('A', 'B'), ('A', 'B'), ('A', 'B')]
    test_multigraph.add_edges_from(edges)

    vf3py.get_automorphisms(test_multigraph)
    # vf3py.ApplicabilityScopeError: Cannot accept Multigraph type for isomorphism calculations


Node & edge attributes
----------------------

In many practical tasks, we are interested in only a subset of graph isomorphisms - the ones that match nodes and/or edges of the same "color" (i.e. attribute value). In NetworkX, such constrained graph isomorphism calculations can be done by adding attributes to nodes and/or edges and, then, passing functions ``node_match`` and/or ``edge_match`` as optional keyword-arguments. ``VF3Py`` uses the same approach:

.. code-block:: python

    import vf3py
    import networkx as nx

    three_path = nx.Graph()
    three_path.add_edges_from([['A', 'B'], ['B', 'C']])
    three_path.add_nodes_from(['A', 'C'], color='red')
    three_path.add_nodes_from(['B'], color='green')

    square_graph = nx.Graph()
    square_graph.add_edges_from([[0, 1], [1, 2], [2, 3], [3, 0]])
    square_graph.add_nodes_from([0, 2], color='red')
    square_graph.add_nodes_from([1, 3], color='green')

    vf3py.get_subgraph_isomorphisms(
        subgraph=three_path,
        graph=square_graph,
        node_match=lambda subgraph_dict, graph_dict: subgraph_dict['color'] == graph_dict['color']
    )
    # [{'A': 0, 'B': 1, 'C': 2}, {'A': 2, 'B': 1, 'C': 0},
    #  {'A': 0, 'B': 3, 'C': 2}, {'A': 2, 'B': 3, 'C': 0}]
    # Note, that green node B of subgraph is matched only with green 1 and 3 nodes of the main graph

However, ``VF3lib`` does not work with functions and, instead, takes concrete values of attributes (integers representing "colors") of each node and edge. Thus, VF3Py under the hood uses ``node_match`` / ``edge_match`` functions to compute equivalent "coloring" of nodes and/or edges - this is not always possible. For VF3Py user this means that ``node_match`` and/or ``edge_match`` functions (if they are provided) have to correspond to a valid coloring of graph's nodes and/or edges. This restricts the use of VF3Py for complex rules for matching nodes and edges. In particular, prohibited are the rules allowing the same node (or edge) to be matched with multiple 'colors':

.. code-block:: python

    import vf3py
    import networkx as nx

    three_path = nx.Graph()
    three_path.add_edges_from([['A', 'B'], ['B', 'C']])
    three_path.add_nodes_from(['A', 'C'], color='red')
    three_path.add_nodes_from(['B'], color='green')

    square_graph = nx.Graph()
    square_graph.add_edges_from([[0, 1], [1, 2], [2, 3], [3, 0]])
    square_graph.add_nodes_from([0, 2], color='red')
    square_graph.add_nodes_from([1, 3], color='green')

    # Without node labels as separate node attributes,
    # it would be impossible to distinguish 'B' from everything else
    for node in three_path.nodes:
        three_path.nodes[node]['label'] = node

    def node_match(subgraph_dict, graph_dict):
        if subgraph_dict['label'] == 'B':
            return True # Explicitly allow 'B' to be matched with any color
        else:
            return graph_dict['color'] == subgraph_dict['color']

    vf3py.get_subgraph_isomorphisms(
        subgraph=three_path,
        graph=square_graph,
        node_match=node_match
    )
    # vf3py.ApplicabilityScopeError: Unable to create valid node attributes for <function node_match at 0x7f2e72cb44c0>


Limitations
-----------

#. ``nx.MultiGraph`` and ``nx.MultiDiGraph`` are not supported.

#. Complex rules for matching nodes and edges. In particular, can not allow the same node (or edge) to be matched with multiple 'colors'.

On first and second limitation, VF3Py throws a special type of exception ``vf3py.ApplicabilityScopeError`` so that try-except can be used as a fallback to NetworkX isomorphism algorithm.
