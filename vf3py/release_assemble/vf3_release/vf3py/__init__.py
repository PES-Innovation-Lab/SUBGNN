import platform
import multiprocessing
from typing import Union, Callable, Tuple, List, Dict, Optional, Literal, Set, Mapping
import networkx as nx

assert platform.system() != "Windows", (
    "Only Linux platform is supported in newer releases")

from .cpppart import cpppart

MAX_THREADS = multiprocessing.cpu_count()


class NetworkxView(Set, Mapping):
    pass


# yapf: disable
VF3_CALLS = {
    # (variant, use_node_attrs, use_edge_attrs) => Appropriate function of the base pybind11 module
    ('B', False, False): cpppart.calc_noattrs,
    ('B', True, False): cpppart.calc_nodeattr,
    ('B', False, True): cpppart.calc_edgeattr,
    ('B', True, True): cpppart.calc_bothattrs,
    ('L', False, False): cpppart.calc_l_noattrs,
    ('L', True, False): cpppart.calc_l_nodeattr,
    ('L', False, True): cpppart.calc_l_edgeattr,
    ('L', True, True): cpppart.calc_l_bothattrs,
    ('P', False, False): cpppart.calc_p_noattrs,
    ('P', True, False): cpppart.calc_p_nodeattr,
    ('P', False, True): cpppart.calc_p_edgeattr,
    ('P', True, True): cpppart.calc_p_bothattrs,
}
# yapf: enable


class ApplicabilityScopeError(Exception):
    """VF3Py has two limitations:

    * ``nx.MultiGraph`` and ``nx.MultiDiGraph`` are not supported.

    * Complex rules for matching nodes and edges. In particular, can not allow the same node (or edge) to be matched with multiple 'colors'.
    
    This exception can be thrown by any of these functions: ``are_isomorphic``, ``get_automorphisms``, ``get_exact_isomorphisms``, ``get_subgraph_isomorphisms``, ``has_subgraph``, ``main_vf3_caller``.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _ensure_graph_correctness(graph: Union[nx.Graph, nx.DiGraph]) -> None:
    if isinstance(graph, nx.MultiGraph) or isinstance(graph, nx.MultiDiGraph):
        raise ApplicabilityScopeError(
            "Cannot accept Multigraph type for isomorphism calculations")

    assert isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph), \
        f"Cannot accept graph of type '{type(graph)}' (nx.Graph or nx.DiGraph is expected)"

    assert graph.number_of_nodes() > 0, \
        "Graph must contain non-zero number of nodes"

    if graph.number_of_edges() == 0:
        raise ApplicabilityScopeError("Cannot accept Graph with no edges")


def _process_graph(
        graph: Union[nx.Graph, nx.DiGraph]) -> Tuple[Dict, Dict, bool]:
    assert isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph), \
        f"Provided '{repr(graph)}' is not a NetworkX graph"

    directed = isinstance(graph, nx.DiGraph)

    int_relabel = [node for node in graph.nodes]

    relabeling = {
        'to_int': {
            node: i
            for i, node in enumerate(int_relabel)
        },
        'from_int': int_relabel
    }

    return {
        'nodes': [i for i in range(graph.number_of_nodes())],
        'edges': [[relabeling['to_int'][vA], relabeling['to_int'][vB]]
                  for vA, vB in graph.edges],

        # Attr lists will be filled later
        'node_attrs': [],
        'edge_attrs': [],
    }, relabeling, directed


def _group_attrs(
    itemviewA: NetworkxView,  # Nodes/edges of the two graphs
    itemviewB: NetworkxView,
    match_function,
    item_type: Literal['node', 'edge'],
    bijective=False,
) -> Tuple[Union[Dict, None], Union[Dict, None]]:

    assert match_function is not None

    dep_graph = nx.Graph()
    a_nodes = []
    i = 0
    for item_name in itemviewA:
        dep_graph.add_node(i, base='A', name=item_name)
        a_nodes.append(i)
        i += 1

    b_nodes = []
    for item_name in itemviewB:
        dep_graph.add_node(i, base='B', name=item_name)
        b_nodes.append(i)
        i += 1

    for a_node in a_nodes:
        for b_node in b_nodes:
            if match_function(itemviewA[dep_graph.nodes[a_node]['name']],
                              itemviewB[dep_graph.nodes[b_node]['name']]):
                dep_graph.add_edge(a_node, b_node)

    a_attrs = {}
    b_attrs = {}
    a_failed = False
    b_failed = False
    for attr_index, component in enumerate(nx.connected_components(dep_graph)):
        comp_subgraph = dep_graph.subgraph(component)
        a_number = sum(1 for node, keys in comp_subgraph.nodes(data=True)
                       if keys['base'] == 'A')
        b_number = sum(1 for node, keys in comp_subgraph.nodes(data=True)
                       if keys['base'] == 'B')

        if a_number == 0 and bijective:
            # Some attrs of target graph may be missing in source if bijection is not required
            a_failed = True
        if b_number == 0:
            # This condition means that there are no appropriate nodes/edges in B for node/edge of A to be mapped to
            b_failed = True

        num_edges = comp_subgraph.number_of_edges()
        if num_edges != a_number * b_number:
            raise ApplicabilityScopeError(
                f"Unable to create valid {item_type} attributes for {repr(match_function)}"
            )

        for node, keys in comp_subgraph.nodes(data=True):
            if keys['base'] == 'A':
                a_attrs[keys['name']] = attr_index
            else:
                b_attrs[keys['name']] = attr_index

    if a_failed:
        a_attrs = None
    if b_failed:
        b_attrs = None
    return a_attrs, b_attrs


def _get_log_function(verbose: bool) -> Callable[[str], None]:
    if verbose:
        return lambda message: print(f"[VF3Py] {message}")
    else:
        return lambda message: None


def main_vf3_caller(
    subgraph: Union[nx.Graph, nx.DiGraph],
    graph: Union[nx.Graph, nx.DiGraph],
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    all_solutions: bool = True,
    return_integers: bool = False,
    verbose: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    induced: bool = True,
    bijective_for_nodes: bool = False,
    start_target_node_id: Optional[int] = None,
) -> List[Dict]:
    """The core routine of the VF3Py library. It solves the subgraph iso/monomorphism problem, i.e. finds ways to match nodes of ``subgraph`` with some/all nodes of ``graph``. Only NetworkX graphs are accepted (either nx.Graph or nx.DiGraph). **NOTE**: It is not practical to call this function directly -- use one of the front-end functions instead: ``are_isomorphic``, ``get_exact_isomorphisms``, ``get_automorphisms``, ``has_subgraph``, ``get_subgraph_isomorphisms``.

    Args:
        subgraph (nx.Graph | nx.DiGraph): Searching for maps *of* this graph.
        graph (nx.Graph | nx.DiGraph): Searching for maps *into* this graph.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        all_solutions (bool, optional): Whether to generate all possible subgraph->graph matches (``True``) or only one (``False``). Defaults to ``True``.
        return_integers (bool, optional): Whether to represent isomorphisms using integers (``True``) or the original labels of NetworkX graphs (``False``). Defaults to ``False``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.
        induced (bool, optional): Whether to search for monomorphisms (``False``) or isomorphisms (``True``). Iso does not allow the target graph to have additional edges that are not present in the source graph. Defaults to ``True``.
        bijective_for_nodes (bool, optional): Whether to search for mono/isomorphisms between entire graphs (``True``) or allow to match with a subgraph of the target graph (``False``). Defaults to ``False``.
        start_target_node_id (int, optional): Starting node ID in the target graph from which to begin the search. Only supported in variant 'B'. Defaults to ``None``.

    Returns:
        List[Dict]: List of subgraph->graph isomorphisms. Each isomorphism is represented by a dict that maps: 'subgraph' labels -> 'graph' labels.
    """

    log = _get_log_function(verbose)

    if (bijective_for_nodes
            and (subgraph.number_of_nodes() != graph.number_of_nodes())):
        log("Initial check indicates the absence of solutions")
        return []

    assert num_threads == 1 or variant == 'P', "Only VF3P variant supports multithreading"
    assert num_threads >= 1 and num_threads <= MAX_THREADS, f"Number of threads must be between 1 and {MAX_THREADS}"
    assert variant in ('B', 'P', 'L'), f"V"
    if variant == 'P':
        assert num_threads > 1, "Parallel version should use more than one thread. Set num_threads accordingly"

    _ensure_graph_correctness(graph)
    _ensure_graph_correctness(subgraph)
    log("Graph correctness checks were passed")

    target_dict, target_labels, target_directed = _process_graph(graph)
    pattern_dict, pattern_labels, pattern_directed = _process_graph(subgraph)
    assert not (target_directed ^ pattern_directed), \
        f"Both graphs must be either directed or undirected"
    directed = target_directed
    log("Graphs were loaded successfully")

    if node_match is not None:
        log("Generating node attributes")
        subgraph_node_attrs, graph_node_attrs = _group_attrs(
            subgraph.nodes,
            graph.nodes,
            node_match,
            'node',
            bijective=bijective_for_nodes)
        if graph_node_attrs is None or subgraph_node_attrs is None:
            log("Initial check indicates the absence of solutions")
            return []

        use_node_attrs = (len(set(graph_node_attrs.values())) > 1) or (len(
            set(subgraph_node_attrs.values())) > 1)
        if use_node_attrs:
            target_dict['node_attrs'] = [
                graph_node_attrs[original_label]
                for original_label in target_labels['from_int']
            ]
            pattern_dict['node_attrs'] = [
                subgraph_node_attrs[original_label]
                for original_label in pattern_labels['from_int']
            ]
            log("Node attributes were generated successfully")
        else:
            log("Use of node attributes is redundant")
    else:
        log("Skipping node attributes generation")
        use_node_attrs = False

    if edge_match is not None:
        log("Generating edge attributes")
        subgraph_edge_attrs, graph_edge_attrs = _group_attrs(
            subgraph.edges,
            graph.edges,
            edge_match,
            'edge',
            bijective=(bijective_for_nodes and induced))
        if graph_edge_attrs is None or subgraph_edge_attrs is None:
            log("Initial check indicates the absence of solutions")
            return []

        use_edge_attrs = (len(set(graph_edge_attrs.values())) > 1) or (len(
            set(subgraph_edge_attrs.values())) > 1)
        log(f"graph_edge_attrs = {repr(graph_edge_attrs)}")
        log(f"subgraph_edge_attrs = {repr(subgraph_edge_attrs)}")
        if use_edge_attrs:
            target_dict['edge_attrs'] = [
                graph_edge_attrs[target_labels['from_int'][vA],
                                 target_labels['from_int'][vB]]
                for vA, vB in target_dict['edges']
            ]
            pattern_dict['edge_attrs'] = [
                subgraph_edge_attrs[pattern_labels['from_int'][vA],
                                    pattern_labels['from_int'][vB]]
                for vA, vB in pattern_dict['edges']
            ]
            log("Edge attributes were generated successfully")
        else:
            log("Use of edge attributes is redundant")
    else:
        log("Skipping edge attributes generation")
        use_edge_attrs = False

    log("Loading finished. Entering C++ part...")
    
    # Handle start_target_node_id parameter
    if start_target_node_id is not None:
        if variant != 'B':
            raise ValueError("start_target_node_id parameter is only supported in variant 'B'")
        
        # Convert original node label to internal integer if needed
        if not return_integers and start_target_node_id in target_labels['to_int']:
            start_target_node_id_int = target_labels['to_int'][start_target_node_id]
        elif return_integers and isinstance(start_target_node_id, int):
            start_target_node_id_int = start_target_node_id
        else:
            # If the node doesn't exist or is invalid, use default behavior
            start_target_node_id_int = 4294967295  # UINT32_MAX
    else:
        start_target_node_id_int = 4294967295  # UINT32_MAX (default)
    
    result = VF3_CALLS[variant, use_node_attrs, use_edge_attrs](
        target=target_dict,
        pattern=pattern_dict,
        directed=directed,
        all_solutions=all_solutions,
        verbose=verbose,
        induced=induced,
        num_threads=num_threads,
        start_target_node_id=start_target_node_id_int,
    )
    log(f"Returned to Python. Found {len(result)} solutions")
    if not return_integers:
        result = [{
            pattern_labels['from_int'][source]:
            target_labels['from_int'][target]
            for source, target in match_data
        } for match_data in result]
        log("Successfully translated solutions to the original node labels")
    return result


# FRONT-END FUNCTIONS
# Exact matches


def are_isomorphic(
    source_graph: Union[nx.Graph, nx.DiGraph],
    target_graph: Union[nx.Graph, nx.DiGraph],
    get_mapping: bool = False,
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    return_integers: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    verbose: bool = False,
) -> Union[bool, Tuple[bool, Union[Dict, None]]]:
    """Check if two graphs are isomorphic. This includes checks for number of nodes - ``are_isomorphic`` always returns False if two graphs have different number of nodes, as opposed to graph-subgraph isomorphisms.

    Args:
        source_graph (nx.Graph | nx.DiGraph): First graph.
        target_graph (nx.Graph | nx.DiGraph): Second graph. Swap between source_graph <-> target_graph changes nothing.
        get_mapping (bool): If true, return a single mapping if such mapping exists.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.

    Returns:
        bool: True if graphs are isomorphic, False - otherwise.
        Also, optionally returns Dict | None. The Dict represents the mapping 'target_graph' labels -> 'source_graph' labels. Returned only if get_mapping set to True.
    """

    isom_list = main_vf3_caller(
        source_graph,
        target_graph,
        bijective_for_nodes=True,
        all_solutions=False,
        node_match=node_match,
        edge_match=edge_match,
        return_integers=return_integers,
        verbose=verbose,
        variant=variant,
        num_threads=num_threads,
    )
    if get_mapping:
        return len(isom_list) > 0, isom_list[0] if len(isom_list) > 0 else None
    else:
        return len(isom_list) > 0


def get_exact_isomorphisms(
    source_graph: Union[nx.Graph, nx.DiGraph],
    target_graph: Union[nx.Graph, nx.DiGraph],
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    return_integers: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    verbose: bool = False,
) -> List[Dict]:
    """Get a list of all isomorphisms between two NetworkX graphs. This includes checks for number of nodes - ``get_exact_isomorphisms`` always returns ``[]`` if two graphs have different number of nodes, as opposed to graph-subgraph isomorphisms.

    Args:
        source_graph (nx.Graph | nx.DiGraph): First graph.
        target_graph (nx.Graph | nx.DiGraph): Second graph. Swap between source_graph <-> target_graph swaps keys with values in the resulting isomorphisms dicts.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        return_integers (bool, optional): Whether to represent isomorphisms using integers (``True``) or the original labels of NetworkX graphs (``False``). Defaults to ``False``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.

    Returns:
        List[Dict]: List of source_graph<->target_graph isomorphisms. Each isomorphism is represented by a dict that maps: 'target_graph' labels -> 'source_graph' labels.
    """
    return main_vf3_caller(
        source_graph,
        target_graph,
        bijective_for_nodes=True,
        all_solutions=True,
        node_match=node_match,
        edge_match=edge_match,
        return_integers=return_integers,
        verbose=verbose,
        variant=variant,
        num_threads=num_threads,
    )


def get_automorphisms(
    graph: Union[nx.Graph, nx.DiGraph],
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    return_integers: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    verbose: bool = False,
) -> List[Dict]:
    """Get isomorphic mappings of NetworkX graph onto itself (automorphism).

    Args:
        graph (nx.Graph | nx.DiGraph): The graph of interest.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        return_integers (bool, optional): Whether to represent isomorphisms using integers (``True``) or the original labels of NetworkX graphs (``False``). Defaults to ``False``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.

    Returns:
        List[Dict]: List of graph<->graph isomorphisms (represented as dicts).
    """
    return main_vf3_caller(
        graph,
        graph,
        bijective_for_nodes=True,
        all_solutions=True,
        node_match=node_match,
        edge_match=edge_match,
        return_integers=return_integers,
        verbose=verbose,
        variant=variant,
        num_threads=num_threads,
    )


# FRONT-END FUNCTIONS
# Subgraph matches


def has_subgraph(
    subgraph: Union[nx.Graph, nx.DiGraph],
    graph: Union[nx.Graph, nx.DiGraph],
    get_mapping: bool = False,
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    return_integers: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    verbose: bool = False,
) -> Union[bool, Tuple[bool, Union[Dict, None]]]:
    """Check if `subgraph` is a subgraph of `graph` using VF3 algorithm.

    Args:
        subgraph (nx.Graph | nx.DiGraph): Searching for maps *of* this graph.
        graph (nx.Graph | nx.DiGraph): Searching for maps *into* this graph.
        get_mapping (bool): If true, return a single mapping if such mapping exists.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        return_integers (bool, optional): Whether to represent isomorphisms using integers (``True``) or the original labels of NetworkX graphs (``False``). Defaults to ``False``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.

    Returns:
        bool: True if `subgraph` is, indeed, a subgraph of `graph`, False - otherwise.
        Also, optionally returns Dict | None: subgraph->graph isomorphism represented by a dict that maps: 'subgraph' labels -> 'graph' labels. Returned only if get_mapping set to True.
    """
    isom_list = main_vf3_caller(
        subgraph,
        graph,
        all_solutions=False,
        node_match=node_match,
        edge_match=edge_match,
        return_integers=return_integers,
        verbose=verbose,
        variant=variant,
        num_threads=num_threads,
    )
    if get_mapping:
        return len(isom_list) > 0, isom_list[0] if len(isom_list) > 0 else None
    else:
        return len(isom_list) > 0


def get_subgraph_isomorphisms(
    subgraph: Union[nx.Graph, nx.DiGraph],
    graph: Union[nx.Graph, nx.DiGraph],
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    return_integers: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    verbose: bool = False,
    start_target_node_id: Optional[int] = None,
) -> List[Dict]:
    """Solve subgraph isomorphism problem, i.e. find ways to match nodes of `subgraph` with some/all nodes of `graph`.

    Args:
        subgraph (nx.Graph | nx.DiGraph): Searching for maps *of* this graph.
        graph (nx.Graph | nx.DiGraph): Searching for maps *into* this graph.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        return_integers (bool, optional): Whether to represent isomorphisms using integers (``True``) or the original labels of NetworkX graphs (``False``). Defaults to ``False``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.
        start_target_node_id (int, optional): Starting node ID in the target graph from which to begin the search. Only supported in variant 'B'. Defaults to ``None``.

    Returns:
        List[Dict]: List of subgraph->graph isomorphisms. Each isomorphism is represented by a dict that maps: 'subgraph' labels -> 'graph' labels.
    """
    return main_vf3_caller(
        subgraph,
        graph,
        all_solutions=True,
        node_match=node_match,
        edge_match=edge_match,
        return_integers=return_integers,
        verbose=verbose,
        variant=variant,
        num_threads=num_threads,
        start_target_node_id=start_target_node_id,
    )


def has_monomorphic_subgraph(
    subgraph: Union[nx.Graph, nx.DiGraph],
    graph: Union[nx.Graph, nx.DiGraph],
    get_mapping: bool = False,
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    return_integers: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    verbose: bool = False,
) -> Union[bool, Tuple[bool, Union[Dict, None]]]:
    """Check if `subgraph` is a subgraph of `graph` using VF3 algorithm.

    Args:
        subgraph (nx.Graph | nx.DiGraph): Searching for maps *of* this graph.
        graph (nx.Graph | nx.DiGraph): Searching for maps *into* this graph.
        get_mapping (bool): If true, return a single mapping if such mapping exists.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        return_integers (bool, optional): Whether to represent isomorphisms using integers (``True``) or the original labels of NetworkX graphs (``False``). Defaults to ``False``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.

    Returns:
        bool: True if `subgraph` is, indeed, a subgraph of `graph`, False - otherwise.
        Also, optionally returns Dict | None: subgraph->graph isomorphism represented by a dict that maps: 'subgraph' labels -> 'graph' labels. Returned only if get_mapping set to True.
    """

    isom_list = main_vf3_caller(
        subgraph,
        graph,
        induced=False,
        all_solutions=False,
        node_match=node_match,
        edge_match=edge_match,
        return_integers=return_integers,
        verbose=verbose,
        variant=variant,
        num_threads=num_threads,
    )
    if get_mapping:
        return len(isom_list) > 0, isom_list[0] if len(isom_list) > 0 else None
    else:
        return len(isom_list) > 0


def get_subgraph_monomorphisms(
    subgraph: Union[nx.Graph, nx.DiGraph],
    graph: Union[nx.Graph, nx.DiGraph],
    node_match: Optional[Callable[[dict, dict], bool]] = None,
    edge_match: Optional[Callable[[dict, dict], bool]] = None,
    return_integers: bool = False,
    variant: Literal['B', 'P', 'L'] = 'B',
    num_threads: int = 1,
    verbose: bool = False,
) -> List[Dict]:
    """Solve subgraph monomorphism problem, i.e. find ways to match nodes of `subgraph` with some/all nodes of `graph`.

    Args:
        subgraph (nx.Graph | nx.DiGraph): Searching for maps *of* this graph.
        graph (nx.Graph | nx.DiGraph): Searching for maps *into* this graph.
        node_match (Callable[[dict, dict], bool], optional): Nodes are allowed to be matched only if this function returns ``True`` for the nodes' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        edge_match (Callable[[dict, dict], bool], optional): Edges are allowed to be matched only if this function returns ``True`` for the edges' attributes. Order of arguments: (1) dict of the source graph/subgraph, (2) target graph. Defaults to ``None``.
        return_integers (bool, optional): Whether to represent isomorphisms using integers (``True``) or the original labels of NetworkX graphs (``False``). Defaults to ``False``.
        variant (str, optional): The VF3 variant to be used. One of ``'B', 'P', 'L'``. Defaults to ``'B'``.
        num_threads (int, optional): Number of threads to be used in parallel variant. Defaults to ``1``.
        verbose (bool, optional): Whether to print info on some intermediate steps. Defaults to ``False``.

    Returns:
        List[Dict]: List of subgraph->graph isomorphisms. Each isomorphism is represented by a dict that maps: 'subgraph' labels -> 'graph' labels.
    """
    return main_vf3_caller(
        subgraph,
        graph,
        induced=False,
        all_solutions=True,
        node_match=node_match,
        edge_match=edge_match,
        return_integers=return_integers,
        verbose=verbose,
        variant=variant,
        num_threads=num_threads,
    )
