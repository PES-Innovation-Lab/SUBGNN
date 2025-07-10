import glob
import ntpath
import sys
import os

import vf3py

import networkx as nx
from networkx.algorithms import isomorphism


SDF_TEMPLATE = os.path.join(os.path.dirname(__file__), './mols/*.sdf')
MOL_FILES = {
    ntpath.basename(file.replace('.sdf', '')): file
    for file in glob.glob(SDF_TEMPLATE)
}


def sdf_to_graph(sdf_fname: str) -> nx.Graph:
    molgraph = nx.Graph()

    with open(sdf_fname, "r") as f:
        lines = f.readlines()
    natoms = int(lines[3][0:3])
    nbonds = int(lines[3][3:6])
    for i in range(4, 4 + natoms):
        molgraph.add_node(i-4)
        parts = lines[i].replace("\n", "").split()
        molgraph.nodes[i-4]['symbol'] = parts[3]
    for i in range(4 + natoms, 4 + natoms + nbonds):
        at1 = int(lines[i][0:3])
        at2 = int(lines[i][3:7])
        bondtype = int(lines[i][7:10])
        molgraph.add_edge(at1 - 1, at2 - 1)
        molgraph[at1 - 1][at2 - 1]['type'] = bondtype
    return molgraph


def same_element(n1_attrib: dict, n2_attrib: dict) -> bool:
    return n1_attrib['symbol'] == n2_attrib['symbol']


def get_hcarbon_subgraph(graph) -> nx.Graph:
    save_atoms = []
    for node in graph.nodes:
        if graph.nodes[node]['symbol'] == 'H':
            nb_list = list(graph.neighbors(node))
            if len(nb_list) == 1 and graph.nodes[nb_list[0]]['symbol'] == 'C':
                continue
        save_atoms.append(node)
    subgraph = graph.subgraph(save_atoms)

    return subgraph


def generate_isomorphisms(graph):
    GM = isomorphism.GraphMatcher(graph, graph, node_match=same_element)
    isomorphisms = set()
    atoms = list(graph.nodes)
    for isom in GM.isomorphisms_iter():
        isomorphisms.add(tuple(isom[i] for i in atoms))
    return isomorphisms


def main(verbose=False):
    for molname, molfile in MOL_FILES.items():
        # print(f"Processing test molecule '{molname}'")
        mol_graph = sdf_to_graph(molfile)
        subgraph = get_hcarbon_subgraph(mol_graph)
        reference_answer = generate_isomorphisms(subgraph)

        atoms = list(subgraph.nodes)
        for variant in ('B', 'P', 'L'):
            result = set(
                tuple(
                    isom[i]
                    for i in atoms
                )
                for isom in vf3py.get_automorphisms(subgraph, node_match=same_element,
                                                    verbose=verbose, variant=variant,
                                                    num_threads=vf3py.MAX_THREADS if variant == 'P' else 1)
            )
            assert result == reference_answer, \
                f"Failed for test graph '{molname}': correct={repr(reference_answer)},\ngot={repr(result)}"
    print("Molecular isomorphisms test passed")


if __name__ == "__main__":
    VERBOSE = '-v' in sys.argv
    main(verbose=VERBOSE)
