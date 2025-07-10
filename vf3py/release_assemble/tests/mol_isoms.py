import glob, ntpath, sys
from chemscripts import geom

import vf3py
import networkx as nx
from networkx.algorithms import isomorphism


MOL_FILES = {
    ntpath.basename(file.replace('.sdf', '')): file
    for file in glob.glob('./mols/pdb*.sdf')
}


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
        print(f"Processing test molecule '{molname}'")
        mol = geom.Molecule(sdf=molfile)
        subgraph = get_hcarbon_subgraph(mol.G)
        reference_answer = generate_isomorphisms(subgraph)

        result = set(
            tuple(
                i[1]
                for i in isom
            )
            for isom in vf3py.get_automorphisms(subgraph, node_match=same_element, verbose=verbose)
        )
        assert result == reference_answer, \
            f"Failed for test graph '{molname}': correct={repr(reference_answer)},\ngot={repr(result)}"

if __name__ == "__main__":
    VERBOSE = '-v' in sys.argv
    main(verbose=VERBOSE)
