#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <algorithm>
#ifndef WIN32
	#include <csignal>
	#include <unistd.h>
	#include <sys/time.h>
	#include <sys/stat.h>
	#include <errno.h>
#else
	#include <Windows.h>
	#include <stdint.h>
#endif

#include "WindowsTime.h"
#include "VFLib.h"
#include "Options.hpp"

namespace py = pybind11;


//----------------------------------------------------------------------------

namespace vflib {

template <typename Node, typename Edge>
class PythonARGLoader: public FastStreamARGLoader<Node, Edge> {
    public:
        PythonARGLoader(
            std::vector<nodeID_t> passed_nodes, std::vector<data_t> passed_node_attrs,
            std::vector<std::pair<nodeID_t, nodeID_t>> passed_edges, std::vector<data_t> passed_edge_attrs,
			bool undirected=false
		);
};

template <typename Node, typename Edge>
PythonARGLoader<Node, Edge>
::PythonARGLoader (
		std::vector<nodeID_t> passed_nodes, std::vector<data_t> passed_node_attrs,
		std::vector<std::pair<nodeID_t, nodeID_t>> passed_edges, std::vector<data_t> passed_edge_attrs,
		bool undirected
	) {
        
	this->last_edge_node = NULL_NODE;

    this->node_count = passed_nodes.size();
    this->nodes.resize(this->node_count);
    this->edges.resize(this->node_count);
    
    uint32_t edge_count;
    nodeID_t i, j, n1, n2;

    // Nodes
    if constexpr(std::is_base_of_v<Empty, Node>) {
        if (passed_node_attrs.size() > 0)
            error("Cannot accept node attrs");
    } else {
        if (passed_nodes.size() != passed_node_attrs.size())
            error("Mismatch of sizes 'passed_nodes' and 'passed_node_attrs'");
    }

    for(i = 0; i < this->node_count; i++) {
        n1 = passed_nodes.at(i);
        if constexpr(!std::is_base_of_v<Empty, Node>) {
            this->nodes[i] = passed_node_attrs[i];
        }
        if (n1 != i)
            error("Cannot add node. Indices must be provided in order 0, ..., N-1 :)", n1);
    }
    
    // Edges
    if constexpr(std::is_base_of_v<Empty, Edge>) {
        if (passed_edge_attrs.size() > 0)
            error("Cannot accept edge attrs");
    } else {
        if (passed_edges.size() != passed_edge_attrs.size())
            error("Mismatch of sizes 'passed_edges' and 'passed_edge_attrs'");
    }
    
    for(i = 0; i < passed_edges.size(); i++) {
        n1 = passed_edges.at(i).first;
        n2 = passed_edges.at(i).second;

        if (n1 >= this->node_count || n2 >= this->node_count || n1 == n2)
            error("Cannot add edge ", n1, n2);
        
        if constexpr(std::is_base_of_v<Empty, Edge>) {
            this->edges[n1][n2];
        } else {
            this->edges[n1][n2] = passed_edge_attrs[i];
        }
        
        if (undirected)
            this->edges[n2][n1] = this->edges[n1][n2];
    }
}

}
//----------------------------------------------------------------------------


using namespace vflib;

template <typename Node, typename Edge>
using argloader_t = ARGLoader<Node, Edge>;

template <typename Node, typename Edge>
py::object vf3_driver(
		Options opt, argloader_t<Node, Edge>* pattloader, argloader_t<Node, Edge>* targloader
	) {
	uint32_t n1, n2;
	std::vector<MatchingSolution> solutions;
	std::vector<uint32_t> class_patt;
	std::vector<uint32_t> class_targ;
	uint32_t classes_count;
	bool preloaded = true;

	// preloaded = false;
	// char* argv[] = {"", "-t", "8", "./test4.grf", "./test4.grf"};
	// int32_t argc = 4;
	// for (int32_t i = 0; i < argc + 1; ++i) {
	// 	std::cout << "Argument " << i << ": " << argv[i] << std::endl;
	// }
	// Options optX;
	// if(!GetOptions(optX, argc, argv))
	// 	exit(-1);
	// std::cout << optX.undirected << std::endl;
	// std::cout << optX.storeSolutions << std::endl;
	// std::cout << optX.all_solutions << std::endl;
	// std::cout << optX.algo << std::endl;
	// std::cout << optX.cpu << std::endl;
	// std::cout << optX.numOfThreads << std::endl;
	// std::cout << optX.lockFree << std::endl;
	// std::cout << optX.ssrHighLimit << std::endl;
	// std::cout << optX.ssrLocalStackLimit << std::endl;
	
	// std::ifstream graphInPat(opt.pattern);
	// std::ifstream graphInTarg(opt.target);
	// argloader_t* pattloader = CreateLoader<Node, Edge>(opt, graphInPat);
	// argloader_t* targloader = CreateLoader<Node, Edge>(opt, graphInTarg);

	ARGraph<Node, Edge> patt_graph(pattloader);
	ARGraph<Node, Edge> targ_graph(targloader);

	n1 = patt_graph.NodeCount();
	n2 = targ_graph.NodeCount();

	MatchingEngine<state_t<Node, Edge>>* me = CreateMatchingEngine<Node, Edge>(opt);

	if(!me)
	{
		exit(-1);
	}

	FastCheck<Node, Node, Edge, Edge> check(&patt_graph, &targ_graph);
	if(check.CheckSubgraphIsomorphism())
	{
		NodeClassifier<Node, Edge> classifier(&targ_graph);
		NodeClassifier<Node, Edge> classifier2(&patt_graph, classifier);
		class_patt = classifier2.GetClasses();
		class_targ = classifier.GetClasses();
		classes_count = classifier.CountClasses();
	}

	me->ResetSolutionCounter();
	if(check.CheckSubgraphIsomorphism())
	{
		VF3NodeSorter<Node, Edge, SubIsoNodeProbability<Node, Edge> > sorter(&targ_graph);
		std::vector<nodeID_t> sorted = sorter.SortNodes(&patt_graph);

#ifdef VF3
		state_t<Node, Edge> s0(&patt_graph, &targ_graph, class_patt.data(), class_targ.data(), classes_count, sorted.data(), opt.induced, (opt.start_target_node_id == UINT32_MAX) ? NULL_NODE : (nodeID_t)opt.start_target_node_id);
#else
		state_t<Node, Edge> s0(&patt_graph, &targ_graph, class_patt.data(), class_targ.data(), classes_count, sorted.data(), opt.induced);
#endif
        if (opt.all_solutions)
		    me->FindAllMatchings(s0);
        else
		    me->FindFirstMatching(s0);
	}

	me->GetSolutions(solutions);
    if (opt.verbose) {
        std::cout << "Solution Found " << solutions.size() << std::endl;
        std::vector<MatchingSolution>::iterator it;
        for(it = solutions.begin(); it != solutions.end(); it++)
            std::cout<< me->SolutionToString(*it) << std::endl;
    }

#ifdef VF3P
	delete (ParallelMatchingEngine<state_t<Node, Edge>>*)me;
#else
	delete me;
#endif

	return py::cast(solutions);
}

template <typename Node, typename Edge>
using python_loader_t = PythonARGLoader<Node, Edge>;

template <typename Node, typename Edge>
python_loader_t<Node, Edge> get_loader_from_dict(py::dict graph_data, Options opt) {
	auto nodes = graph_data["nodes"].cast<std::vector<nodeID_t>>();
    auto node_attr = graph_data["node_attrs"].cast<std::vector<data_t>>();

	auto edges = graph_data["edges"].cast<std::vector<std::pair<nodeID_t, nodeID_t>>>();   
    auto edge_attr = graph_data["edge_attrs"].cast<std::vector<data_t>>();
    
    return python_loader_t<Node, Edge>(
		nodes,
		node_attr,
		edges,
		edge_attr,
		opt.undirected
	);
}

template <typename Node, typename Edge>
py::object calc(
		py::dict pattern, py::dict target,
		const bool directed, const bool all_solutions, const bool verbose, const bool induced, const int num_threads, const uint32_t start_target_node_id
	) {

	Options opt;
	opt.undirected = !directed;
	opt.all_solutions = all_solutions;
	opt.format = "python";
	opt.storeSolutions = true;
	opt.repetitionTimeLimit = 0.0;
    opt.verbose = verbose;
	opt.induced = induced;
#ifdef VF3
	opt.start_target_node_id = start_target_node_id;
#endif
	#ifdef VF3P
	if (num_threads == 1)
		throw py::value_error("Number of threads must be more than 1 for parallel variant of VF3");
    opt.numOfThreads = num_threads;
    opt.algo = VF3PWLS;
	#else
	if (num_threads > 1)
		throw py::value_error("Number of threads must be 1 for this variant of VF3");
	#endif
	
	auto pattloader = get_loader_from_dict<Node, Edge>(pattern, opt);
	auto targloader = get_loader_from_dict<Node, Edge>(target, opt);

	return vf3_driver(opt, &pattloader, &targloader);
}

#define PYBIND_CALC_ARGS py::arg("pattern"), py::arg("target"), py::arg("directed") = false, py::arg("all_solutions") = true, py::arg("verbose") = false, py::arg("induced") = true, py::arg("num_threads") = 1, py::arg("start_target_node_id") = UINT32_MAX

#ifdef VF3
PYBIND11_MODULE(vf3py_base, m) {
	m.def("calc_noattrs",
        &calc<Empty, Empty>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_nodeattr",
        &calc<data_t, Empty>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_edgeattr",
        &calc<Empty, data_t>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_bothattrs",
        &calc<data_t, data_t>,
        PYBIND_CALC_ARGS
    );
}
#endif

#ifdef VF3L
PYBIND11_MODULE(vf3py_vf3l, m) {
	m.def("calc_l_noattrs",
        &calc<Empty, Empty>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_l_nodeattr",
        &calc<data_t, Empty>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_l_edgeattr",
        &calc<Empty, data_t>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_l_bothattrs",
        &calc<data_t, data_t>,
        PYBIND_CALC_ARGS
    );
}
#endif

#ifdef VF3P
PYBIND11_MODULE(vf3py_vf3p, m) {
	m.def("calc_p_noattrs",
        &calc<Empty, Empty>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_p_nodeattr",
        &calc<data_t, Empty>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_p_edgeattr",
        &calc<Empty, data_t>,
        PYBIND_CALC_ARGS
    );
	m.def("calc_p_bothattrs",
        &calc<data_t, data_t>,
        PYBIND_CALC_ARGS
    );
}
#endif