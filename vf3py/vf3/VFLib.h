#ifndef VFLIB_H
#define VFLIB_H

#include "loaders/ARGLoader.hpp"
#include "loaders/FastStreamARGLoader.hpp"
#include "loaders/EdgeStreamARGLoader.hpp"
#include "ARGraph.hpp"
#include "NodeSorter.hpp"
#include "VF3NodeSorter.hpp"
#include "RINodeSorter.hpp"
#include "FastCheck.hpp"
#include "State.hpp"
#include "ProbabilityStrategy.hpp"
#include "NodeClassifier.hpp"
#include "MatchingEngine.hpp"


#ifndef VF3BIO
typedef int32_t data_t;
#else
typedef std::string data_t;
#endif

#ifdef VF3
    #include "VF3SubState.hpp"

    template <typename Node, typename Edge>
    using state_t = vflib::VF3SubState<Node, Node, Edge, Edge>;
#elif defined(VF3L)
    #include "VF3LightSubState.hpp"

    template <typename Node, typename Edge>
    using state_t = vflib::VF3LightSubState<Node, Node, Edge, Edge>;
#elif defined(VF3P)
    #include "parallel/ParallelMatchingEngine.hpp"
    #include "parallel/ParallelMatchingEngineWLS.hpp"
    #include "parallel/CloneableVF3ParallelSubState.hpp"

    template <typename Node, typename Edge>
    using state_t = vflib::CloneableVF3ParallelSubState<Node, Node, Edge, Edge>;
#endif

#endif /* VFLIB_H*/
