//
// Created by xetql on 19.03.18.
//

#ifndef NBMPI_GRAPH_HPP
#define NBMPI_GRAPH_HPP

#include <algorithm>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <memory>
#include "utils.hpp"
#include "spatial_elements.hpp"

typedef boost::adjacency_matrix<boost::undirectedS> AdjMGraph;

typedef boost::adjacency_list<
        boost::vecS, boost::vecS, boost::undirectedS,
        boost::no_property, boost::no_property, boost::no_property,
        boost::vecS> Graph;

template<int N, class MapType>
Graph build_graph(
        const int M, /* Number of subcell in a row  */
        const float lsub, /* length of a cell */
        std::vector<elements::Element<N>> &local_elements,
        const MapType &plist) /* Simulation parameters */ {
    int linearcellidx;
    int nlinearcellidx;
    int num_vertices = 0;
    const int n_elements = local_elements.size();
    Graph g(n_elements);

    for (int lid = 0; lid < n_elements; ++lid)
        auto force_recepter = local_elements[lid];

    // worst case O(n²), best case is a strict O(n) , average O(n*k) => O(n) because the number of neighbors (k) is << n
    for (int lid = 0; lid < n_elements; ++lid) { //auto &force_recepter : local_elements) { // O(n)
        auto force_recepter = local_elements[lid];
        // find cell from particle position
        linearcellidx = (int) (std::floor(force_recepter.position.at(0) / lsub)) + M * (std::floor(force_recepter.position.at(1) / lsub));
        // convert linear index to grid position
        int xcellidx = linearcellidx % M; int ycellidx = linearcellidx / M;
        // average case 3*3 (worst also), best case 2*2
        for (int neighborx = -1; neighborx < 2; neighborx++) {
            for (int neighbory = -1; neighbory < 2; neighbory++) {
                /* Check boundary conditions */
                if (xcellidx + neighborx < 0 || xcellidx + neighborx >= M) continue;
                if (ycellidx + neighbory < 0 || ycellidx + neighbory >= M) continue;
                nlinearcellidx = (xcellidx + neighborx) + M * (ycellidx + neighbory);
                if(plist.find(nlinearcellidx) != plist.end()) { // avg. O(1), worst O(P), where P is the number of processors
                    auto el_list = plist.at(nlinearcellidx).get(); //get the pointer but we will never release it manually.
                    size_t cell_el_size = el_list->size();
                    num_vertices += cell_el_size;
                    for(size_t el_idx = 0; el_idx < cell_el_size; ++el_idx) {
                        auto const& force_source = el_list->at(el_idx);
                        if (force_recepter.gid < force_source.gid && force_source.lid < n_elements && force_recepter.lid < n_elements) {
                            boost::add_edge(force_recepter.lid, force_source.lid, g);
                        }
                    }
                }
            }
        }
    }
    return g;
}


#endif //NBMPI_GRAPH_HPP
