
#ifndef NBMPI_ASTAR_HPP
#define NBMPI_ASTAR_HPP

//
// Created by xetql on 23.05.18.
//

#include <forward_list>
#include <queue>
#include <memory>
#include <future>

template<typename MESH_DATA, typename Domain>
struct Node : public std::enable_shared_from_this<Node<MESH_DATA, Domain>> {
    int idx;                // index of the node
    int iteration;
    bool decision;          // Y / N
    float   parent_cost,   // cost of until now
            node_cost,
            heuristic_cost;      // estimated cost to the solution
    MESH_DATA* mesh_data;    // particles informations
    std::shared_ptr<Node<MESH_DATA, Domain>> parent;
    Domain domain;

    float cost() const {
        return parent_cost + node_cost + heuristic_cost;
    }

    Node (int idx, int it, bool decision, float parent_cost, float node_cost, float heuristic,
          MESH_DATA* mesh_data, std::shared_ptr<Node<MESH_DATA, Domain>> p, Domain domain) :
        idx(idx),
        iteration(it),
        decision(decision),
        parent_cost(parent_cost),
        node_cost(node_cost),
        heuristic_cost(heuristic),
        mesh_data(mesh_data),
        parent(p),
        domain(domain){}

    std::pair<std::shared_ptr<Node<MESH_DATA, Domain>>, std::shared_ptr<Node<MESH_DATA, Domain>>> get_children(){
        return std::make_pair(std::make_shared<Node<MESH_DATA, Domain>>(2*idx+1, iteration, true,  parent_cost + node_cost, 0, 0, mesh_data, this->shared_from_this(), domain),
                              std::make_shared<Node<MESH_DATA, Domain>>(2*idx+2, iteration, false, parent_cost + node_cost, 0, 0, mesh_data, this->shared_from_this(), domain)
        );
    };

};
template<typename MESH_DATA, typename Domain>
bool operator<(const Node<MESH_DATA, Domain> &n1, const Node<MESH_DATA, Domain> &n2) {
    return n1.cost() > n2.cost();
}

template<typename MESH_DATA, typename Domain>
bool operator==(const Node<MESH_DATA, Domain> &n1, const Node<MESH_DATA, Domain> &n2) {
    return n1.idx == n2.idx;
}

#endif //NBMPI_ASTAR_HPP
