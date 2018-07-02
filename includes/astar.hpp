
#ifndef NBMPI_ASTAR_HPP
#define NBMPI_ASTAR_HPP

#include <set>
#include <forward_list>
#include <queue>
#include <memory>
#include <future>
#include "metrics.hpp"
#include "utils.hpp"


template<typename MESH_DATA, typename Domain>
struct Node : public metric::FeatureContainer, public std::enable_shared_from_this<Node<MESH_DATA, Domain>>{
    long long idx = 0;                // index of the node
    int start_it, end_it;
    std::shared_ptr<Node<MESH_DATA, Domain>> parent;

    bool decision;          // Y / N
    double  node_cost,
            path_cost,
            heuristic_cost;      // estimated cost to the solution
    std::vector<double> metrics_before_decision, last_metric;
    MESH_DATA mesh_data;    // particles informations
    Domain domain;

    inline double get_total_path_cost() const {
        return path_cost + node_cost;
    }

    inline double cost() const {
        return get_total_path_cost() + heuristic_cost;
    }

    Node (int idx, int it, bool decision, double node_cost, double heuristic,
          MESH_DATA mesh_data, std::shared_ptr<Node<MESH_DATA, Domain>> p, Domain domain) :
        idx(idx),
        start_it(it),
        end_it(it),
        parent(p),
        decision(decision),
        node_cost(node_cost),
        path_cost(parent->path_cost + parent->node_cost),
        heuristic_cost(heuristic),
        metrics_before_decision(parent->last_metric),
        mesh_data(mesh_data),
        domain(domain) {}

    Node(MESH_DATA mesh_data, Domain domain):
            start_it(0), end_it(0), parent(nullptr), decision(true),
            node_cost(0), path_cost(0), heuristic_cost(0),
            mesh_data(mesh_data),  domain(domain){}

    std::pair<std::shared_ptr<Node<MESH_DATA, Domain>>, std::shared_ptr<Node<MESH_DATA, Domain>>> get_children(){
        return std::make_pair(std::make_shared<Node<MESH_DATA, Domain>>(2*idx+1, end_it, true,  0, 0, mesh_data, this->shared_from_this(), domain),
                              std::make_shared<Node<MESH_DATA, Domain>>(2*idx+2, end_it, false, 0, 0, mesh_data, this->shared_from_this(), domain)
        );
    };

    std::vector<double> get_features() override {
        return functional::slice(metrics_before_decision, 0, metrics_before_decision.size() - 1);
    }

    int get_target() override {
        return decision? 1:0;
    }
};

template<class MESH_DATA, class Domain>
class Compare
{
public:
    bool operator() (std::shared_ptr<Node<MESH_DATA, Domain>> a, std::shared_ptr<Node<MESH_DATA, Domain>> b) {
        return a->cost() > b->cost();
    }
};

template<typename MESH_DATA, typename Domain>
bool operator<(const std::shared_ptr<Node<MESH_DATA, Domain>> &n1, const std::shared_ptr<Node<MESH_DATA, Domain>> &n2) {
    return n1->cost() > n2->cost();
}

template<typename MESH_DATA, typename Domain>
bool operator==(const std::shared_ptr<Node<MESH_DATA, Domain>> &n1, const std::shared_ptr<Node<MESH_DATA, Domain>> &n2) {
    return n1->idx == n2->idx;
}

template<typename MESH_DATA, typename Domain>
std::ostream &operator <<(std::ostream& output, const std::shared_ptr<Node<MESH_DATA, Domain>>& value)
{
    output << "ID: "           << std::setw(4) << value->idx;
    output << " Iteration: " << std::setw(6) << value->start_it <<  " -> " << std::setw(6) << value->end_it;
    output << " Edge Cost: " << std::setw(6) << std::fixed << std::setprecision(5) << value->node_cost;
    output << " Features: (";
    for(auto const& feature: value->metrics_before_decision){
        output << std::setw(6) << std::fixed << std::setprecision(3) << feature << ",";
    }
    output << (value->decision ? " Y":" N") << " )" <<std::endl;
    return output;
}

template<typename Domain>
struct NodeWithoutParticles : public metric::FeatureContainer, public std::enable_shared_from_this<NodeWithoutParticles<Domain>>{
    long long idx = 0;                // index of the node
    int start_it, end_it;
    std::shared_ptr<NodeWithoutParticles<Domain>> parent;

    bool decision;          // Y / N
    double  node_cost,
            path_cost,
            heuristic_cost;      // estimated cost to the solution
    std::vector<double> metrics_before_decision, last_metric;
    Domain domain;

    inline double get_total_path_cost() const {
        return path_cost + node_cost;
    }

    inline double cost() const {
        return get_total_path_cost() + heuristic_cost;
    }

    NodeWithoutParticles (int idx, int it, bool decision, double node_cost, double heuristic, std::shared_ptr<NodeWithoutParticles<Domain>> p, Domain domain) :
            idx(idx),
            start_it(it),
            end_it(it),
            parent(p),
            decision(decision),
            node_cost(node_cost),
            path_cost(parent->path_cost),
            heuristic_cost(heuristic),
            metrics_before_decision(parent->last_metric),
            domain(domain) {}

    NodeWithoutParticles(Domain domain):
            start_it(0), end_it(0), parent(nullptr), decision(true),
            node_cost(0), path_cost(0), heuristic_cost(0),
            domain(domain){
    }

    std::pair<std::shared_ptr<NodeWithoutParticles<Domain>>, std::shared_ptr<NodeWithoutParticles<Domain>>> get_children(){
        return std::make_pair(std::make_shared<NodeWithoutParticles<Domain>>(2*idx+1, end_it, true,  0, 0, this->shared_from_this(), domain),
                              std::make_shared<NodeWithoutParticles<Domain>>(2*idx+2,end_it, false, 0, 0, this->shared_from_this(), domain)
        );
    };

    std::vector<double> get_features() override {
        return functional::slice(metrics_before_decision, 0, metrics_before_decision.size() - 1);
    }

    int get_target() override {
        return decision? 1:0;
    }
};


template<class Domain>
class CompareNodeWithoutParticles
{
public:
    bool operator() (std::shared_ptr<NodeWithoutParticles<Domain>> a, std::shared_ptr<NodeWithoutParticles<Domain>> b) {
        return a->cost() > b->cost();
    }
};

template<typename MESH_DATA, typename Domain>
bool operator<(const std::shared_ptr<NodeWithoutParticles<Domain>> &n1, const std::shared_ptr<NodeWithoutParticles<Domain>> &n2) {
    return n1->cost() > n2->cost();
}

template<typename MESH_DATA, typename Domain>
bool operator==(const std::shared_ptr<NodeWithoutParticles<Domain>> &n1, const std::shared_ptr<NodeWithoutParticles<Domain>> &n2) {
    return n1->idx == n2->idx;
}

template<typename MESH_DATA, typename Domain>
std::ostream &operator <<(std::ostream& output, const std::shared_ptr<NodeWithoutParticles<Domain>>& value)
{
    output << "ID: "           << std::setw(4) << value->idx;
    output << " Iteration: " << std::setw(6) << value->start_it <<  " -> " << std::setw(6) << value->end_it;
    output << " Edge Cost: " << std::setw(6) << std::fixed << std::setprecision(5) << value->node_cost;
    output << " Features: (";
    for(auto const& feature: value->metrics_before_decision){
        output << std::setw(6) << std::fixed << std::setprecision(3) << feature << ",";
    }
    output << (value->decision ? " Y":" N") << " )" <<std::endl;
    return output;
}
template<class Data, class Domain>
int has_been_explored(std::multiset<std::shared_ptr<Node<Data, Domain> >, Compare<Data, Domain> > c, int start_it) {

    for(auto const& node : c){
        if(node->start_it > start_it) return true;
    }
    return false;
}


#endif //NBMPI_ASTAR_HPP
