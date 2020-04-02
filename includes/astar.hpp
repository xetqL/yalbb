
#ifndef NBMPI_ASTAR_HPP
#define NBMPI_ASTAR_HPP

#include <armadillo>

#include <set>
#include <forward_list>
#include <queue>
#include <memory>
#include <future>
#include <list>
#include <ostream>
#include <mpi.h>

#include "utils.hpp"
#include "feature_container.hpp"

enum NodeLBDecision {DoLB=1, DontLB=0};

struct Node : public std::enable_shared_from_this<Node>{
private:
    Time node_cost = 0.0;
public:
    int start_it, end_it, batch_size;
    Rank rank;
    Index id;
    std::shared_ptr<Node> parent;
    NodeLBDecision decision;          // Y / N boolean
    IterationStatistics stats;
    Time concrete_cost = 0.0;      // estimated cost to the solution

    Zoltan_Struct* lb;

    void set_cost(Time ncost) {
        this->node_cost = ncost;
        this->concrete_cost += ncost;
    }

    Time get_node_cost() const {
        return node_cost;
    }

    inline Time cost() const {
        return concrete_cost;
    }

    NodeLBDecision get_decision() const {
        return decision;
    }

    int get_target() {
        return decision == NodeLBDecision::DoLB;
    }

    Node (Index id, int startit, int batch_size, NodeLBDecision decision, IterationStatistics stats, std::shared_ptr<Node> p) :
        id(id),
        start_it(startit), end_it(startit+batch_size), batch_size(batch_size),
        parent(p), decision(decision), stats(stats),
        concrete_cost(parent->concrete_cost), lb(Zoltan_Copy(parent->lb)) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    };

    Node(Zoltan_Struct* zz, int batch_size) :
            id(0),
            start_it(0), end_it(batch_size), batch_size(batch_size), parent(nullptr),
            decision(NodeLBDecision::DoLB),
            lb(zz) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        stats = IterationStatistics(size);
    }

    Node(Zoltan_Struct* zz, int start_it, int batch_size, NodeLBDecision decision) :
            id(0),
            start_it(start_it), end_it(start_it+batch_size), batch_size(batch_size), parent(nullptr),
            decision(decision),
            lb(zz) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        stats = IterationStatistics(size);
    }

    ~Node() {
        std::cout << "Destruction" << std::endl;
        Zoltan_Destroy(&lb);
    }

    std::array<std::shared_ptr<Node>, 2> get_children() {

        return {
            std::make_shared<Node>(0, end_it, batch_size, NodeLBDecision::DoLB, stats, this->shared_from_this()),
            std::make_shared<Node>(0, end_it, batch_size, NodeLBDecision::DontLB, stats, this->shared_from_this())
        };
    }

};

class Compare
{
public:
    bool operator() (std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        return a->cost() < b->cost();
    }
};

template<typename MESH_DATA>
bool operator<(const std::shared_ptr<Node> &n1, const std::shared_ptr<Node> &n2) {
    return n1->cost() < n2->cost();
}

std::ostream &operator <<(std::ostream& output, const std::shared_ptr<Node>& value)
{
    output << " Iteration: " << std::setw(6) << value->start_it <<  " -> " << std::setw(6) << value->end_it;
    output << " Edge Cost: " << std::setw(6) << std::fixed << std::setprecision(5) << value->get_node_cost();
    output << " Features: (";
    output << (value->decision == NodeLBDecision::DoLB ? "Y":"N") << " )";
    return output;
}

template<class Container>
void prune_similar_nodes(const std::shared_ptr<Node>& n, Container& c){
    auto it = c.begin();
    const auto end = c.end();
    while(it != end) {
        auto current = it++; // copy the current iterator then increment it
        std::shared_ptr<Node> node = *current;
        if(node->start_it == n->start_it && node->decision == DoLB) {
            c.erase(current);
        }
    }
}

bool has_been_explored(const std::multiset<std::shared_ptr<Node>, Compare>& c, std::shared_ptr<Node> target) {
    return std::any_of(c.cbegin(), c.cend(), [target](auto node){return node->start_it >= target->start_it;});
}

#endif //NBMPI_ASTAR_HPP
