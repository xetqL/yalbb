
#ifndef NBMPI_ASTAR_HPP
#define NBMPI_ASTAR_HPP

#include "utils.hpp"

#include <set>
#include <forward_list>
#include <queue>
#include <memory>
#include <future>
#include <list>
#include <ostream>
#include <mpi.h>

enum NodeLBDecision {DoLB=1, DontLB=0};

struct Node : public std::enable_shared_from_this<Node>{
private:
    Time node_cost = 0.0;
public:
    int start_it, end_it, batch_size;
    Rank rank;
    Index id;
    std::shared_ptr<Node> parent;
    std::vector<Time> li_slowdown_hist;
    std::vector<int> dec_hist;
    std::vector<Time> time_hist;

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
        start_it(startit), end_it(startit+batch_size), batch_size(batch_size), li_slowdown_hist(batch_size), dec_hist(batch_size), time_hist(batch_size),
        parent(p), decision(decision), stats(stats), lb(Zoltan_Copy(parent->lb)),
        concrete_cost(parent->concrete_cost){
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    };

    Node(Zoltan_Struct* zz, int batch_size) :
            id(0),
            start_it(0), end_it(batch_size), batch_size(batch_size), li_slowdown_hist(batch_size), dec_hist(batch_size), time_hist(batch_size), parent(nullptr),
            decision(NodeLBDecision::DoLB),
            lb(Zoltan_Copy(zz)) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        stats = IterationStatistics(size);
    }

    Node(Zoltan_Struct* zz, int start_it, int batch_size, NodeLBDecision decision) :
            id(0),
            start_it(start_it), end_it(start_it+batch_size), batch_size(batch_size), li_slowdown_hist(batch_size), dec_hist(batch_size), time_hist(batch_size), parent(nullptr),
            decision(decision),
            lb(Zoltan_Copy(zz)) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        stats = IterationStatistics(size);
    }

    ~Node() {
        Zoltan_Destroy(&lb);
    }

    std::array<std::shared_ptr<Node>, 2> get_children() {
        if(this->end_it == 0){
            return {
                std::make_shared<Node>(0, end_it, batch_size, NodeLBDecision::DontLB, stats, this->shared_from_this()),
                nullptr
            };
        }else
            return {
                std::make_shared<Node>(0, end_it, batch_size, NodeLBDecision::DoLB, stats, this->shared_from_this()),
                std::make_shared<Node>(0, end_it, batch_size, NodeLBDecision::DontLB, stats, this->shared_from_this())
            };
    }

    std::vector<NodeLBDecision> get_sequence() {
        auto current = this->shared_from_this();
        std::vector<NodeLBDecision > seq;
        while(current != nullptr){
            seq.push_back(current->decision);
            current = current->parent;
        }
        std::reverse(seq.begin(), seq.end());
        return seq;
    }

};

class Compare
{
public:
    bool operator() (std::shared_ptr<Node> a, std::shared_ptr<Node> b) const {
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
    output << value->get_sequence() << " )";
    return output;
}

template<class Container>
void prune_similar_nodes(const std::shared_ptr<Node>& n, Container& c){
    auto it = c.begin();
    const auto end = c.end();
    while(it != end) {
        std::shared_ptr<Node> node = *it;
        if(node->start_it == n->start_it && node->end_it == n->end_it && node->decision == DoLB) {
            it = c.erase(it);
        } else {
            it++;
        }
    }
}

bool has_been_explored(const std::multiset<std::shared_ptr<Node>, Compare>& c, std::shared_ptr<Node> target) {
    return std::any_of(c.cbegin(), c.cend(), [target](auto node){return node->start_it >= target->start_it;});
}

#endif //NBMPI_ASTAR_HPP
