
#ifndef NBMPI_NODE_HPP
#define NBMPI_NODE_HPP

#include "utils.hpp"
#include "probe.hpp"

#include <set>
#include <forward_list>
#include <queue>
#include <memory>
#include <future>
#include <list>
#include <ostream>
#include <mpi.h>

enum Decision {DoLB=1, DontLB=0};

template<class LBStruct, class LBStructCopyF, class LBStructDeleteF>
struct Node : public std::enable_shared_from_this<Node<LBStruct, LBStructCopyF, LBStructDeleteF>> {
    using NodeType = Node<LBStruct, LBStructCopyF, LBStructDeleteF>;
private:
    Time node_cost = 0.0;
public:
    int start_it, end_it, batch_size;
    Rank rank;
    Index id;

    std::shared_ptr< NodeType > parent;
    std::vector<Time> li_slowdown_hist, van_li_slowdown_hist, time_hist, time_per_it, efficiency_hist;
    std::vector<int>  dec_hist;

    Decision decision;             // Y / N boolean
    Probe stats{0};
    Time concrete_cost = 0.0;      // estimated cost to the solution

    LBStructCopyF lb_copy_f;
    LBStructDeleteF lb_delete_f;
    LBStruct* lb;

    void init_hist(size_t s) {
        li_slowdown_hist.resize(s);
        van_li_slowdown_hist.resize(s);
        time_hist.resize(s);
        time_per_it.resize(s);
        efficiency_hist.resize(s);
        dec_hist.resize(s);
    }

    Node (Index id, int startit, int batch_size, Decision decision, Probe stats, std::shared_ptr<NodeType> p) :
        id(id),
        start_it(startit), end_it(startit+batch_size), batch_size(batch_size),
        parent(p), decision(decision), stats(std::move(stats)),
        lb_copy_f(p->lb_copy_f), lb_delete_f(p->lb_delete_f),
        lb(lb_copy_f(parent->lb)),
        concrete_cost(parent->concrete_cost) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        init_hist(batch_size);
    };

    Node(LBStruct* zz, int start_it, int batch_size, Decision decision, LBStructCopyF copy_f, LBStructDeleteF delete_f) :
            id(0),
            start_it(start_it), end_it(start_it+batch_size), batch_size(batch_size), parent(nullptr),
            decision(decision),
            lb_copy_f(copy_f),
            lb_delete_f(delete_f),
            lb(lb_copy_f(zz)) {
        int size;
        init_hist(batch_size);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        stats = Probe(size);
    }
    ~Node() {
        lb_delete_f(lb);
    }
    std::array<std::shared_ptr<Node<LBStruct, LBStructCopyF, LBStructDeleteF>>, 2> get_children() {
        if(this->end_it == 0){
            return {
                    std::make_shared<Node>(0, end_it, batch_size, Decision::DontLB, stats, this->shared_from_this()),
                    nullptr
            };
        }else
            return {
                    std::make_shared<Node>(0, end_it, batch_size, Decision::DoLB,   stats, this->shared_from_this()),
                    std::make_shared<Node>(0, end_it, batch_size, Decision::DontLB, stats, this->shared_from_this())
            };
    }
    std::vector<Decision> get_sequence() {
        auto current = this->shared_from_this();
        std::vector<Decision > seq;
        while(current != nullptr){
            seq.push_back(current->decision);
            current = current->parent;
        }
        std::reverse(seq.begin(), seq.end());
        return seq;
    }
    void set_cost(Time ncost) {
        this->node_cost = ncost;
        this->concrete_cost += ncost;
    }
    Time get_node_cost() const { return node_cost; }
    Time cost() const { return concrete_cost; }
    Decision get_decision() const { return decision; }
    int get_target() { return decision == Decision::DoLB; }
};

class Compare
{
public:
    template<class S,class C, class D>
    bool operator() (std::shared_ptr<Node<S, C, D>> a, std::shared_ptr<Node<S, C, D>> b) const {
        return a->cost() < b->cost();
    }
};

template<class S,class C, class D>
bool operator<(const std::shared_ptr<Node<S, C, D>> &n1, const std::shared_ptr<Node<S, C, D>> &n2) {
    return n1->cost() < n2->cost();
}
template<class S,class C, class D>
std::ostream &operator <<(std::ostream& output, const std::shared_ptr<Node<S, C, D>>& value) {
    output << " Iteration: " << std::setw(6) << value->start_it <<  " -> " << std::setw(6) << value->end_it;
    output << " Edge Cost: " << std::setw(6) << std::fixed << std::setprecision(5) << value->get_node_cost();
    output << " Features: (";
    output << value->get_sequence() << " )";
    return output;
}


template<class Container, class S,class C, class D>
void prune_similar_nodes(const std::shared_ptr<Node<S, C, D>>& n, Container& c){
    using NodeType = Node<S, C, D>;
    auto it = c.begin();
    const auto end = c.end();
    while(it != end) {
        std::shared_ptr<NodeType> node = *it;
        if(node->start_it == n->start_it && node->end_it == n->end_it && node->decision == DoLB) {
            it = c.erase(it);
        } else {
            it++;
        }
    }
}

#endif //NBMPI_NODE_HPP
