
#ifndef NBMPI_NODE_HPP
#define NBMPI_NODE_HPP

#include "utils.hpp"
#include "example/zoltan_fn.hpp"
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

    Decision decision;             // Y / N boolean
    Probe stats;
    Time concrete_cost = 0.0;      // estimated cost to the solution

    Zoltan_Struct* lb;

    Node (Index id, int startit, int batch_size, Decision decision, Probe stats, std::shared_ptr<Node> p) :
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
            decision(Decision::DoLB),
            lb(Zoltan_Copy(zz)), stats(0) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        stats = Probe(size);
    }

    Node(Zoltan_Struct* zz, int start_it, int batch_size, Decision decision) :
            id(0),
            start_it(start_it), end_it(start_it+batch_size), batch_size(batch_size), li_slowdown_hist(batch_size), dec_hist(batch_size), time_hist(batch_size), parent(nullptr),
            decision(decision),
            lb(Zoltan_Copy(zz)), stats(0) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        stats = Probe(size);
    }

    ~Node() {
        Zoltan_Destroy(&lb);
    }


    void set_cost(Time ncost);
    Time get_node_cost() const;
    Time cost() const;
    Decision get_decision() const;
    int get_target();
    std::array<std::shared_ptr<Node>, 2> get_children();
    std::vector<Decision> get_sequence();
};

class Compare
{
public:
    bool operator() (std::shared_ptr<Node> a, std::shared_ptr<Node> b) const;
};

bool operator<(const std::shared_ptr<Node> &n1, const std::shared_ptr<Node> &n2);

std::ostream &operator <<(std::ostream& output, const std::shared_ptr<Node>& value);

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

#endif //NBMPI_NODE_HPP
