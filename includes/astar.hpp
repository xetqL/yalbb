
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

enum NodeType {Partitioning, Computing};
enum NodeLBDecision {LoadBalance, DoNothing};

template<typename MESH_DATA>
struct Node : public FeatureContainer, public std::enable_shared_from_this<Node<MESH_DATA>>{
private:
    double node_cost = 0.0;
public:

    int start_it, end_it, rank;
    uint64_t id;
    std::shared_ptr<Node<MESH_DATA>> parent;
    //std::shared_ptr<SlidingWindow<double>> window_gini_times, window_gini_complexities , window_times, window_gini_communications;
    NodeLBDecision decision;          // Y / N boolean
    NodeType type;

    double heuristic_cost = 0.0,
            concrete_cost = 0.0;      // estimated cost to the solution

    std::vector<double> metrics_before_decision, last_metric;
    //MESH_DATA mesh_data;    // particles informations

    Zoltan_Struct* lb;

    void set_cost(double node_cost) {
        this->node_cost = node_cost;
        this->concrete_cost += node_cost;
    }

    double get_node_cost() const {
        return node_cost;
    }

    inline double cost() const {
        return concrete_cost + heuristic_cost;
    }

    NodeType get_node_type() const {
        return type;
    }

    NodeLBDecision get_decision() const {
        return decision;
    }

    std::vector<double> get_features() override {
        return functional::slice(metrics_before_decision, 0, metrics_before_decision.size() - 1);
    }

    int get_target() override {
        return decision == NodeLBDecision::LoadBalance ? 1:0;
    }

    Node (uint64_t id, int startit, NodeLBDecision decision, NodeType type, std::shared_ptr<Node<MESH_DATA>> p) :
            id(id),
            start_it(startit), end_it(startit),
            parent(p),

            /*window_gini_times(std::make_shared<SlidingWindow<double>>(*p->window_gini_times)),
            window_gini_complexities(std::make_shared<SlidingWindow<double>>(*p->window_gini_complexities)),
            window_times(std::make_shared<SlidingWindow<double>>(*p->window_times)),
            window_gini_communications(std::make_shared<SlidingWindow<double>>(*p->window_gini_communications)),*/
            decision(decision),
            type(type),
            concrete_cost(parent->concrete_cost),
            metrics_before_decision(parent->last_metric),
            //mesh_data(mesh_data),
            lb(Zoltan_Copy(parent->lb)){
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    };

    Node(Zoltan_Struct* zz, int size) :
            id(0),
            start_it(0), end_it(0), parent(nullptr),
            /*window_gini_times(std::make_shared<SlidingWindow<double>>(size)),
            window_gini_complexities(std::make_shared<SlidingWindow<double>>(size)),
            window_times(std::make_shared<SlidingWindow<double>>(size)),
            window_gini_communications(std::make_shared<SlidingWindow<double>>(size)),*/
            decision(NodeLBDecision::LoadBalance), type(NodeType::Computing),
            //mesh_data(mesh_data),
            lb(zz) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    ~Node() {
        Zoltan_Destroy(&lb);
    }

    std::array<std::shared_ptr<Node<MESH_DATA>>, 2> get_children(){

        std::ofstream treef;
        if (rank == 0) {
            treef.open("tree_file", std::ofstream::out | std::ofstream::app);
            treef << id << " " << 2*id+1 << " " << 2*binary_node_max_id_for_level(id)+2 << std::endl;
            treef.close();
        }

        switch(type) {
            case NodeType::Partitioning:
                return {
                        std::make_shared<Node<MESH_DATA>>(id, end_it, NodeLBDecision::LoadBalance, NodeType::Computing,  this->shared_from_this()),
                        nullptr
                };
            case NodeType::Computing:
                if(end_it == 0) //starting case to start using the TCP connection ... does not ask me why ...
                    return {
                            std::make_shared<Node<MESH_DATA>>(2*binary_node_max_id_for_level(id)+2, end_it, NodeLBDecision::LoadBalance, NodeType::Computing, this->shared_from_this()),
                            std::make_shared<Node<MESH_DATA>>(2*id+1, end_it, NodeLBDecision::DoNothing,   NodeType::Computing, this->shared_from_this())
                    };

                if(start_it == 0 && decision == NodeLBDecision::LoadBalance)
                    return {nullptr, nullptr};

                return {
                        std::make_shared<Node<MESH_DATA>>(2*binary_node_max_id_for_level(id)+2, end_it, NodeLBDecision::LoadBalance, NodeType::Partitioning, this->shared_from_this()),
                        std::make_shared<Node<MESH_DATA>>(2*id+1, end_it, NodeLBDecision::DoNothing,   NodeType::Computing,    this->shared_from_this())
                };

        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Node &node) {

        return os;
    }
};

template<typename MESH_DATA>
std::pair<arma::mat,arma::mat> to_armadillo_mat(std::list<std::shared_ptr<Node<MESH_DATA>>> dataset, int nfeatures) {
    const size_t nfeat = (*dataset.begin())->get_features().size();
    arma::mat arma_features(0, nfeat);
    arma::mat arma_targets(0, 1);
    std::cout << arma_targets.n_rows << std::endl;

    int i = 0;
    for(auto& line : dataset) {
        if( line->type == NodeType::Computing ) {
            arma::rowvec features(line->get_features());
            arma::rowvec target;
            target << (int) line->get_target() << arma::endr;
            arma_features.insert_rows(i, features);
            arma_targets.insert_rows(i, target);
            i++;
        }
    }

    return std::make_pair(arma_features, arma_targets);
}

template<class MESH_DATA>
class Compare
{
public:
    bool operator() (std::shared_ptr<Node<MESH_DATA>> a, std::shared_ptr<Node<MESH_DATA>> b) {
        return a->cost() < b->cost();
    }
};

template<typename MESH_DATA>
bool operator<(const std::shared_ptr<Node<MESH_DATA>> &n1, const std::shared_ptr<Node<MESH_DATA>> &n2) {
    return n1->cost() < n2->cost();
}

template<typename MESH_DATA>
std::ostream &operator <<(std::ostream& output, const std::shared_ptr<Node<MESH_DATA>>& value)
{
    output << " Iteration: " << std::setw(6) << value->start_it <<  " -> " << std::setw(6) << value->end_it;
    output << " Edge Cost: " << std::setw(6) << std::fixed << std::setprecision(5) << value->get_node_cost();
    output << " Features: (";
    for(auto const& feature: value->metrics_before_decision){
        output << std::setw(6) << std::fixed << std::setprecision(3) << feature << ",";
    }
    output << (value->decision == NodeLBDecision::LoadBalance ? "Y":"N") << " )";
    output << (value->type == NodeType::Computing ? "Cpt":"Part") << std::endl;
    return output;
}

template<typename Domain>
struct NodeWithoutParticles : public FeatureContainer, public std::enable_shared_from_this<NodeWithoutParticles<Domain>>{
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
    for(auto const& feature: value->metrics_before_decision)
        output << std::setw(6) << std::fixed << std::setprecision(3) << feature << ",";

    output << (value->decision ? " Y":" N") << " )" <<std::endl;
    return output;
}

template<class Data>
int has_been_explored(std::multiset<std::shared_ptr<Node<Data> >, Compare<Data> > c, int start_it) {
    return std::any_of(c.cbegin(), c.cend(), [&start_it](auto node){return node->start_it >= start_it;});
}

#endif //NBMPI_ASTAR_HPP
