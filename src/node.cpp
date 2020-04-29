//
// Created by xetql on 4/29/20.
//

#include "node.hpp"

void Node::set_cost(Time ncost) {
    this->node_cost = ncost;
    this->concrete_cost += ncost;
}
Time Node::get_node_cost() const { return node_cost; }
Time Node::cost() const { return concrete_cost; }
Decision Node::get_decision() const { return decision; }
int Node::get_target() { return decision == Decision::DoLB; }

bool Compare::operator() (std::shared_ptr<Node> a, std::shared_ptr<Node> b) const {
    return a->cost() < b->cost();
}

bool operator<(const std::shared_ptr<Node> &n1, const std::shared_ptr<Node> &n2) {
    return n1->cost() < n2->cost();
}

std::ostream &operator <<(std::ostream& output, const std::shared_ptr<Node>& value) {
    output << " Iteration: " << std::setw(6) << value->start_it <<  " -> " << std::setw(6) << value->end_it;
    output << " Edge Cost: " << std::setw(6) << std::fixed << std::setprecision(5) << value->get_node_cost();
    output << " Features: (";
    output << value->get_sequence() << " )";
    return output;
}

std::array<std::shared_ptr<Node>, 2> Node::get_children() {
    if(this->end_it == 0){
        return {
                std::make_shared<Node>(0, end_it, batch_size, Decision::DontLB, stats, this->shared_from_this()),
                nullptr
        };
    }else
        return {
                std::make_shared<Node>(0, end_it, batch_size, Decision::DoLB, stats, this->shared_from_this()),
                std::make_shared<Node>(0, end_it, batch_size, Decision::DontLB, stats, this->shared_from_this())
        };
}
std::vector<Decision> Node::get_sequence() {
    auto current = this->shared_from_this();
    std::vector<Decision > seq;
    while(current != nullptr){
        seq.push_back(current->decision);
        current = current->parent;
    }
    std::reverse(seq.begin(), seq.end());
    return seq;
}
