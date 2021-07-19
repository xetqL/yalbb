//
// Created by xetql on 4/29/20.
//


#include <numeric>
#include <probe.hpp>
Probe::Probe(int nproc) : nproc(nproc) {}

void Probe::update_cumulative_imbalance_time(Real imbalance_time) {
    /**The cumulative load imbalance since a load balancing.
     * I don't think that we have to take into account the remaining imbalance after a load balancing because
     * re-using the LB algorithm won't remove it. For instance, let pretends that the load balancing cost is
     * C=0.1s and the remaining imbalance (max-avg) >= C, then the criterion is matched at every iteration, alas,
     * we know that the algorithm is not capable to do better. Hence, we can treat C as a baseline.*/
    if(this->balanced) lb_imbalance_baseline = imbalance_time; //(max_it - (sum_it/nproc)); // use the remaining imbalance as baseline
    cumulative_imbalance_time += std::max(0.0, (imbalance_time - lb_imbalance_baseline));
    vanilla_cumulative_imbalance_time += imbalance_time;
}

void  Probe::reset_cumulative_imbalance_time() {
    cumulative_imbalance_time = 0.0;
    vanilla_cumulative_imbalance_time = 0.0;
    iteration_times_since_lb.clear();
}

Time* Probe::max_it_time() { return &max_it; }
Time* Probe::min_it_time() { return &min_it; }

void Probe::set_balanced(bool lb_status) {
    if(lb_status) Probe::previous_lb_it = current_iteration;
    Probe::balanced = lb_status;
}

bool Probe::is_balanced() const {
    return balanced;
}

int Probe::get_current_iteration() const {
    return current_iteration;
}
Time Probe::get_avg_it() const {
    return sum_it/nproc;
}

Time Probe::get_max_it() const {
    return max_it;
}

Time Probe::get_min_it() const {
    return min_it;
}

Time Probe::get_sum_it() const {
    return sum_it;
}

Time Probe::get_batch_time() const{
    return batch_time;
}

Time Probe::get_cumulative_imbalance_time() const {
    return cumulative_imbalance_time;
}
Time Probe::get_vanilla_cumulative_imbalance_time() const {
    return vanilla_cumulative_imbalance_time;
}

Time* Probe::sum_it_time() { return &sum_it; }

void  Probe::push_load_balancing_time(Time lb_time){
    lb_times.push_back(lb_time);
}
void  Probe::push_load_balancing_parallel_efficiency(Real lb_parallel_efficiency){
    lb_parallel_efficiencies.push_back(lb_parallel_efficiency); }
void  Probe::update_lb_parallel_efficiencies() { lb_parallel_efficiencies.push_back(get_avg_it() / get_max_it()); }

Real Probe::get_current_parallel_efficiency() const { return lb_parallel_efficiencies.back();}

void Probe::next_iteration() {current_iteration++;}
void Probe::start_batch(Index batch) { current_batch = batch; }
void Probe::end_batch(Time time) { batch_time = time; }
bool Probe::is_batch_started() { return batch_started; }
void Probe::sync_it_time_across_processors(Time *t, MPI_Comm comm) {
    // Synchronize
    MPI_Allreduce(t, this->max_it_time(), 1, get_mpi_type<Time>(), MPI_MAX, comm);
    MPI_Allreduce(t, this->min_it_time(), 1, get_mpi_type<Time>(), MPI_MIN, comm);
    MPI_Allreduce(t, this->sum_it_time(), 1, get_mpi_type<Time>(), MPI_SUM, comm);
    // Set
    *t = get_max_it();
    iteration_times_since_lb.push_back(get_max_it());
}
