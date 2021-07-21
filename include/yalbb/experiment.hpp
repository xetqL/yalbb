//
// Created by xetql on 7/21/21.
//

#ifndef NBMPI_EXPERIMENT_HPP
#define NBMPI_EXPERIMENT_HPP
#include <string>
#include <mpi.h>
#include "utils.hpp"

template<unsigned N, class E, class TParam> class Experiment {
protected:
    BoundingBox<N> simbox;
    const std::unique_ptr<TParam>& params;
    MPI_Datatype datatype;
    MPI_Comm APP_COMM;
    std::string name;
    int rank{}, nproc{};

    virtual void setup(E* mesh_data) = 0;
    virtual std::unique_ptr<E> alloc() = 0;

public:
    [[nodiscard]] const std::string& get_exp_name() const { return name; }

    Experiment(BoundingBox<N> simbox, const std::unique_ptr<TParam>& params,
               MPI_Datatype datatype, MPI_Comm APP_COMM,
               std::string name) :
            simbox(std::move(simbox)),
            params(params),
            datatype(datatype),
            APP_COMM(APP_COMM),
            name(std::move(name)) {
        MPI_Comm_rank(APP_COMM, &rank);
        MPI_Comm_size(APP_COMM, &nproc);
    }

    template<class BalancerType, class GetPosFunc>
    auto init(BalancerType* zlb, GetPosFunc getPos, const std::string& preamble, simulation::MonitoringSession& report_session) {
        io::ParallelOutput pcout(std::cout);
        pcout << preamble << std::endl;

        auto mesh_data = alloc();

        lb::InitLB<BalancerType>      init {};
        lb::DoPartition<BalancerType> doPartition {};
        lb::AssignPoint<BalancerType> pointAssign {};

        setup(mesh_data.get());
        init(zlb, mesh_data.get());

        Probe probe(nproc);

        PAR_START_TIMER(lbtime, APP_COMM);
        doPartition(zlb, mesh_data.get(), getPos);
        migrate_data(zlb, mesh_data->els, pointAssign, datatype, APP_COMM);
        END_TIMER(lbtime);

        size_t n_els = mesh_data->els.size(), max_els, tot_els;

        MPI_Allreduce(MPI_IN_PLACE, &lbtime, 1, get_mpi_type<decltype(lbtime)>(), MPI_MAX, APP_COMM);
        MPI_Allreduce(&n_els,       &tot_els,1, get_mpi_type<size_t>(),           MPI_SUM, APP_COMM);
        MPI_Allreduce(&n_els,       &max_els,1, get_mpi_type<size_t>(),           MPI_MAX, APP_COMM);

        probe.push_load_balancing_time(lbtime);
        Real efficiency = (static_cast<Real>(tot_els) / static_cast<Real>(nproc)) / static_cast<Real>(max_els);
        probe.push_load_balancing_parallel_efficiency(efficiency);

        report_session.report(simulation::LoadBalancingCost, lbtime, " ");

        probe.set_balanced(true);

        pcout << name << std::endl;

        return std::make_tuple(std::move(mesh_data), probe);
    }
};

#endif //NBMPI_EXPERIMENT_HPP
