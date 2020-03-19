//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_LJPOTENTIAL_HPP
#define NBMPI_LJPOTENTIAL_HPP

#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>

#include "params.hpp"
#include "physics.hpp"
#include "utils.hpp"
#include "zoltan_fn.hpp"

namespace algorithm {

    constexpr Integer EMPTY = -1;

    template<int N>
    void CLL_init(const elements::Element<N> *local_elements, Integer local_n_elements,
                  const elements::Element<N> *remote_elements, Integer remote_n_elements,
                  Integer lc[N], Real rc, Integer *lscl,
                  Integer *head) {
        Integer lcxyz = lc[0] * lc[1];
        if constexpr (N == 3) {
            lcxyz *= lc[2];
        }
        Integer c;
        for (size_t i = 0; i < lcxyz; ++i) head[i] = EMPTY;

        for (size_t i = 0; i < local_n_elements; ++i) {
            c = position_to_cell<N>(local_elements[i].position, rc, lc[0], lc[1]);
            lscl[i] = head[c];
            head[c] = i;
        }

        for (size_t i = 0; i < remote_n_elements; ++i) {
            c = position_to_cell<N>(remote_elements[i].position, rc, lc[0], lc[1]);
            lscl[i+local_n_elements] = head[c];
            head[c] = i+local_n_elements;
        }
    }

    template<int N>
    void CLL_init(const elements::Element<N> *local_elements, Integer local_n_elements,
                  const elements::Element<N> *remote_elements, Integer remote_n_elements,
                  const BoundingBox<N>& bbox, Real rc,
                  Integer *lscl,
                  Integer *head) {
        auto lc = elements::get_cell_number_by_dimension<N>(bbox, rc);
        Integer lcxyz = std::accumulate(lc.cbegin(), lc.cend(), 1, [](auto prev, auto v){ return prev*v; }),
                c;
        for (size_t i = 0; i < lcxyz; ++i) head[i] = EMPTY;

        for (size_t i = 0; i < local_n_elements; ++i) {
            c = position_to_local_cell_index<N>(local_elements[i].position, rc, bbox, lc[0], lc[1]);
            lscl[i] = head[c];
            head[c] = i;
        }

        for (size_t i = 0; i < remote_n_elements; ++i) {
            c = position_to_local_cell_index<N>(remote_elements[i].position, rc, bbox, lc[0], lc[1]);
            lscl[i+local_n_elements] = head[c];
            head[c] = i+local_n_elements;
        }
    }

    template<int N>
    inline void CLL_append(Integer i, Integer cell, const elements::Element<N>& elements, Integer *lscl, Integer *head) {
        lscl[i] = head[cell];
        head[cell] = i;
    }

    void CLL_compute_forces3d(elements::Element<3> *elements, Integer n_elements,
                              const elements::Element<3> *remote_elements, Integer remote_n_elements,
                              const Integer lc[3], Real rc,
                              const Integer *lscl, const Integer *head,
                              const sim_param_t* params) {
        std::array<Real, 3> delta_dim;
        Real delta;
        Real sig2 = params->sig_lj*params->sig_lj;
        Integer c, c1, ic[3], ic1[3], j;
        elements::Element<3> source, receiver;
        for (size_t i = 0; i < n_elements; ++i) {
            c = position_to_cell<3>(elements[i].position, rc, lc[0], lc[1]);
            receiver = elements[i];
            for (auto d = 0; d < 3; ++d)
                ic[d] = c / lc[d];
            for (ic1[0] = ic[0] - 1; ic1[0] < (ic[0]+1); ic1[0]++) {
                for (ic1[1] = ic[1] - 1; ic1[1] < ic[1] + 1; ic1[1]++) {
                    for (ic1[2] = ic[2] - 1; ic1[2] < ic[2] + 1; ic1[2]++) {
                        /* this is for bounce back, to avoid heap-buffer over/under flow*/
                        if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) || (ic1[2] < 0 || ic1[2] >= lc[2])) continue;
                        c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                        j = head[c1];
                        while(j != EMPTY){
                            if(i < j) {
                                delta = 0.0;
                                source = j < n_elements ? elements[j] : remote_elements[j - n_elements];
                                for (int dim = 0; dim < 3; ++dim)
                                    delta_dim[dim] = receiver.position.at(dim) - source.position.at(dim);
                                for (int dim = 0; dim < 3; ++dim)
                                    delta += (delta_dim[dim] * delta_dim[dim]);
                                Real C_LJ = compute_LJ_scalar<Real>(delta, params->eps_lj, sig2);
                                for (int dim = 0; dim < 3; ++dim) {
                                    receiver.acceleration[dim] += (C_LJ * delta_dim[dim]);
                                }
                            }
                            j = lscl[j];
                        }
                    }
                }
            }
        }
    }

    Integer CLL_compute_forces3d(elements::Element<3> *elements, Integer n_elements,
                              const elements::Element<3> *remote_elements, Integer remote_n_elements,
                              const BoundingBox<3>& bbox, Real rc,
                              const Integer *lscl, const Integer *head,
                              const sim_param_t* params) {
        auto lc = elements::get_cell_number_by_dimension<3>(bbox, rc);

        std::array<Real, 3> delta_dim;
        Real delta;
        Real sig2 = params->sig_lj*params->sig_lj;
        Integer c, c1, ic[3], ic1[3], j;
        elements::Element<3> source, receiver;
        Integer cmplx = n_elements;
        for (size_t i = 0; i < n_elements; ++i) {
            c = position_to_local_cell_index<3>(elements[i].position, rc, bbox, lc[0], lc[1]);
            receiver = elements[i];
            for (auto d = 0; d < 3; ++d)
                ic[d] = c / lc[d];
            for (ic1[0] = ic[0] - 1; ic1[0] < (ic[0]+1); ic1[0]++) {
                for (ic1[1] = ic[1] - 1; ic1[1] < ic[1] + 1; ic1[1]++) {
                    for (ic1[2] = ic[2] - 1; ic1[2] < ic[2] + 1; ic1[2]++) {
                        /* this is for bounce back, to avoid heap-buffer over/under flow */
                        if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) || (ic1[2] < 0 || ic1[2] >= lc[2])) continue;
                        c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                        j = head[c1];
                        while(j != EMPTY) {
                            if(i < j) {
                                delta = 0.0;
                                source = j < n_elements ? elements[j] : remote_elements[j - n_elements];
                                for (int dim = 0; dim < 3; ++dim)
                                    delta_dim[dim] = receiver.position.at(dim) - source.position.at(dim);
                                for (int dim = 0; dim < 3; ++dim)
                                    delta += (delta_dim[dim] * delta_dim[dim]);
                                Real C_LJ = compute_LJ_scalar<Real>(delta, params->eps_lj, sig2);
                                for (int dim = 0; dim < 3; ++dim) {
                                    receiver.acceleration[dim] += (C_LJ * delta_dim[dim]);
                                }
                                cmplx++;
                            }
                            j = lscl[j];
                        }
                    }
                }
            }
        }
        return cmplx;
    }

    void CLL_compute_forces2d(elements::Element<2> *elements, Integer n_elements, Integer lc[2], Real rc,
                              Integer *lscl, Integer *head, sim_param_t* params) {
        std::stringstream str; str << __func__ << " is not implemented ("<<__FILE__<<":"<<__LINE__ <<")"<< std::endl;
        throw std::runtime_error(str.str());
    }

    void CLL_compute_forces2d(elements::Element<2> *elements, Integer n_elements, const BoundingBox<2>& bbox, Real rc,
                              Integer *lscl, Integer *head, sim_param_t* params) {
        std::stringstream str; str << __func__ << " is not implemented ("<<__FILE__<<":"<<__LINE__ <<")"<< std::endl;
        throw std::runtime_error(str.str());
    }

    template<int N>
    Integer CLL_compute_forces(elements::Element<N> *elements, Integer n_elements,
                            const elements::Element<N> *remote_elements, Integer remote_n_elements,
                            const BoundingBox<N>& bbox, Real rc,
                            const Integer *lscl, const Integer *head,
                            const sim_param_t* params) {
        if constexpr(N==3) {
            return CLL_compute_forces3d(elements, n_elements, remote_elements, remote_n_elements, bbox, rc, lscl, head, params);
        }else {
            CLL_compute_forces2d(elements,n_elements, bbox, rc, lscl, head, params);
            return 0;
        }
    }

    template<int N>
    void CLL_compute_forces(elements::Element<N> *elements, Integer n_elements,
                            const elements::Element<N> *remote_elements, Integer remote_n_elements,
                            const Integer lc[N], Real rc,
                            const Integer *lscl, const Integer *head,
                            const sim_param_t* params) {
        if constexpr(N==3) {
            CLL_compute_forces3d(elements, n_elements, remote_elements, remote_n_elements, lc, rc, lscl, head, params);
        }else {
            CLL_compute_forces2d(elements,n_elements, lc, rc, lscl, head, params);
        }
    }
}

using Complexity        = Integer;
using Time = double;

namespace lj {

    template<int N>
    std::tuple<::Complexity, ::Time, ::Time> compute_one_step (
            MESH_DATA<N> *mesh_data,
            std::vector<Integer> *lscl,             //the particle linked list
            std::vector<Integer> *head,             //the cell starting point
            Zoltan_Struct *load_balancer,           //load balancing structure
            const CommunicationDatatype &datatype,  //structure holding the datatypes //TODO replace by mpi datatype
            sim_param_t *params,                    //simulation parameters
            const MPI_Comm comm,                    //mpi communicator for workers
            const int step = -1)                    //current step to compute (used for debugging purposes)
    {
        elements::BoundingBox<N> bbox;

        int received, sent;
        Real cut_off_radius = params->rc; // cut_off

        const Real dt = params->dt;

        // update local ids
        const size_t nb_elements = mesh_data->els.size();
        for (size_t i = 0; i < nb_elements; ++i) mesh_data->els[i].lid = i;

        START_TIMER(comm_time);
        auto remote_el = zoltan_exchange_data<N>(mesh_data->els, load_balancer, datatype, comm, received, sent, cut_off_radius);
        END_TIMER(comm_time);

        START_TIMER(comp_time);
        bbox = elements::get_bounding_box<N>(mesh_data->els, remote_el, params->rc);
        const auto n_cells = elements::get_total_cell_number<N>(bbox, params->rc);

        if(head->size() < n_cells){
            head->resize(n_cells);
        }
        if((mesh_data->els.size()+remote_el.size()) > lscl->size()) {
            lscl->resize(mesh_data->els.size()+remote_el.size());
        }

        algorithm::CLL_init<N>(mesh_data->els.data(), nb_elements, remote_el.data(), remote_el.size(), bbox, cut_off_radius, lscl->data(), head->data());
        Complexity cmplx = algorithm::CLL_compute_forces<N>(mesh_data->els.data(), nb_elements, remote_el.data(), remote_el.size(), bbox, cut_off_radius, lscl->data(), head->data(), params);

        leapfrog2(dt, mesh_data->els);
        leapfrog1(dt, mesh_data->els, cut_off_radius);
        apply_reflect(mesh_data->els, params->simsize);

        END_TIMER(comp_time);
        return {cmplx, comp_time, comm_time};
    };


}
#endif //NBMPI_LJPOTENTIAL_HPP
