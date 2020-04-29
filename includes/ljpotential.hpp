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
#include "parallel_utils.hpp"

auto MPI_TIME       = MPI_DOUBLE;
auto MPI_COMPLEXITY = MPI_LONG_LONG;

namespace lj {
    namespace {
        std::vector<Real> acc;
    }

    template<int N, class T, class SetPosFunc, class SetVelFunc, class GetForceFunc>
    Complexity compute_one_step (
            std::vector<T>&        elements,
            const std::vector<T>& remote_el,
            SetPosFunc getPosPtrFunc,                  // function to get force of an entity
            SetVelFunc getVelPtrFunc,                  // function to get force of an entity
            std::vector<Integer> *head,                // the cell starting point
            std::vector<Integer> *lscl,                // the particle linked list
            BoundingBox<N>& bbox,                      // bounding box of particles
            GetForceFunc getForceFunc,                 // function to compute force between entities
            const Borders& borders,                    // bordering cells and neighboring processors
            const sim_param_t *params) {               // simulation parameters

        const Real cut_off_radius = params->rc; // cut_off
        const Real dt = params->dt;
        const size_t nb_elements = elements.size();

        if(const auto n_cells = get_total_cell_number<N>(bbox, params->rc); head->size() < n_cells) {
            head->resize(n_cells);
        }
        if(const auto n_force_elements = N*elements.size(); acc.size() < n_force_elements) {
            acc.resize(N*n_force_elements);
        }
        if(const auto n_particles = elements.size()+remote_el.size();  lscl->size() < n_particles) {
            lscl->resize(n_particles);
        }

        algorithm::CLL_init<N, T>({ {elements.data(), nb_elements}, {elements.data(), remote_el.size()} }, getPosPtrFunc, bbox, cut_off_radius, head, lscl);

        Complexity cmplx = algorithm::CLL_compute_forces<N, T>(&acc, elements, remote_el, getPosPtrFunc, bbox, cut_off_radius, head, lscl, getForceFunc);

        leapfrog2<N, T>(dt, acc, elements, getVelPtrFunc);
        leapfrog1<N, T>(dt, cut_off_radius, acc, elements, getPosPtrFunc, getVelPtrFunc);
        apply_reflect<N, T>(elements, params->simsize, getPosPtrFunc, getVelPtrFunc);

        return cmplx;
    };
}
#endif //NBMPI_LJPOTENTIAL_HPP
