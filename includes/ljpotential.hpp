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
#include "geometric_load_balancer.hpp"

namespace algorithm {

    constexpr Integer EMPTY = -1;

    template<int N>
    void CLL_init(const elements::Element<N> *elements, Integer n_elements, Integer lc[N], Real rc, Integer *lscl,
                  Integer *head) {
        Integer lcxyz = lc[0] * lc[1];
        if constexpr (N == 3) {
            lcxyz *= lc[2];
        }

        Integer c;
        for (size_t i = 0; i < lcxyz; ++i) head[i] = EMPTY;

        for (size_t i = 0; i < n_elements; ++i) {
            c = position_to_cell<N>(elements[i].position, rc, lc[0], lc[1]);
            lscl[i] = head[c];
            head[c] = i;
        }
    }
    template<int N>
    inline void CLL_append(Integer i, Integer cell, const elements::Element<N>& elements, Integer *lscl, Integer *head) {
        lscl[i] = head[cell];
        head[cell] = i;
    }

    void CLL_compute_forces3d(elements::Element<3> *elements, Integer n_elements, Integer lc[3], Real rc,
                              Integer *lscl, Integer *head, sim_param_t* params) {
        Integer lcxyz = lc[0] * lc[1] * lc[2];
        std::array<Real, 3> delta_dim;
        Real delta, ddim;
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
                        if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) || (ic1[2] < 0 || ic1[2] >= lc[2])) continue;
                        c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                        j = head[c1];
                        while(j != EMPTY){
                            if(i < j){
                                delta = 0.0;
                                source = elements[j];
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

    void CLL_compute_forces2d(elements::Element<2> *elements, Integer n_elements, Integer lc[2], Real rc,
                              Integer *lscl, Integer *head, sim_param_t* params) {
    }

    template<int N>
    void CLL_compute_forces(elements::Element<N> *elements, Integer n_elements, Integer lc[N], Real rc,
                            Integer *lscl, Integer *head, sim_param_t* params) {
        if constexpr(N==3){
            CLL_compute_forces3d(elements,n_elements,lc, rc, lscl, head, params);
        }else{
            CLL_compute_forces2d(elements,n_elements,lc, rc, lscl, head, params);
        }
    }

}

namespace lennard_jones {

    template<int N, class MapType>
    int create_cell_linkedlist(
            const long long nsub, /* number of subdomain per row*/
            const elements::ElementRealType lsub, /* width of subdomain */
            const std::vector<elements::Element<N>> &local_elements, /* particle location */
            const std::vector<elements::Element<N>> &remote_elements, /* particle location */
            MapType &plist) {
        long long cell_of_particle;

        plist.clear();
        size_t local_size = local_elements.size(),
                remote_size = remote_elements.size(),
                total_size = local_size + remote_size;
        size_t cpt = 0;
        while (cpt < total_size) {
            auto const &particle = cpt >= local_size ? remote_elements[cpt - local_size] : local_elements[cpt];
            cell_of_particle = position_to_cell<N>(particle.position, lsub, nsub, nsub);

            if (cell_of_particle >= (std::pow(nsub, N)) || cell_of_particle < 0) { //a particle is out!
                std::cout << particle << std::endl;
                std::cout << lsub << std::endl;
                std::cout << nsub << std::endl;
                std::cout << (long long) std::pow(nsub, N) << std::endl;
                std::cout << cell_of_particle << std::endl;
            } else {
                if (plist.find(cell_of_particle) == plist.end())
                    plist[cell_of_particle] = std::make_unique<std::vector<elements::Element<N>>>();
                plist[cell_of_particle]->push_back(particle);
            }
            cpt++;
        }
        return 0;
    }

    template<int N>
    int compute_forces(
            const long long M, /* Number of subcell in a row  */
            const elements::ElementRealType lsub, /* length of a cell */
            std::vector<elements::Element<N>> &local_elements,
            const std::vector<elements::Element<N>> &remote_elements,
            const std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<N>>>> &plist,
            const sim_param_t *params) noexcept {

        elements::ElementRealType g = dto<elements::ElementRealType>(params->G);
        elements::ElementRealType eps = dto<elements::ElementRealType>(params->eps_lj);
        elements::ElementRealType sig = dto<elements::ElementRealType>(params->sig_lj);
        elements::ElementRealType sig2 = sig * sig;

        size_t complexity = local_elements.size();
        // each particle MUST checks the local particles and the particles from neighboring PE
        std::unordered_map<int, elements::Element<N>> element_map;

        for (auto &el : local_elements) {
            std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
            el.acceleration.at(N - 1) = -g;
        }

        long long linearcellidx, nlinearcellidx;
        // process only the particle we are interested in
        for (auto &force_recepter : local_elements) { //O(n)
            // find cell from particle position
            linearcellidx = position_to_cell<N>(force_recepter.position, lsub, M, M);
            // convert linear index to grid position
            int xcellidx, ycellidx, zcellidx;

            linear_to_grid(linearcellidx, M, M, xcellidx, ycellidx, zcellidx);

            constexpr int zstart = N == 3 ? -1 : 0; // if in 2D, there is only 1 depth
            constexpr int zstop = N == 3 ? 2 : 1; // so, iterate only through [0,1)
            // Explore neighboring cells
            for (int neighborx = -1; neighborx < 2; neighborx++) {
                for (int neighbory = -1; neighbory < 2; neighbory++) {
                    for (int neighborz = zstart; neighborz < zstop; neighborz++) {
                        // Check boundary conditions
                        if (xcellidx + neighborx < 0 || xcellidx + neighborx >= M) continue;
                        if (ycellidx + neighbory < 0 || ycellidx + neighbory >= M) continue;
                        if (zcellidx + neighborz < 0 || zcellidx + neighborz >= M) continue;
                        nlinearcellidx =
                                (xcellidx + neighborx) + M * (ycellidx + neighbory) + M * M * (zcellidx + neighborz);
                        if (plist.find(nlinearcellidx) != plist.end()) {
                            auto el_list = plist.at(nlinearcellidx).get();
                            size_t cell_el_size = el_list->size();
                            for (size_t el_idx = 0; el_idx < cell_el_size; ++el_idx) {
                                auto const &force_source = el_list->at(el_idx);
                                if (force_recepter.gid != force_source.gid) {
                                    complexity++;
                                    std::array<elements::ElementRealType, N> delta_dim;
                                    elements::ElementRealType delta = 0.0;

                                    for (size_t dim = 0; dim < N; ++dim) {
                                        const elements::ElementRealType ddim =
                                                force_source.position.at(dim) - force_recepter.position.at(dim);
                                        delta += (ddim * ddim);
                                        delta_dim[dim] = ddim;
                                    }

                                    elements::ElementRealType C_LJ = compute_LJ_scalar<elements::ElementRealType>(delta,
                                                                                                                  eps,
                                                                                                                  sig2);

                                    for (int dim = 0; dim < N; ++dim) {
                                        force_recepter.acceleration[dim] += (C_LJ * delta_dim[dim]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return complexity;
    }

    template<int N, class MapType>
    void _compute_forces(
            const int M, /* Number of subcell in a row  */
            const float lsub, /* length of a cell */
            std::vector<elements::Element<N>> &local_elements,
            const std::vector<elements::Element<N>> &remote_elements,
            const MapType &plist,
            const sim_param_t *params) /* Simulation parameters */ {

        double g = (double) params->G;
        double eps = (double) params->eps_lj;
        double sig = (double) params->sig_lj;
        double sig2 = sig * sig;

        // each particle MUST checks the local particles and the particles from neighboring PE
        std::unordered_map<int, elements::Element<N>> element_map;

        for (auto &el : local_elements) {
            std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
            el.acceleration.at(1) = -g;
        }

        int linearcellidx;
        int nlinearcellidx;

        // process only the particle we are interested in
        for (auto &force_recepter : local_elements) { //O(n)

            // find cell from particle position
            linearcellidx = (int) (std::floor(force_recepter.position.at(0) / lsub)) +
                            M * (std::floor(force_recepter.position.at(1) / lsub));
            // convert linear index into grid position
            int xcellidx = linearcellidx % M;
            int ycellidx = linearcellidx / M;
            /* Explore neighboring cells */
            for (int neighborx = -1; neighborx < 2; neighborx++) {
                for (int neighbory = -1; neighbory < 2; neighbory++) {

                    /// Check boundary limits
                    if (xcellidx + neighborx < 0 || xcellidx + neighborx >= M) continue;
                    if (ycellidx + neighbory < 0 || ycellidx + neighbory >= M) continue;

                    ////////////////////////////////////////////////////////////////////

                    nlinearcellidx = (xcellidx + neighborx) + M * (ycellidx + neighbory);
                    if (plist.find(nlinearcellidx) != plist.end()) {
                        auto el_list = plist.at(nlinearcellidx).get();
                        size_t cell_el_size = el_list->size();
                        for (size_t el_idx = 0; el_idx < cell_el_size; ++el_idx) {
                            auto const &force_source = el_list->at(el_idx);
                            if (force_recepter.gid != force_source.gid) {
                                double dx = force_source.position.at(0) - force_recepter.position.at(0);
                                double dy = force_source.position.at(1) - force_recepter.position.at(1);
                                double C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);
                                force_recepter.acceleration[0] += (C_LJ * dx);
                                force_recepter.acceleration[1] += (C_LJ * dy);
                            }
                        }
                    }
                }
            }
        }
    }

    template<typename RealType>
    void compute_forces(
            const int n, /* Number of particle */
            const int M, /* Number of subcell in a row  */
            const float lsub, /* length of a cell */
            const std::vector<RealType> x, /* Particles position */
            const int istart,
            const int iend,
            const std::vector<RealType> &xlocal,
            std::vector<RealType> &Flocal,
            const std::vector<int> &head, /* Starting neighboring particles */
            const std::vector<int> &plklist, /* Neighboring subcell particles organized as a linkedlist */
            const sim_param_t *params) /* Simulation parameters */ {

        RealType g = (RealType) params->G;
        RealType eps = (RealType) params->eps_lj;
        RealType sig = (RealType) params->sig_lj;
        RealType sig2 = (RealType) sig * sig;

        /* Global force downward (e.g. gravity) */
        for (int i = 0; i < (iend - istart); ++i) {
            Flocal[2 * i + 0] = 0;
            Flocal[2 * i + 1] = -g;
        }

        int linearcellidx;
        int nlinearcellidx;
        int nparticleidx;

        // process only the particle we are interested in
        for (int particleidx = istart; particleidx < iend; ++particleidx) {

            int ii = particleidx - istart;

            // find cell from particle position
            linearcellidx =
                    (int) (std::floor(x[2 * particleidx] / lsub)) + M * (std::floor(x[2 * particleidx + 1] / lsub));

            // convert linear index to grid position
            int xcellidx = linearcellidx % M;
            int ycellidx = linearcellidx / M;

            /* Explore neighboring cells */
            for (int neighborx = -1; neighborx < 2; neighborx++) {
                for (int neighbory = -1; neighbory < 2; neighbory++) {

                    /* Check boundary conditions */
                    if (xcellidx + neighborx < 0 || xcellidx + neighborx >= M) continue;
                    if (ycellidx + neighbory < 0 || ycellidx + neighbory >= M) continue;

                    nlinearcellidx = (xcellidx + neighborx) + M * (ycellidx + neighbory);

                    nparticleidx = head[nlinearcellidx];
                    while (nparticleidx != -1) {
                        if (particleidx != nparticleidx) {

                            RealType dx = x[2 * nparticleidx + 0] - x[2 * particleidx + 0];
                            RealType dy = x[2 * nparticleidx + 1] - x[2 * particleidx + 1];

                            RealType C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);

                            Flocal[2 * ii + 0] += (C_LJ * dx);
                            Flocal[2 * ii + 1] += (C_LJ * dy);
                        }
                        nparticleidx = plklist[nparticleidx];
                    }
                }
            }
        }
    }

    template<typename RealType>
    void compute_forces(
            const int n,
            const std::vector<RealType> &x,
            const int istart,
            const int iend,
            const std::vector<RealType> &xlocal,
            std::vector<RealType> &Flocal,
            const sim_param_t *params) {

        int nlocal = iend - istart;
        RealType g = (RealType) params->G;
        RealType eps = (RealType) params->eps_lj;
        RealType sig = (RealType) params->sig_lj;
        RealType sig2 = sig * sig;

        /* Global force downward (e.g. gravity) */
        for (int i = 0; i < nlocal; ++i) {
            Flocal[2 * i + 0] = 0;
            Flocal[2 * i + 1] = -g;
        }

        /* Particle-particle interactions (Lennard-Jones) */
        for (int i = istart; i < iend; ++i) {
            int ii = i - istart;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    RealType dx = x[2 * j + 0] - xlocal[2 * ii + 0];
                    RealType dy = x[2 * j + 1] - xlocal[2 * ii + 1];
                    RealType C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);
                    Flocal[2 * ii + 0] += (C_LJ * dx);
                    Flocal[2 * ii + 1] += (C_LJ * dy);
                }
            }
        }
    }

    template<int N>
    void compute_forces(
            std::vector<elements::Element<N>> &local_elements,
            const std::vector<elements::Element<N>> &remote_elements,
            const sim_param_t *params) {

        double g = (double) params->G;
        double eps = (double) params->eps_lj;
        double sig = (double) params->sig_lj;
        double sig2 = sig * sig;

        for (auto &el : local_elements) {
            std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
            el.acceleration.at(1) = -g;//set GRAVITY MOFO YO
        }

        /* Particle-particle interactions (Lennard-Jones) */
        size_t local_size = local_elements.size(), remote_size = remote_elements.size();
        for (auto &force_recepter : local_elements) {
            for (size_t cpt = 0; cpt < local_size + remote_size; ++cpt) {
                auto force_source = cpt >= local_size ? remote_elements[cpt - local_size] : local_elements[cpt];
                if (force_source.identifier != force_recepter.identifier) {
                    double dx = force_source.position.at(0) - force_recepter.position.at(0);
                    double dy = force_source.position.at(1) - force_recepter.position.at(1);
                    double C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);
                    force_recepter.acceleration[0] += (C_LJ * dx);
                    force_recepter.acceleration[1] += (C_LJ * dy);
                }
            }
        }
    }

    template<int N>
    inline std::tuple<int, int, int> compute_one_step(
            MESH_DATA<N> *mesh_data,
            Integer *lscl,
            Integer *head,
            Zoltan_Struct *load_balancer,
            const partitioning::CommunicationDatatype &datatype,
            sim_param_t *params,
            const MPI_Comm comm,
            const int step = -1 /* by default we don't care about the step*/ ) {

        int received, sent;
        Real cut_off_radius = params->rc; // cut_off
        auto cell_per_row = (Integer) std::ceil(params->simsize / cut_off_radius); // number of cell in a row
        Real cell_size = cut_off_radius; //cell size
        const Real dt = params->dt;

        // update local ids
        const size_t nb_elements = mesh_data->els.size();
        for (size_t i = 0; i < nb_elements; ++i) mesh_data->els[i].lid = i;

        auto remote_el = load_balancing::geometric::zoltan_exchange_data<N>(mesh_data->els, load_balancer, datatype,
                                                                            comm, received, sent, cut_off_radius);

        auto n_cells = cell_per_row*cell_per_row*cell_per_row;
        Integer lc[N];
        lc[0] = cell_per_row;
        lc[1] = cell_per_row;
        if constexpr (N==3){
            lc[2] = cell_per_row;
        }

        algorithm::CLL_init<N>(mesh_data->els.data(), nb_elements, lc, cut_off_radius, lscl, head);
        algorithm::CLL_compute_forces<N>(mesh_data->els.data(), nb_elements, lc, cut_off_radius, lscl, head, params);

        leapfrog2(dt, mesh_data->els);
        leapfrog1(dt, mesh_data->els, cut_off_radius);
        apply_reflect(mesh_data->els, params->simsize);

        return std::make_tuple(0, received, sent);
    };

    template<int N>
    inline std::tuple<int, int, int> compute_one_step(
            MESH_DATA<N> *mesh_data,
            std::unordered_map<long long, std::unique_ptr<std::vector<elements::Element<N> > > > &plklist,
            const std::vector<partitioning::geometric::Domain<N>> &domain_boundaries,
            const partitioning::CommunicationDatatype &datatype,
            sim_param_t *params,
            const MPI_Comm comm,
            const int step = -1 /* by default we don't care about the step*/ ) {

        int received, sent;

        elements::ElementRealType cut_off_radius = dto<elements::ElementRealType>(3.2 * params->sig_lj); // cut_off
        auto cell_per_row = (long long) std::ceil(params->simsize / cut_off_radius); // number of cell in a row
        elements::ElementRealType cell_size = cut_off_radius; //cell size
        const elements::ElementRealType dt = params->dt;

        auto remote_el = load_balancing::geometric::__exchange_data<N>(mesh_data->els, domain_boundaries, datatype,
                                                                       comm, received, sent, cell_size);

        // update local ids
        const size_t nb_elements = mesh_data->els.size();
        for (size_t i = 0; i < nb_elements; ++i) mesh_data->els[i].lid = i;

        int err = lennard_jones::create_cell_linkedlist(cell_per_row, cell_size, mesh_data->els, remote_el, plklist);

        if (err) {
            std::cerr << err << std::endl;
            throw std::runtime_error("Particle out of domain");
        }

        int cmplx = lennard_jones::compute_forces(cell_per_row, cell_size, mesh_data->els, remote_el, plklist, params);

        /**!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         * WE HAVE TO REMOVE THIS AFTER TESTS!!!!!
         * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
        /*if(step >= 0) {
            // freeze after T/2 !
            if(step > params->nframes / 2) params->frozen_factor = 0.0;
            else params->frozen_factor = 1.0;

            for (auto &p : mesh_data->els)
                for (int dim = 0; dim < N; ++dim) {
                    p.velocity[dim] *= params->frozen_factor;
                    //////////////////////////////////////////////////////////////////////////////
                    p.acceleration[dim] *= 0.0; //cancel all the forces, /!\ to remove after tests
                    //////////////////////////////////////////////////////////////////////////////
                }
        }*/
        /// IT STOPS HERE

        leapfrog2(dt, mesh_data->els);
        leapfrog1(dt, mesh_data->els);
        apply_reflect(mesh_data->els, params->simsize);

        return std::make_tuple(cmplx, received, sent);
    };

}
#endif //NBMPI_LJPOTENTIAL_HPP
