//
// Created by xetql on 04.01.18.
//

#ifndef NBMPI_LJPOTENTIAL_HPP
#define NBMPI_LJPOTENTIAL_HPP

#include <cmath>
#include <liblj/params.hpp>
#include <vector>
#include <unordered_map>

#include "physics.hpp"
template<typename RealType>
void create_cell_linkedlist(
        const int nsub, /* number of subdomain per row*/
        const double lsub, /* width of subdomain */
        const int nparticle, /* number of particles */
        const RealType* x, /* particle location */
        int* particleslklist, /* particle linked list */
        int* head) throw() /* head of particle linked list */{
    int cell_of_particle;
    for (int icell = 0; icell < nsub * nsub; icell++) head[icell] = -1;
    for (int iparticle = 0; iparticle < nparticle; iparticle++) {
        cell_of_particle = (int) (std::floor(x[2 * iparticle] / lsub)) + nsub * (std::floor(x[2 * iparticle + 1] / lsub));
        if (cell_of_particle >= (nsub * nsub)) throw std::runtime_error("Particle "+std::to_string(cell_of_particle) + " is out of domain");
        particleslklist[iparticle] = head[cell_of_particle];
        head[cell_of_particle] = iparticle;
    }
}

template<int N>
void create_cell_linkedlist(
        const int nsub, /* number of subdomain per row*/
        const double lsub, /* width of subdomain */
        const std::vector<elements::Element<N>> &particles, /* particle location */
        std::unordered_map<int, int> &particleslklist, /* particle linked list */
        int* head) throw() /* head of particle linked list */{
    int cell_of_particle;
    for (int icell = 0; icell < nsub * nsub; icell++) head[icell] = -1;
    for (auto const &particle : particles) {
        cell_of_particle = (int) (std::floor(particle.position.at(0) / lsub)) + nsub * (std::floor(particle.position.at(1) / lsub));
        if (cell_of_particle >= (nsub * nsub)) throw std::runtime_error("Particle "+std::to_string(cell_of_particle) + " is out of domain");
        particleslklist[particle.identifier] = head[cell_of_particle];
        head[cell_of_particle] = particle.identifier;
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
        std::vector<RealType>& Flocal,
        const std::vector<int> &head, /* Starting neighboring particles */
        const std::vector<int> &plklist, /* Neighboring subcell particles organized as a linkedlist */
        const sim_param_t* params) /* Simulation parameters */ {

    RealType g    = (RealType) params->G;
    RealType eps  = (RealType) params->eps_lj;
    RealType sig  = (RealType) params->sig_lj;
    RealType sig2 = (RealType) sig*sig;

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
        linearcellidx = (int) (std::floor(x[2 * particleidx] / lsub)) + M * (std::floor(x[2 * particleidx + 1] / lsub));

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
        std::vector<RealType>& Flocal,
        const sim_param_t* params) {

    int nlocal = iend - istart;
    RealType g    = (RealType) params->G;
    RealType eps  = (RealType) params->eps_lj;
    RealType sig  = (RealType) params->sig_lj;
    RealType sig2 = sig*sig;

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
                RealType C_LJ = compute_LJ_scalar(dx*dx + dy*dy, eps, sig2);
                Flocal[2 * ii + 0] += (C_LJ * dx);
                Flocal[2 * ii + 1] += (C_LJ * dy);
            }
        }
    }
}

template<int N>
void compute_forces(
        const std::vector<elements::Element<N>> &distant_elements,
        std::vector<elements::Element<N>> &local_elements,
        const sim_param_t* params) {

    double g    = (double) params->G;
    double eps  = (double) params->eps_lj;
    double sig  = (double) params->sig_lj;
    double sig2 = sig*sig;
    // each particle MUST checks the local particles and the particles from neighboring PE
    std::vector<elements::Element<N>> all_elements_to_check(distant_elements.size() + local_elements.size());
    std::copy(distant_elements.begin(), distant_elements.end(), std::front_inserter(all_elements_to_check));
    std::copy(  local_elements.begin(),   local_elements.end(), std::front_inserter(all_elements_to_check));

    for(auto &el : local_elements){
        std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
        el.acceleration.at(1) = -g;
    }

    /* Particle-particle interactions (Lennard-Jones) */
    for(auto &force_recepter : local_elements){
        for(auto &force_source : all_elements_to_check){
            if(force_source.identifier != force_recepter.identifier){
                double dx = force_source.position.at(0) - force_recepter.position.at(0);
                double dy = force_source.position.at(1) - force_recepter.position.at(1);
                double C_LJ = compute_LJ_scalar(dx*dx + dy*dy, eps, sig2);
                force_recepter.acceleraton[0] += (C_LJ * dx);
                force_recepter.acceleraton[1] += (C_LJ * dy);
            }
        }
    }
}


template<int N>
void compute_forces(
        const int n, /* Number of particle */
        const int M, /* Number of subcell in a row  */
        const float lsub, /* length of a cell */
        const std::vector<elements::Element<N>> &distant_elements,
        std::vector<elements::Element<N>> &local_elements,
        const std::vector<int> &head, /* Starting neighboring particles */
        const std::unordered_map<int, int> &plklist, /* Neighboring subcell particles organized as a linkedlist */
        const sim_param_t* params) /* Simulation parameters */ {

    double g    = (double) params->G;
    double eps  = (double) params->eps_lj;
    double sig  = (double) params->sig_lj;
    double sig2 = sig*sig;

    // each particle MUST checks the local particles and the particles from neighboring PE
    //std::vector<elements::Element<N>> all_elements_to_check(distant_elements.size() + local_elements.size());

    std::unordered_map<int, elements::Element<N>> element_map;
    for(auto &el : distant_elements){
        element_map.insert(el.identifier, el);
    }
    for(auto &el : local_elements){
        element_map.insert(el.identifier, el);
    }
    for(auto &el : local_elements){
        std::fill(el.acceleration.begin(), el.acceleration.end(), 0.0); //fill all dimension with zero
        el.acceleration.at(1) = -g;
    }

    int linearcellidx;
    int nlinearcellidx;
    int nparticleidx;

    // process only the particle we are interested in
    for (auto &force_recepter : local_elements) {

        // find cell from particle position
        linearcellidx = (int) (std::floor(force_recepter.position.at(0) / lsub)) + M * (std::floor(force_recepter.position.at(1) / lsub));

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
                    if (force_recepter.identifier != nparticleidx) {
                        double dx = force_recepter.position.at(0) - element_map.at(nparticleidx).position.at(0);
                        double dy = force_recepter.position.at(1) - element_map.at(nparticleidx).position.at(1);

                        double C_LJ = compute_LJ_scalar(dx * dx + dy * dy, eps, sig2);

                        force_recepter.acceleration[0] += (C_LJ * dx);
                        force_recepter.acceleration[1] += (C_LJ * dy);
                    }
                    nparticleidx = plklist.at(nparticleidx);
                }
            }
        }
    }
}


#endif //NBMPI_LJPOTENTIAL_HPP
