//
// Created by xetql on 20.09.18.
//
#include "../includes/initial_conditions.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/params.hpp"

int main(int argc, char** argv){
    constexpr int DIMENSION = 3;
    sim_param_t params;
    int rank, nproc, dim;
    float ver;
    MESH_DATA<DIMENSION> original_data, mesh_data;

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    std::shared_ptr<initial_condition::lennard_jones::RejectionCondition<DIMENSION>> condition;
    const int MAX_TRIAL = 100000;
    int NB_CLUSTERS;
    std::vector<int> clusters;
    using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;
    std::queue<ElementGeneratorCfg> elements_generators;
    switch (params.particle_init_conf) {
        case 1: //uniformly distributed
            condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                    &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
            );
            elements_generators.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(
                            params.seed, MAX_TRIAL), params.npart));
            break;
        case 2: //Half full half empty
            condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                    &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
            );
            elements_generators.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                            params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
            break;
        case 3: //Wall of particle
            condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                    &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
            );
            elements_generators.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::ParticleWallElementsGenerator<DIMENSION>>(
                            params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
            break;
        case 4: //cluster(s)
            condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                    &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
            );
            NB_CLUSTERS = 1;
            clusters.resize(NB_CLUSTERS);
            std::fill(clusters.begin(), clusters.end(), params.npart);
            elements_generators.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                            clusters, params.seed, MAX_TRIAL), params.npart));
            break;
        case 5: //custom various density
            condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                    &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
            );
            NB_CLUSTERS = 2;
            clusters.resize(NB_CLUSTERS);
            std::fill(clusters.begin(), clusters.end(), params.npart / 4);
            elements_generators.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                            clusters, params.seed, MAX_TRIAL), params.npart / 4));
            elements_generators.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(
                            params.simsize / 10, false, params.seed, MAX_TRIAL), 3 * params.npart / 4));
            break;
        case 6: //custom various density
            condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(
                    &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                            params.simsize, params.simsize, params.simsize
            );
            NB_CLUSTERS = 1;
            clusters.resize(NB_CLUSTERS);
            std::fill(clusters.begin(), clusters.end(), params.npart);
            elements_generators.push(std::make_pair(
                    std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(
                            clusters, params.seed, MAX_TRIAL), params.npart));
            break;
        default:
            throw std::runtime_error("Unknown particle distribution.");
    }
    while (!elements_generators.empty()) {
        ElementGeneratorCfg el_gen = elements_generators.front();
        el_gen.first->generate_elements(mesh_data.els, el_gen.second, condition);
        elements_generators.pop();
        std::cout << el_gen.second << std::endl;
    }

    elements::export_to_file<DIMENSION>(std::to_string(params.npart) + "-"+ std::to_string(params.particle_init_conf) + ".particles", mesh_data.els);
    mesh_data.els = {};

    return 0;
}