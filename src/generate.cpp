//
// Created by xetql on 20.09.18.
//
#include <string>
#include <mpi.h>
#include <random>

#include <zoltan.h>

#include "../includes/runners/simulator.hpp"
#include "../includes/initial_conditions.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/params.hpp"
#include "../includes/runners/branch_and_bound.hpp"
#include "../includes/generate.hpp"

int main(int argc, char** argv) {
    constexpr int DIMENSION = 3;
    sim_param_t params;
    int rank, nproc, dim;
    float ver;
    MESH_DATA<DIMENSION> original_data, mesh_data;

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    std::shared_ptr<initial_condition::lj::RejectionCondition<DIMENSION>> condition;
    const int MAX_TRIAL = 100000;
    int NB_CLUSTERS;
    std::vector<int> clusters;
    using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;
    std::queue<ElementGeneratorCfg> elements_generators;
    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
            &(mesh_data.els), params.sig_lj, 6.25 * params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
            params.simsize, params.simsize, params.simsize
    );
    init_generator<DIMENSION>(elements_generators, condition, params.particle_init_conf, &params);
    std::cout << "Configuration setup: done!" << std::endl;
    int cfg_idx = 0;
    while (!elements_generators.empty()) {
        std::cout << "Starting generation of particles with configuration ("
                  << cfg_idx<<"/"<<elements_generators.size()<<") ..." <<std::endl;
        ElementGeneratorCfg el_gen = elements_generators.front();
        el_gen.first->generate_elements(mesh_data.els, el_gen.second, condition);
        elements_generators.pop();
        std::cout << el_gen.second <<"/"<< params.npart << " particles generated." << std::endl;
        cfg_idx++;
    }

    std::cout << "Generation done!\nStarting exportation..." <<std::endl;

    elements::export_to_file<DIMENSION>(std::to_string(params.npart) + "-" +
                                        std::to_string(params.particle_init_conf) + "-" +
                                        std::to_string(params.simsize) + ".particles", mesh_data.els);
    std::cout << "Done!" << std::endl;
    return 0;
}