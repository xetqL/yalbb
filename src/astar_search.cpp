//
// Created by xetql on 23.05.18.
//

#include <string>
#include <mpi.h>
#include <random>

#include <zoltan.h>
#include "../includes/runners/branch_and_bound.hpp"
#include "../includes/initial_conditions.hpp"

int main(int argc, char **argv) {
    constexpr int DIMENSION = 3;
    sim_param_t params;
    FILE *fp = NULL;
    int rank, nproc, dim;
    float ver;
    MESH_DATA<DIMENSION> mesh_data;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (get_params(argc, argv, &params) != 0) {
        MPI_Finalize();
        return -1;
    }

    params.world_size = nproc;

    if (rank == 0 && params.record) {
        fp = fopen(params.fname, "w");
    }

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    partitioning::CommunicationDatatype datatype = elements::register_datatype<DIMENSION>();

    int rc = Zoltan_Initialize(argc, argv, &ver);
    if (rc != ZOLTAN_OK) {
        MPI_Finalize();
        exit(0);
    }

    if(rank == 0) {
        auto condition = std::make_shared<initial_condition::lennard_jones::RejectionCondition<DIMENSION>>(&(mesh_data.els),
                                                                                  params.sig_lj,
                                                                                  params.sig_lj*params.sig_lj,
                                                                                  params.T0,
                                                                                  0, 0, 0,
                                                                                  params.simsize,
                                                                                  params.simsize,
                                                                                  params.simsize);
        const int MAX_TRIAL = 100000;
        int NB_CLUSTERS;
        std::vector<int> clusters;
        using ElementGeneratorCfg = std::pair<std::shared_ptr<initial_condition::RandomElementsGenerator<DIMENSION>>, int>;
        std::queue<ElementGeneratorCfg> elements_generators;
        switch(params.particle_init_conf) {
            case 1: //uniformly distributed
                elements_generators.push(std::make_pair(std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(params.seed, MAX_TRIAL), params.npart));
                break;
            case 2: //Half full half empty
                elements_generators.push(std::make_pair(std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                break;
            case 3: //Wall of particle
                elements_generators.push(std::make_pair(std::make_shared<initial_condition::lennard_jones::ParticleWallElementsGenerator<DIMENSION>>(params.simsize / 2, false, params.seed, MAX_TRIAL), params.npart));
                break;
            case 4: //cluster(s)
                NB_CLUSTERS = 1;
                clusters.resize(NB_CLUSTERS);
                std::fill(clusters.begin(), clusters.end(), params.npart);
                elements_generators.push(std::make_pair(std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(clusters, params.seed, MAX_TRIAL), params.npart));
                break;
            case 5: //custom various density
                NB_CLUSTERS = 2;
                clusters.resize(NB_CLUSTERS);
                std::fill(clusters.begin(), clusters.end(), params.npart / 4);
                elements_generators.push(std::make_pair(std::make_shared<initial_condition::lennard_jones::RandomElementsInNClustersGenerator<DIMENSION>>(clusters, params.seed, MAX_TRIAL), params.npart / 4));
                elements_generators.push(std::make_pair(std::make_shared<initial_condition::lennard_jones::HalfLoadedRandomElementsGenerator<DIMENSION>>(params.simsize / 10, false, params.seed, MAX_TRIAL), 3*params.npart / 4));
                break;
            default:
                elements_generators.push(std::make_pair(std::make_shared<initial_condition::lennard_jones::UniformRandomElementsGenerator<DIMENSION>>(params.seed, MAX_TRIAL), params.npart));
        }
        while(elements_generators.size() > 0) {
            ElementGeneratorCfg el_gen = elements_generators.front();
            el_gen.first->generate_elements(mesh_data.els, el_gen.second, condition);
            elements_generators.pop();
            std::cout << el_gen.second << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << "==============================================" << std::endl;
        std::cout << "= Simulation is starting now...                 " << std::endl;
        std::cout << "= Parameters: " << std::endl;
        std::cout << "= Particles: " << params.npart << std::endl;
        std::cout << "= Seed: " << params.seed << std::endl;
        std::cout << "= PEs: " << params.world_size << std::endl;
        std::cout << "= Simulation size: " << params.simsize << std::endl;
        std::cout << "= Number of time-steps: " << params.nframes * params.npframe << std::endl;
        std::cout << "= Initial conditions: " << std::endl;
        std::cout << "= SIG:" << params.sig_lj << std::endl;
        std::cout << "= EPS:  " << params.eps_lj << std::endl;
        std::cout << "= Borders: collisions " << std::endl;
        std::cout << "= Gravity:  " << params.G << std::endl;
        std::cout << "= Temperature: " << params.T0 << std::endl;
        std::cout << "==============================================" << std::endl;
    }

    auto zz = zoltan_create_wrapper();

    zoltan_fn_init<DIMENSION>(zz, &mesh_data);

    rc = Zoltan_LB_Partition(zz,                 // input (all remaining fields are output)
                             &changes,           // 1 if partitioning was changed, 0 otherwise
                             &numGidEntries,     // Number of integers used for a global ID
                             &numLidEntries,     // Number of integers used for a local ID
                             &numImport,         // Number of vertices to be sent to me
                             &importGlobalGids,  // Global IDs of vertices to be sent to me
                             &importLocalGids,   // Local IDs of vertices to be sent to me
                             &importProcs,       // Process rank for source of each incoming vertex
                             &importToPart,      // New partition for each incoming vertex
                             &numExport,         // Number of vertices I must send to other processes
                             &exportGlobalGids,  // Global IDs of the vertices I must send
                             &exportLocalGids,   // Local IDs of the vertices I must send
                             &exportProcs,       // Process to which I send each of the vertices
                             &exportToPart);     // Partition to which each vertex will belong
    double xmin, ymin, zmin, xmax, ymax, zmax;

    std::vector<partitioning::geometric::Domain<DIMENSION>> domain_boundaries(nproc);

    for (int part = 0; part < nproc; ++part) {
        Zoltan_RCB_Box(zz, part, &dim, &xmin, &ymin, &zmin, &xmax, &ymax, &zmax);
        auto domain = partitioning::geometric::borders_to_domain<DIMENSION>(xmin, ymin, zmin,
                                                                            xmax, ymax, zmax, params.simsize);
        domain_boundaries[part] = domain;
    }

    load_balancing::geometric::migrate_zoltan<DIMENSION>(mesh_data.els, numImport, numExport,
                                                         exportProcs, exportGlobalGids, datatype, MPI_COMM_WORLD);

    if(!rank) std::cout << "Standard implementation of A*" << std::endl;
    auto res = Astar_runner<DIMENSION>(&mesh_data, zz, &params, MPI_COMM_WORLD);

    std::ofstream dataset;
    const std::string DATASET_FILENAME = "lj_dataset-" + std::to_string(params.seed) +
                                         "-" + std::to_string(params.nframes) + "x" + std::to_string(params.npframe) +
                                         "-" + std::to_string(params.world_size) +
                                         "-" + std::to_string(params.npart) +
                                         "-" + std::to_string((params.T0)) +
                                         "-" + std::to_string((params.G)) +
                                         "-" + std::to_string((params.simsize)) +
                                         "-" + std::to_string((params.eps_lj)) +
                                         "-" + std::to_string((params.sig_lj)) +
                                         "-" + std::to_string(params.dt) + ".data";
    std::ofstream fsolution;
    const std::string SOLUTION_FILENAME = "lj_solution-" + std::to_string(params.seed) +
                                         "-" + std::to_string(params.nframes) + "x" + std::to_string(params.npframe) +
                                         "-" + std::to_string(params.world_size) +
                                         "-" + std::to_string(params.npart) +
                                         "-" + std::to_string((params.T0)) +
                                         "-" + std::to_string((params.G)) +
                                         "-" + std::to_string((params.simsize)) +
                                         "-" + std::to_string((params.eps_lj)) +
                                         "-" + std::to_string((params.sig_lj)) +
                                         "-" + std::to_string(params.dt) + ".data";
    auto sol = res.at(0);

    std::ofstream lb_file, metric_file, frame_file;
    if(!rank){
        std::string mkdir_cmd = "mkdir -p data/time-series/"+std::to_string(params.seed);
        system(mkdir_cmd.c_str());
    }
    SimpleCSVFormatter frame_formater(',');
    std::vector<elements::Element<DIMENSION>> recv_buf(params.npart);
    int i = 0;

    for(auto const& solution : res) {
        double total_time = 0.0;
        metric::io::write_dataset(dataset, DATASET_FILENAME, solution, rank, (*(std::next(solution.end(), -1)))->cost());
        metric::io::write_solution( fsolution, std::to_string(i)+"-"+SOLUTION_FILENAME, solution, rank);
        i++;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Zoltan_LB_Free_Part(&importGlobalGids, &importLocalGids, &importProcs, &importToPart);
    Zoltan_LB_Free_Part(&exportGlobalGids, &exportLocalGids, &exportProcs, &exportToPart);
    Zoltan_Destroy(&zz);

    if (fp) fclose(fp);

    MPI_Finalize();
    return 0;
}
