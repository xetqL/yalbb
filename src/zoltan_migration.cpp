#include <string>
#include <mpi.h>
#include <random>
#include <zoltan.h>
#include "../includes/zoltan_fn.hpp"
#include "../includes/initial_conditions.hpp"
#include "../includes/generate.hpp"
#include "../includes/nbody_io.hpp"
#include "../includes/params.hpp"
#include "../includes/unloading_model.hpp"
int main(int argc, char** argv) {
    constexpr int DIMENSION = 3;
    sim_param_t params;
    FILE* fp = NULL;
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
    std::cout << nproc << std::endl;
    params.world_size = nproc;

    if (rank == 0 && params.record) {
        fp = fopen(params.fname, "w");
    }

    int changes, numGidEntries, numLidEntries, numImport, numExport;
    ZOLTAN_ID_PTR importGlobalGids, importLocalGids, exportGlobalGids, exportLocalGids;
    int *importProcs, *importToPart, *exportProcs, *exportToPart;

    CommunicationDatatype datatype = elements::register_datatype<DIMENSION>();

    int rc = Zoltan_Initialize(argc, argv, &ver);
    if(rc != ZOLTAN_OK){
        MPI_Finalize();
        exit(0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////START PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (rank == 0) {
        const std::string IMPORT_FILENAME
                = std::to_string(params.npart) + "-" +
                  std::to_string(params.particle_init_conf) + "-" +
                  std::to_string(params.simsize) + ".particles";
        if(file_exists(IMPORT_FILENAME)) {
            std::cout << "importing from file ..." << std::endl;
            elements::import_from_file<DIMENSION, Real >(IMPORT_FILENAME, mesh_data.els);
        } else {
            std::cout << "Generating data ..." << std::endl;
            std::shared_ptr<initial_condition::lj::RejectionCondition<DIMENSION>>
                    condition = std::make_shared<initial_condition::lj::RejectionCondition<DIMENSION>>(
                    &(mesh_data.els), params.sig_lj, params.sig_lj * params.sig_lj, params.T0, 0, 0, 0,
                    params.simsize, params.simsize, params.simsize);
            auto elements_generators = init_generator(condition, params.particle_init_conf, &params);
            while (!elements_generators.empty()) {
                auto el_gen = elements_generators.front();
                el_gen.first->generate_elements(mesh_data.els, el_gen.second, condition);
                elements_generators.pop();
                std::cout << el_gen.second << std::endl;
            }
        }
        std::cout << "Done !" << std::endl;
    }
    MESH_DATA<DIMENSION> bottom_data = mesh_data;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////FINISHED PARITCLE INITIALIZATION///////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    MPI_Comm bottom = MPI_COMM_WORLD;
    auto zz = zoltan_create_wrapper(true, bottom);
    zoltan_load_balance(&bottom_data, zz, datatype, bottom, true);
    MPI_Barrier(bottom);
    if(rank == 0) std::cout << "Data are balanced" << std::endl;
    std::for_each(bottom_data.els.cbegin(),bottom_data.els.cend(), [&rank](auto el){std::cout << rank <<" " << el << std::endl;});

    MPI_Comm top;

    double my_slope = 0.;
    if(rank == 0) my_slope = 0.3; // 0 is not in top comm
    if(rank == 1) my_slope = 0.3; // 0 is not in top comm
    std::vector<int> increasing_cpus;
    load_balancing::esoteric::get_communicator(my_slope, rank, bottom, &increasing_cpus, &top);
    std::for_each(increasing_cpus.cbegin(),increasing_cpus.cend(), [&rank](auto el){std::cout << rank <<" " << el << std::endl;});


    MESH_DATA<DIMENSION> top_data;
    Zoltan_Struct* zz_top = load_balancing::esoteric::divide_data_into_top_bottom2(&bottom_data.els, &top_data.els, increasing_cpus, datatype,   bottom);

    MPI_Barrier(bottom);
    if(rank == 0) std::cout << "After the SPLIT" << std::endl;
    std::for_each(bottom_data.els.cbegin(),bottom_data.els.cend(), [&rank](auto el){std::cout <<"B "<< rank <<" " << el << std::endl;});
    std::for_each(top_data.els.cbegin(),top_data.els.cend(), [&rank](auto el){std::cout <<"T "<< rank <<" " << el << std::endl;});

    sleep(1);

    double coords[3] = {1,2,3};
    int PE;
    MPI_Barrier(bottom);
    if(rank == 0) std::cout << "After the MIGRATION" << std::endl;
    if(rank == 2) top_data.els[0].position = {0.290078,0.383462,0.618015};
    load_balancing::esoteric::migrate(&bottom_data.els, &top_data.els, zz, zz_top, bottom, increasing_cpus, datatype);

    MPI_Barrier(bottom);
    std::for_each(bottom_data.els.cbegin(),bottom_data.els.cend(), [&rank](auto el){std::cout <<">B "<< rank <<" " << el << std::endl;});
    std::for_each(top_data.els.cbegin(),top_data.els.cend(), [&rank](auto el){std::cout <<">T "<< rank <<" " << el << std::endl;});
    MPI_Barrier(bottom);
    auto remote = load_balancing::esoteric::exchange(&bottom_data.els, &top_data.els, zz, zz_top, bottom, increasing_cpus, datatype);
    MPI_Barrier(bottom);

    sleep(1);
    std::for_each(remote.cbegin(),remote.cend(), [&rank](auto el){std::cout <<"Remote> "<< rank <<" " << el << std::endl;});

    MPI_Finalize();
    return 0;
}
