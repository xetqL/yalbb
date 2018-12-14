//
// Created by xetql on 12/11/18.
//

#ifndef NBMPI_UNLOADING_MODEL_HPP
#define NBMPI_UNLOADING_MODEL_HPP

#include <vector>
#include "spatial_elements.hpp"
#include "zoltan_fn.hpp"

namespace load_balancing {
    namespace esoteric {
        enum MIGRATION_TYPE {BOTTOM, TOP};
        const float SLOPE_THRESHOLD = 0.2f;

        /**
         * get communicators for the new LB paradigm
         * @param loading_ranks  all processors must know this.
         * @param world_comm
         * @param top the new communicator for people how will communicate on "top".
         */
        //TODO: If everybody has increasing load, then nobody in top, then the execution will fail.
        void get_communicator(double my_load_slope, int my_rank, MPI_Comm bottom, std::vector<int> *increasing_cpus, MPI_Comm *top) {
            //top is set for PE that does not have an increasing load others have undefined.
            //bottom is simply MPI_COMM_WORLD normally
            //easy as fuck right?
            const int TOP_CPU_TAG = 800;

            MPI_Comm_split(bottom, my_load_slope < SLOPE_THRESHOLD ? 1 : MPI_UNDEFINED, my_rank, top);
            MPI_Group top_gr;
            MPI_Group bottom_gr; MPI_Comm_group(bottom, &bottom_gr);
            if(*top != MPI_COMM_NULL){

                int top_rank; MPI_Comm_rank(*top, &top_rank);
                MPI_Comm_group(*top, &top_gr);

                int top_gr_size; MPI_Group_size(top_gr, &top_gr_size);

                std::vector<int> top_ranks(top_gr_size); std::iota(top_ranks.begin(), top_ranks.end(), 0);
                MPI_Group_translate_ranks(top_gr, top_ranks.size(), &top_ranks.front(), bottom_gr, &top_ranks.front());

                MPI_Group increasing_gr; MPI_Group_difference(bottom_gr, top_gr, &increasing_gr);
                int increasing_gr_size; MPI_Group_size(increasing_gr, &increasing_gr_size);

                increasing_cpus->resize(increasing_gr_size); std::iota(increasing_cpus->begin(), increasing_cpus->end(), 0);
                MPI_Group_translate_ranks(increasing_gr, increasing_cpus->size(), &increasing_cpus->front(), bottom_gr, &increasing_cpus->front());

                if(top_rank < increasing_cpus->size()) { // No more increasing than P/2 !
                    MPI_Send(&increasing_cpus->front(), increasing_cpus->size(), MPI_INT, increasing_cpus->at(top_rank), TOP_CPU_TAG, bottom);
                }

            } else {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, TOP_CPU_TAG, bottom, &status);
                int count; MPI_Get_count(&status, MPI_INT, &count);
                increasing_cpus->resize(count);
                MPI_Recv(&increasing_cpus->front(), count, MPI_INT, status.MPI_SOURCE, TOP_CPU_TAG, bottom, MPI_STATUS_IGNORE);
            }
        }

        template<int N>
        Zoltan_Struct* divide_data_into_top_bottom(std::vector<elements::Element<N>>  *data_bottom, //becomes bottom
                                         std::vector<elements::Element<N>>  *data_top,
                                         const std::vector<int>& increasing_cpus,
                                         const partitioning::CommunicationDatatype datatype,
                                         MPI_Comm bottom, MPI_Comm top) {
            const int TAG = 900;
            MESH_DATA<N> top_mesh_data;
            Zoltan_Struct* zz_top = nullptr;

            if (top == MPI_COMM_NULL) { // I am not in top comm (i have an increasing load)

                int bottom_size; MPI_Comm_size(bottom, &bottom_size);
                std::vector<int> all_ranks(bottom_size), top_ranks;
                std::iota(all_ranks.begin(), all_ranks.end(), 0);
                std::set_difference(all_ranks.begin(), all_ranks.end(),
                                    increasing_cpus.begin(), increasing_cpus.end(),
                                    std::back_inserter(top_ranks));
                //send the data within the bottom comm
                MPI_Send(&data_bottom->front(), data_bottom->size(), datatype.elements_datatype, top_ranks.at(0), TAG, bottom);
                data_bottom->clear();
            } else { // maybe better to repartition for each increasing CPU? Dont know.
                int my_top_rank;
                MPI_Comm_rank(top, &my_top_rank);
                if (my_top_rank == 0) {
                    int number_of_recv, bottom_size, top_size;
                    // Compute size of communicator
                    MPI_Comm_size(bottom, &bottom_size); MPI_Comm_size(top, &top_size);
                    // Everybody outside TOP will do a send
                    number_of_recv = bottom_size - top_size;
                    // Start receiving
                    std::vector<elements::Element<N>> recv_buff;
                    MPI_Status status;
                    int size;
                    for (int i = 0; i < number_of_recv; ++i) {
                        MPI_Probe(MPI_ANY_SOURCE, TAG, bottom, &status);
                        MPI_Get_count(&status, datatype.elements_datatype, &size);
                        recv_buff.resize(size);
                        MPI_Recv(&recv_buff.front(), size, datatype.elements_datatype, status.MPI_SOURCE,
                                 status.MPI_TAG, bottom, MPI_STATUS_IGNORE);
                        std::move(recv_buff.begin(), recv_buff.end(), std::back_inserter(top_mesh_data.els));
                    }
                }
                //load balance data over the top comm
                zz_top = zoltan_create_wrapper(true, top);
                zoltan_load_balance<N>(&top_mesh_data, zz_top, datatype, top, ENABLE_AUTOMATIC_MIGRATION);
                std::move(top_mesh_data.els.begin(), top_mesh_data.els.end(), std::back_inserter(*data_top));
            }
            return zz_top;
        }

        template<int N>
        Zoltan_Struct* divide_data_into_top_bottom2(std::vector<elements::Element<N>>  *data_bottom, //becomes bottom
                                                   std::vector<elements::Element<N>>  *data_top,
                                                   const std::vector<int>& increasing_cpus,
                                                   const partitioning::CommunicationDatatype datatype,
                                                   MPI_Comm bottom) {
            const int TAG = 900;
            MESH_DATA<N> top_mesh_data;
            Zoltan_Struct* zz_top = nullptr;
            int bottom_rank; MPI_Comm_rank(bottom, &bottom_rank);
            int bottom_size; MPI_Comm_size(bottom, &bottom_size);
            const bool in_top_partition = std::find(increasing_cpus.cbegin(), increasing_cpus.cend(), bottom_rank) == increasing_cpus.cend();
            if(!in_top_partition) top_mesh_data.els = *data_bottom;
            zz_top = zoltan_create_wrapper(ENABLE_AUTOMATIC_MIGRATION, bottom, bottom_size - increasing_cpus.size(), in_top_partition ? 1 : 0);
            zoltan_load_balance<N>(&top_mesh_data, zz_top, datatype, bottom, ENABLE_AUTOMATIC_MIGRATION);
            if(in_top_partition) {
                std::move(top_mesh_data.els.begin(), top_mesh_data.els.end(), std::back_inserter(*data_top));
            } else data_bottom->clear();
            return zz_top;
        }

        template<int N>
        elements::Element<N> take_out(const size_t data_id, std::vector<elements::Element<N>> *data) {
            std::iter_swap(data->begin() + data_id, data->end() - 1);
            elements::Element<N> el = *(data->end() - 1);
            data->pop_back();
            return el;
        }

        template<int N>
        const std::vector<elements::Element<N>> exchange(
                  std::vector<elements::Element<N>>  *bottom_data,
                  std::vector<elements::Element<N>>  *top_data, // it is always empty for increasing load cpus
                  Zoltan_Struct* zoltan_bottom, Zoltan_Struct* zoltan_top,
                  MPI_Comm bottom,
                  const std::vector<int>& increasing_cpus,
                  const partitioning::CommunicationDatatype datatype,
                  const double cell_size = 0.000625) {
            int my_bottom_rank; MPI_Comm_rank(bottom, &my_bottom_rank);
            int wsize; MPI_Comm_size(bottom, &wsize);

            const bool in_top_partition = std::find(increasing_cpus.cbegin(), increasing_cpus.cend(), my_bottom_rank) == increasing_cpus.cend();
            const int BOTTOM_SEND_TAG=903, TOP_SEND_TAG=904;
            std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize), data_to_migrate_bottom(wsize);
            std::vector<int> export_gids_b, export_lids_b, export_procs_b,
                             export_gids, export_lids, export_procs;
            std::vector<int> parts; int num_found = 0, num_known = 0, num_known_top = 0, num_known_bottom = 0;
            num_known = 0;

            size_t data_id;

            // Computing exchange for bottom particles
            data_id = 0;
            std::vector<int> PEs_top(wsize, -1), PEs_bottom(wsize, -1);
            const size_t bot_data_size = bottom_data->size();
            while (data_id < bot_data_size) {

                auto pos_in_double = functional::map<double>(bottom_data->at(data_id).position, [](auto p){return (double) p;});
                int num_found_proc, num_found_part;
                /**********************************************************************************
                 * Check if a work unit must be shared with someone within the bottom partitioning*
                 **********************************************************************************/
                Zoltan_LB_Box_PP_Assign(zoltan_bottom,
                                     pos_in_double.at(0) - cell_size,
                                     pos_in_double.at(1) - cell_size,
                                     N == 3 ? pos_in_double.at(2) - cell_size : 0.0,
                                     pos_in_double.at(0) + cell_size,
                                     pos_in_double.at(1) + cell_size,
                                     N == 3 ? pos_in_double.at(2) + cell_size : 0.0,
                                     &PEs_bottom.front(), &num_found_proc, &parts.front(), &num_found_part);


                /**********************************************************************************
                 * Check if a work unit must be shared with someone within the top partitioning   *
                 **********************************************************************************/
                Zoltan_LB_Box_PP_Assign(zoltan_top,
                                     pos_in_double.at(0) - cell_size,
                                     pos_in_double.at(1) - cell_size,
                                     N == 3 ? pos_in_double.at(2) - cell_size : 0.0,
                                     pos_in_double.at(0) + cell_size,
                                     pos_in_double.at(1) + cell_size,
                                     N == 3 ? pos_in_double.at(2) + cell_size : 0.0,
                                     &PEs_top.front(), &num_found_proc, &parts.front(), &num_found_part);

                std::vector<int> PEs_distinct(PEs_bottom.size() + PEs_top.size());
                auto it = std::set_union(PEs_bottom.begin(), PEs_bottom.end(), PEs_top.begin(), PEs_top.end(), PEs_distinct.begin());
                PEs_distinct.resize(it-PEs_distinct.begin());

                /**********************************************************************************
                 * Mark data as export for every detected PE                                      *
                 **********************************************************************************/
                for(int PE : PEs_distinct) {
                    if (PE >= 0 && PE != my_bottom_rank) {
                        export_gids.push_back(bottom_data->at(data_id).gid);
                        export_lids.push_back(bottom_data->at(data_id).lid);
                        export_procs.push_back(PE);
                        data_to_migrate.at(PE).push_back(bottom_data->at(data_id));
                        num_known++;
                    }
                }
                data_id++; //if the element must stay with me then check the next one
            }

            // Computing exchange for top particles
            data_id = 0;
            const size_t top_data_size = top_data->size();
            if(in_top_partition) { //can't be executed by bottom-only PEs
                while (data_id < top_data_size) {
                    int num_found_proc, num_found_part;
                    auto pos_in_double = functional::map<double>(top_data->at(data_id).position,
                                                                 [](auto p) { return (double) p; });
                    /**********************************************************************************
                     * Check if a work unit must be shared with someone within the bottom partitioning*
                     **********************************************************************************/
                    Zoltan_LB_Box_PP_Assign(zoltan_bottom,
                                         pos_in_double.at(0) - cell_size,
                                         pos_in_double.at(1) - cell_size,
                                         N == 3 ? pos_in_double.at(2) - cell_size : 0.0,
                                         pos_in_double.at(0) + cell_size,
                                         pos_in_double.at(1) + cell_size,
                                         N == 3 ? pos_in_double.at(2) + cell_size : 0.0,
                                         &PEs_bottom.front(), &num_found_proc, &parts.front(), &num_found_part);

                    /**********************************************************************************
                     * Check if a work unit must be shared with someone within the top partitioning   *
                     **********************************************************************************/
                    Zoltan_LB_Box_PP_Assign(zoltan_top,
                                         pos_in_double.at(0) - cell_size,
                                         pos_in_double.at(1) - cell_size,
                                         N == 3 ? pos_in_double.at(2) - cell_size : 0.0,
                                         pos_in_double.at(0) + cell_size,
                                         pos_in_double.at(1) + cell_size,
                                         N == 3 ? pos_in_double.at(2) + cell_size : 0.0,
                                         &PEs_top.front(), &num_found_proc, &parts.front(), &num_found_part);

                    std::vector<int> PEs_distinct(PEs_bottom.size() + PEs_top.size());
                    auto it = std::set_union(PEs_bottom.begin(), PEs_bottom.end(), PEs_top.begin(), PEs_top.end(),
                                             PEs_distinct.begin());
                    PEs_distinct.resize(it - PEs_distinct.begin());

                    /**********************************************************************************
                     * Mark data as export for every detected PE                                      *
                     **********************************************************************************/
                    for (int PE : PEs_distinct) {
                        if (PE >= 0 && PE != my_bottom_rank) {
                            export_gids.push_back(top_data->at(data_id).gid);
                            export_lids.push_back(top_data->at(data_id).lid);
                            export_procs.push_back(PE);
                            data_to_migrate.at(PE).push_back(top_data->at(data_id));
                            num_known++;
                        }
                    }
                    data_id++; //if the element must stay with me then check the next one
                }
            }

            ZOLTAN_ID_PTR known_gids = (ZOLTAN_ID_PTR) &export_gids.front();
            ZOLTAN_ID_PTR known_lids = (ZOLTAN_ID_PTR) &export_lids.front();
            ZOLTAN_ID_PTR found_gids, found_lids;
            int *found_procs, *found_parts;
            std::vector<elements::Element<N>> buffer;
            std::vector<elements::Element<N>> remote_data_gathered;

            // Compute who has to send me something via Zoltan.
            int ierr = Zoltan_Invert_Lists(zoltan_bottom, num_known, known_gids, known_lids, &export_procs[0], &export_procs[0],
                                           &num_found, &found_gids, &found_lids, &found_procs, &found_parts);

            std::vector<int> num_import_from_procs(wsize);
            std::vector<int> import_from_procs;

            // Compute how many elements I have to import from others, and from whom.

            for (size_t i = 0; i < num_found; ++i) {

                num_import_from_procs[found_procs[i]]++;
                if (std::find(import_from_procs.begin(), import_from_procs.end(), found_procs[i]) == import_from_procs.end())
                    import_from_procs.push_back(found_procs[i]);
            }

            // if nothing found, nothing to free.
            if(num_found > 0)
                Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

            int nb_reqs = std::count_if(data_to_migrate.cbegin(), data_to_migrate.cend(), [](auto buf){return !buf.empty();});
            int cpt = 0;

            // Send the data to neighbors
            std::vector<MPI_Request> reqs(nb_reqs);
            int nb_elements_sent = 0;
            for (size_t PE = 0; PE < wsize; PE++) {
                int send_size = data_to_migrate.at(PE).size();
                if (send_size) {
                    nb_elements_sent += send_size;
                    MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype.elements_datatype, PE, 400, bottom,
                              &reqs[cpt]);
                    cpt++;
                }
            }
            // Import the data from neighbors
            int nb_elements_recv = 0;
            for (int proc_id : import_from_procs) {
                size_t size = num_import_from_procs[proc_id];
                nb_elements_recv += size;
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, proc_id, 400, bottom, MPI_STATUS_IGNORE);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
            }

            MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);
            return remote_data_gathered;
        }

        template<int N>
        void migrate(std::vector<elements::Element<N>>  *bottom_data,
                     std::vector<elements::Element<N>>  *top_data, // it is always empty for increasing load cpus
                     Zoltan_Struct* zoltan_bottom, Zoltan_Struct* zoltan_top, // both in the same comm
                     MPI_Comm bottom,
                     const std::vector<int>& increasing_cpus,
                     const partitioning::CommunicationDatatype datatype) {

            int wsize; MPI_Comm_size(bottom, &wsize);
            int my_bottom_rank; MPI_Comm_rank(bottom, &my_bottom_rank);

            const int BOTTOM_SEND_TAG=901, TOP_SEND_TAG=902;
            const bool in_top_partition = std::find(increasing_cpus.cbegin(), increasing_cpus.cend(), my_bottom_rank) == increasing_cpus.cend();

            std::vector<std::vector<elements::Element<N>>> data_to_migrate_top(wsize), data_to_migrate_bottom(wsize);
            std::vector<int> export_gids_b, export_lids_b, export_procs_b,
                             export_gids_t, export_lids_t, export_procs_t;

            size_t data_id = 0;
            int PE, part, num_known_top = 0, num_known_bottom = 0;

            // Computing destination for bottom particles
            while (data_id < bottom_data->size()) {
                auto pos_in_double = functional::map<double>(bottom_data->at(data_id).position, [](auto p){return (double) p;});
                Zoltan_LB_Point_PP_Assign(zoltan_bottom, &pos_in_double.front(), &PE, &part);
                if (PE != my_bottom_rank) {
                    export_gids_b.push_back(bottom_data->at(data_id).gid);
                    export_lids_b.push_back(bottom_data->at(data_id).lid);
                    export_procs_b.push_back(PE);
                    data_to_migrate_bottom.at(PE).push_back(take_out(data_id, bottom_data));
                    num_known_bottom++;
                } else data_id++; //if the element must stay with me then check the next one
            }

	        int my_top_rank = -1;
            // I have top data to migrate because I am not an "increasing" cpu
            if(in_top_partition) {
                data_id = 0;
                // Computing destination for top particles
                while (data_id < top_data->size()) {
                    auto pos_in_double = functional::map<double>(top_data->at(data_id).position, [](auto p){return (double) p;});
                    Zoltan_LB_Point_PP_Assign(zoltan_bottom, &pos_in_double.front(), &PE, &part);
                    if (PE != my_bottom_rank) {
                        if (std::find(increasing_cpus.cbegin(), increasing_cpus.cend(), PE) != increasing_cpus.cend()) {
                            Zoltan_LB_Point_PP_Assign(zoltan_top, &pos_in_double.front(), &PE, &part);
                            if(PE != my_bottom_rank){
			                    export_gids_t.push_back(top_data->at(data_id).gid);
                                export_lids_t.push_back(top_data->at(data_id).lid);
                                export_procs_t.push_back(PE);
                                data_to_migrate_top.at(PE).push_back(take_out(data_id, top_data));
                                num_known_top++;
			                } else data_id++;
                        } else {
                            export_gids_b.push_back(top_data->at(data_id).gid);
                            export_lids_b.push_back(top_data->at(data_id).lid);
                            export_procs_b.push_back(PE);
                            data_to_migrate_bottom.at(PE).push_back(take_out(data_id, top_data));
                            num_known_bottom++;
                        }
                    } else data_id++; // if the element must stay with me then check the next one
                }
            }

	        /*************************************************/
            /** Variables for send and recv                  */
            /*************************************************/

            ZOLTAN_ID_PTR found_gids, found_lids;
            int *found_procs, *found_parts, num_found = 0, ierr;
            std::vector<int> num_import_from_procs_t(wsize), num_import_from_procs_b(wsize);
            std::vector<int> import_from_procs_t, import_from_procs_b;

            /*************************************************/
            /** Sending part to BOTTOM                       */
            /*************************************************/

            auto known_gids = (ZOLTAN_ID_PTR) &export_gids_b.front();
            auto known_lids = (ZOLTAN_ID_PTR) &export_lids_b.front();


            ierr = Zoltan_Invert_Lists(zoltan_bottom, num_known_bottom, known_gids, known_lids, &export_procs_b[0], &export_procs_b[0],
                                       &num_found, &found_gids, &found_lids, &found_procs, &found_parts);

            for (size_t i = 0; i < num_found; ++i) {
                num_import_from_procs_b[found_procs[i]]++;
                if (std::find(import_from_procs_b.begin(), import_from_procs_b.end(), found_procs[i]) == import_from_procs_b.end())
                    import_from_procs_b.push_back(found_procs[i]);
            }

            if(num_found > 0)
                Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

            int nb_reqs = 0;
            for (auto buf: data_to_migrate_bottom)
                if (!buf.empty()) nb_reqs++;


            int cpt = 0;
            std::vector<MPI_Request> reqs_b(nb_reqs);
            for (size_t PE = 0; PE < wsize; PE++) {
                int send_size = data_to_migrate_bottom.at(PE).size();
                if (send_size) {
                    MPI_Isend(&data_to_migrate_bottom.at(PE).front(), send_size, datatype.elements_datatype, PE,
                              BOTTOM_SEND_TAG, bottom, &reqs_b[cpt]);

                    cpt++;
                }
            }

            /*************************************************/
            /** Sending part, to TOP                         */
            /*************************************************/

            //Actually, I take part of the rest of the computation, so I may have data to send to the top comm
            std::vector<MPI_Request> reqs_t;

            known_gids = (ZOLTAN_ID_PTR) &export_gids_t.front();
            known_lids = (ZOLTAN_ID_PTR) &export_lids_t.front();

            ierr = Zoltan_Invert_Lists(zoltan_top, num_known_top, known_gids, known_lids, &export_procs_t[0],
                                       &export_procs_t[0],
                                       &num_found, &found_gids, &found_lids, &found_procs, &found_parts);

            if(in_top_partition) {

                for (size_t i = 0; i < num_found; ++i) {
                    num_import_from_procs_t[found_procs[i]]++;
                    if (std::find(import_from_procs_t.begin(), import_from_procs_t.end(), found_procs[i]) ==
                        import_from_procs_t.end())
                        import_from_procs_t.push_back(found_procs[i]);
                }

                if (num_found > 0)
                    Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

                nb_reqs = 0;
                for (auto buf: data_to_migrate_top) {
                    if (!buf.empty()) nb_reqs++;
                }

                cpt = 0;
                reqs_t.resize(nb_reqs);
                for (size_t PE = 0; PE < wsize; PE++) {
                    int send_size = data_to_migrate_top.at(PE).size();
                    if (send_size) {
                        MPI_Isend(&data_to_migrate_top.at(PE).front(), send_size, datatype.elements_datatype, PE,
                                  TOP_SEND_TAG, bottom, &reqs_t[cpt]);
                        cpt++;
                    }
                }
            }

            /*************************************************/
            /** Receiving part from BOTTOM                   */
            /*************************************************/

            std::vector<elements::Element<N>> buffer;
            for (int proc_id : import_from_procs_b) {
                size_t size = num_import_from_procs_b[proc_id];
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, proc_id, BOTTOM_SEND_TAG, bottom, MPI_STATUS_IGNORE);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(*bottom_data));
            }

            /*************************************************/
            /** Receiving part from TOP                      */
            /*************************************************/
            //I don't have an increasing load so I may receive some data from other "top" cpu.
            if(in_top_partition) {
                for (int proc_id : import_from_procs_t) {
                    size_t size = num_import_from_procs_t[proc_id];
                    buffer.resize(size);
                    MPI_Recv(&buffer.front(), size, datatype.elements_datatype, proc_id, TOP_SEND_TAG, bottom, MPI_STATUS_IGNORE);
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(*top_data));
                }
            }

            // Waiting on requests
            MPI_Waitall(reqs_b.size(), &reqs_b.front(), MPI_STATUSES_IGNORE);
            if(in_top_partition)
                MPI_Waitall(reqs_t.size(), &reqs_t.front(), MPI_STATUSES_IGNORE);

            // Update local ids
            size_t i = 0;
            const int nb_data_b = bottom_data->size();
            for(auto& e : *bottom_data){
                e.lid = i;
                i++;
            }
            i = 0;
            for(auto& e: *top_data) {
                e.lid = i+nb_data_b;
				i++;
            }

        }
        /**
         * Data movement algorithm:
         * IN: b = Communicator bottom (is MPI_COMM_WORLD, normally)
         * IN: t = Communicator top (composed by CPU with load slope < threshold)
         * IN: particleIdx = Index of particle
         * IN: topParticleIndices = indices of particles that belongs to the top communicator
         * ---------------
         * increasingCpuIndices = MPI_Group_difference(b, t)
         * particle = getParticleFromIndex(particleIdx)
         * partitionIdx_bottom = GetPartitionIndex(p, b)
         * if particleIdx in topParticleIndices then
         *     if partitionIdx_bottom in increasingCpuIndices then
         *         partitionIdx_top = GetPartitionIndex(p, t)
         *         Send particle to CPU that owns partitionIdx_top
         *     else
         *         Remove particle from topParticleIndices
         *         Send particle to CPU `partitionIdx_bottom`(assuming that each CPU owns the partition that has the same idx)
         *     endif
         * else
         *     Send particle to CPU `partitionIdx_bottom`(assuming that each CPU owns the partition that has the same idx)
         * endif
         */
        /*
         * if partitionIdx_bottom in increasingCpuIndices then
         *     if particleIdx in topParticleIndices then
         *         partitionIdx_top = GetPartitionIndex(p, t)
         *         Send particle to CPU that owns partitionIdx_top
         *     else
         *         Send particle to CPU `partitionIdx_bottom`(assuming that each CPU owns the partition that has the same idx)
         *     endif
         * else
         *     if particleIdx in topParticleIndices then
         *         Remove particle from topParticleIndices
         *     endif
         *     Send particle to CPU `partitionIdx_bottom`(assuming that each CPU owns the partition that has the same idx)
         * endif
         * */
    }
}

#endif //NBMPI_UNLOADING_MODEL_HPP
