//
// Created by xetql on 18.12.17.
//

#ifndef NBMPI_LOADBALANCER_HPP
#define NBMPI_LOADBALANCER_HPP

#include <map>
#include <mpi.h>

#include "partitioner.hpp"
#include "spatial_elements.hpp"
#include "spatial_bisection.hpp"
#include "zoltan_fn.hpp"

namespace load_balancing {

    template<int N>
    inline void gather_elements_on(const int world_size,
                                   const int my_rank,
                                   const int nb_elements,
                                   const std::vector<elements::Element<N>> &local_el,
                                   const int dest_rank,
                                   std::vector<elements::Element<N>> &dest_el,
                                   const MPI_Datatype &sendtype,
                                   const MPI_Comm &comm) {
        int nlocal = local_el.size();
        std::vector<int> counts(world_size, 0), displs(world_size, 0);
        MPI_Gather(&nlocal, 1, MPI_INT, &counts.front(), 1, MPI_INT, dest_rank, comm);
        for (int cpt = 0; cpt < world_size; ++cpt) displs[cpt] = cpt == 0 ? 0 : displs[cpt - 1] + counts[cpt - 1];
        if (my_rank == dest_rank) dest_el.resize(nb_elements);
        MPI_Gatherv(&local_el.front(), nlocal, sendtype,
                    &dest_el.front(), &counts.front(), &displs.front(), sendtype, dest_rank, comm);
    }

    using ProcessingElementID=int;

    template<typename PartitionType, typename ContainedDataType, typename DomainContainer>
    class LoadBalancer {
    protected:
        const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> *partitioner;
        const std::map<ProcessingElementID, std::vector<ContainedDataType>> storage_table;
        int world_size;
        int caller_rank;
        partitioning::CommunicationDatatype communication_datatypes;
        MPI_Comm LB_COMM;
    public:

        const MPI_Datatype get_element_datatype() {
            return communication_datatypes.elements_datatype;
        }

        const MPI_Datatype get_range_datatype() {
            return communication_datatypes.range_datatype;
        }

        const MPI_Datatype get_domain_datatype() {
            return communication_datatypes.domain_datatype;
        }

        const MPI_Comm get_communicator() {
            return LB_COMM;
        }

        //Assume that the data are all located on PE 0.
        LoadBalancer(
                const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner,
                MPI_Comm comm) : partitioner(&partitioner), communication_datatypes(partitioner.register_datatype()),
                                 LB_COMM(comm) {
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
        };

        //Provide a table that specifies where is the data located (on which PE)
        LoadBalancer(const std::map<ProcessingElementID, std::vector<ContainedDataType>> &storage_table,
                     const partitioning::Partitioner<PartitionType, std::vector<ContainedDataType>, DomainContainer> &partitioner2,
                     MPI_Comm comm) : partitioner(&partitioner2), storage_table(storage_table),
                                      communication_datatypes(partitioner2.register_datatype()), LB_COMM(comm) {
            MPI_Comm_size(comm, &world_size);
            MPI_Comm_rank(comm, &caller_rank);
        };

        void stop() {
            MPI_Barrier(this->LB_COMM);
            communication_datatypes.free_datatypes();
        }

        //TODO: It should somehow retrieves the data from the processing elements, at least makes it possible.
        //TODO: It should map the partitions as function of the PEs' pace
        virtual void load_balance(std::vector<ContainedDataType> &data, std::vector<DomainContainer> &domain_boundary) {
            std::vector<int> counts(this->world_size);
            std::vector<int> displs(this->world_size);
            std::fill(counts.begin(), counts.end(), 0);
            std::vector<partitioning::geometric::Domain<ContainedDataType::number_of_dimensions>> subdomains;
            //TODO: Have to detect if the algorithm is sequential or parallel
            if (this->caller_rank == 0) {
                //partition the dataset in world_size partitions
                std::unique_ptr<PartitionType> partitions = partitioner->partition_data(data, domain_boundary.at(0),
                                                                                        world_size);
                auto partitioned_data = partitions->parts;
                subdomains = partitions->domains; //all the sub domains (geometric)

                //std::for_each(subdomains.begin(), subdomains.end(), [](auto const& el){std::cout << to_string(el) << std::endl;});
                std::sort(partitioned_data.begin(), partitioned_data.end(), [](const auto &a, const auto &b) -> bool {
                    return a.first < b.first; //sort by partition id
                });

                //counts the number to send to each processor... normally the same number of particles has to be sent
                displs[0] = 0;

                for (int i = 0; i < this->world_size; ++i)
                    for (size_t cpt = displs[i]; cpt < partitioned_data.size(); ++cpt)
                        if (partitioned_data.at(cpt).first == i)
                            counts[i] += 1; //increment the number of particles attributed to the PE
                        else {
                            displs[i + 1] = cpt; //set the shift in the buffer
                            break;
                        }
                //send elements
                for (size_t i = 0; i < partitioned_data.size(); ++i) data[i] = partitioned_data.at(i).second;
                for (int i = 0; i < this->world_size; ++i) {
                    //send the particle attributed to the PE
                    MPI_Send(&data[displs[i]], counts[i], this->get_element_datatype(), i, 666, LB_COMM);
                }
                //broadcast the geometric partition of the initial domain
                MPI_Bcast(&subdomains.front(), subdomains.size(), this->get_domain_datatype(), 0, LB_COMM);
            }
            MPI_Status status;
            //Probe the data sent
            MPI_Probe(0, 666, LB_COMM, &status);
            int my_data_size;
            // Get the number of elements according to the status.
            MPI_Get_count(&status, this->get_element_datatype(), &my_data_size);
            // resize the data to fit the number of elements
            data.resize(my_data_size);
            // receive the elements
            MPI_Recv(&data.front(), my_data_size, this->get_element_datatype(), 0, 666, LB_COMM, MPI_STATUS_IGNORE);
            //broadcast the geometric partition of the initial domain
            domain_boundary.clear();
            if (this->caller_rank != 0) {
                domain_boundary.resize(this->world_size);
                MPI_Bcast(&domain_boundary.front(), this->world_size, this->get_domain_datatype(), 0, LB_COMM);
            } else {
                std::move(subdomains.begin(), subdomains.end(), std::back_inserter(domain_boundary));
            }
        }

        /**
         * Acquire data from neighboring PEs and store the data in the buffer.
         * @param data
         * @param domains
         */
        virtual const std::vector<ContainedDataType>
        exchange_data(const std::vector<ContainedDataType> &data, const std::vector<DomainContainer> &domains)= 0;

        virtual void
        migrate_particles(std::vector<ContainedDataType> &data, const std::vector<DomainContainer> &domains)  = 0;
    };

    namespace geometric {

        template<int N>
        const std::vector<elements::Element<N>> __zoltan_exchange_data(const std::vector<elements::Element<N>> &data,
                                                                     Zoltan_Struct *load_balancer,
                                                                     const partitioning::CommunicationDatatype datatype,
                                                                     const MPI_Comm LB_COMM,
                                                                     int &nb_elements_recv,
                                                                     int &nb_elements_sent,
                                                                     double cell_size = 0.007) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);

            std::vector<elements::Element<N>> buffer;
            std::vector<elements::Element<N>> remote_data_gathered;
            // Get the neighbors
            std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);

            //check within the remaining elements which belong to the current PE
            size_t data_id = 0;
            while (data_id < data.size()) {
                auto el = data.at(data_id).position;
                std::vector<int> procs;
                int nprocs;
                if (N == 3)
                    Zoltan_LB_Box_Assign(load_balancer,
                                         el.at(0) - cell_size, el.at(1) - cell_size, el.at(2) - cell_size,
                                         el.at(0) + cell_size, el.at(1) + cell_size, el.at(2) + cell_size,
                                         &procs.front(), &nprocs);
                else
                    Zoltan_LB_Box_Assign(load_balancer,
                                         el.at(0) - cell_size, el.at(1) - cell_size, 0,
                                         el.at(0) + cell_size, el.at(1) + cell_size, 0,
                                         &procs.front(), &nprocs);

                for (const int PE : procs) {
                    if (PE == (size_t) caller_rank) continue; //do not check with myself
                    data_to_migrate.at(PE).push_back(
                            data.at(data_id)); // get the value and push it in the "to migrate" vector
                }
                data_id++; //if the element must stay with me then check the next one
            }

            std::vector<MPI_Request> reqs(wsize);
            std::vector<MPI_Status> statuses(wsize);
            int cpt = 0, nb_neighbors = wsize;
            nb_elements_sent = 0;
            for (size_t neighbor_idx = 0; neighbor_idx < wsize; ++neighbor_idx) {   //give all my data to neighbors
                int send_size = data_to_migrate.at(neighbor_idx).size();
                nb_elements_sent += send_size;
                MPI_Isend(&data_to_migrate.at(neighbor_idx).front(), send_size, datatype.elements_datatype,
                          neighbor_idx, 200, LB_COMM, &reqs[cpt]);
                cpt++;
            }
            //Wait that people have sent their particles such that no PE have to know who are its neighbors
            MPI_Barrier(LB_COMM);
            cpt = 0;
            int flag = 1;
            while (flag) {// receive the data in any order
                int source_rank, size;
                MPI_Iprobe(MPI_ANY_SOURCE, 200, LB_COMM, &flag, &statuses[cpt]);
                if (!flag) break;
                source_rank = statuses[cpt].MPI_SOURCE;
                MPI_Get_count(&statuses[cpt], datatype.elements_datatype, &size);
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, source_rank, 200, LB_COMM, &statuses[cpt]);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
                cpt++;
            }
            //MPI_Waitall(reqs.size(), &reqs.front(), &statuses.front()); //less strict than mpi_barrier
            nb_elements_recv = remote_data_gathered.size();
            return remote_data_gathered;
        }

        template<int N>
        const std::vector<elements::Element<N>> __exchange_data(const std::vector<elements::Element<N>> &data,
                                                                const std::vector<partitioning::geometric::Domain<N>> &domains,
                                                                const partitioning::CommunicationDatatype datatype,
                                                                const MPI_Comm LB_COMM,
                                                                int &nb_elements_recv,
                                                                int &nb_elements_sent,
                                                                double cell_size = 0.007) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);
            const int EXCHANGE_TAG = 200, PRE_EXCHANGE_TAG = 201;

            std::vector<elements::Element<N>> buffer;
            std::vector<elements::Element<N>> remote_data_gathered;
            // Get the neighbors
            std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
            auto neighbors = partitioning::utils::unzip(
                    partitioning::geometric::get_neighboring_domains<N>(caller_rank, domains, cell_size)).first;

            for (const size_t &PE : neighbors) {//size_t PE = 0; PE < wsize; ++PE) {
                if (PE == (size_t) caller_rank) continue; //do not check with myself
                //check within the remaining elements which belong to the current PE
                size_t data_id = 0;
                while (data_id < data.size()) {
                    if (elements::distance2<N>(domains.at(PE), data.at(data_id)) <= cell_size) {
                        data_to_migrate.at(PE).push_back(
                                data.at(data_id)); // get the value and push it in the "to migrate" vector
                    }
                    data_id++; //if the element must stay with me then check the next one
                }
            }

            std::vector<MPI_Request> reqs(neighbors.size()), snd_reqs, rcv_reqs;
            std::vector<MPI_Status> statuses(neighbors.size());
            int cpt = 0, nb_neighbors = neighbors.size();
            nb_elements_sent = 0;

            for (const size_t &neighbor_idx : neighbors) {   //give all my data to neighbors
                int send_size = data_to_migrate.at(neighbor_idx).size();
                //if(send_size) {
                nb_elements_sent += send_size;
                MPI_Isend(&data_to_migrate.at(neighbor_idx).front(), send_size, datatype.elements_datatype,
                          neighbor_idx, 200, LB_COMM, &reqs[cpt]);
                cpt++;
                //}
            }

            cpt = 0;
            while (cpt < nb_neighbors) {// receive the data in any order
                int source_rank, size;
                MPI_Probe(MPI_ANY_SOURCE, 200, LB_COMM, &statuses[cpt]);
                source_rank = statuses[cpt].MPI_SOURCE;
                MPI_Get_count(&statuses[cpt], datatype.elements_datatype, &size);
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, source_rank, 200, LB_COMM, &statuses[cpt]);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
                cpt++;
            }
            MPI_Waitall(reqs.size(), &reqs.front(), &statuses.front()); //less strict than mpi_barrier
            nb_elements_recv = remote_data_gathered.size();
            return remote_data_gathered;
        }

        template<int N>
        void __migrate_particles(std::vector<elements::Element<N>> &data,
                                 const std::vector<partitioning::geometric::Domain<N>> &domains,
                                 const partitioning::CommunicationDatatype datatype,
                                 const MPI_Comm LB_COMM,
                                 std::vector<size_t> neighbors = std::vector<size_t>()) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);

            std::vector<elements::Element<N>> buffer;
            std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
            if (neighbors.empty())
                neighbors = partitioning::utils::unzip(
                        partitioning::geometric::get_neighboring_domains(caller_rank, domains, 0.08)).first;

            for (const size_t &PE : neighbors) {
                if (PE == (size_t) caller_rank) continue; //do not check with myself
                // check within the remaining elements which belong to the current PE
                size_t data_id = 0;
                while (data_id < data.size()) {
                    if (elements::is_inside<N>(data.at(data_id), domains.at(PE))) {
                        //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                        //swap iterator values in constant time
                        std::iter_swap(data.begin() + data_id, data.end() - 1);
                        //get the value and push it in the "to migrate" vector
                        data_to_migrate.at(PE).push_back(*(data.end() - 1));
                        //pop the head of the list in constant time
                        data.pop_back();
                    } else data_id++; //if the element must stay with me then check the next one
                }
            }

            std::vector<MPI_Request> reqs(neighbors.size());
            std::vector<MPI_Status> statuses(neighbors.size());

            int cpt = 0, nb_neighbors = neighbors.size();
            for (const size_t &PE : neighbors) {
                int send_size = data_to_migrate.at(PE).size();
                MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype.elements_datatype, PE, 300, LB_COMM,
                          &reqs[cpt]);
                cpt++;
            }
            cpt = 0;
            while (cpt < nb_neighbors) {// receive the data in any order
                int source_rank, size;
                MPI_Probe(MPI_ANY_SOURCE, 300, LB_COMM, &statuses[cpt]);
                source_rank = statuses[cpt].MPI_SOURCE;
                MPI_Get_count(&statuses[cpt], datatype.elements_datatype, &size);
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, source_rank, 300, LB_COMM, &statuses[cpt]);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(data));
                cpt++;
            }
            MPI_Waitall(cpt, &reqs.front(), &statuses.front());
        }

        template<int N>
        std::tuple<std::vector<std::vector<elements::Element<N>>>, size_t, size_t> get_next_batch(
                std::vector<elements::Element<N>> &data,
                const std::vector<partitioning::geometric::Domain<N>> &domains,
                const size_t start_at,
                const size_t batch_size,
                const MPI_Comm LB_COMM,
                const bool erase_after_add = false) {

            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);
            std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
            auto neighbors = partitioning::utils::unzip(
                    partitioning::geometric::get_neighboring_domains(caller_rank, domains, 0.08)).first;
            size_t current_batch_size = 0;
            size_t data_id = start_at;
            for (const size_t &PE : neighbors) {
                if (PE == (size_t) caller_rank) continue; //do not check with myself
                // check within the remaining elements which belong to the current PE

                while (data_id < data.size() && current_batch_size < batch_size) {
                    if (elements::is_inside<N>(data.at(data_id), domains.at(PE))) {
                        if (erase_after_add) {
                            //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                            //swap iterator values in constant time
                            std::iter_swap(data.begin() + data_id, data.end() - 1);
                            //get the value and push it in the "to migrate" vector
                            data_to_migrate.at(PE).push_back(*(data.end() - 1));
                            //pop the head of the list in constant time
                            data.pop_back();
                        } else {
                            data_to_migrate.at(PE).push_back(
                                    data.at(data_id)); // get the value and push it in the "to migrate" vector
                        }
                        current_batch_size++;
                    } else if (erase_after_add) data_id++; //if the element must stay with me then check the next one
                    if (!erase_after_add) data_id++;
                }
            }
            return std::make_tuple(data_to_migrate, current_batch_size, data_id);
        }

        template<int N>
        const std::vector<elements::Element<N>> exchange_data(std::vector<elements::Element<N>> &data,
                                                              const std::vector<partitioning::geometric::Domain<N>> &domains,
                                                              const partitioning::CommunicationDatatype datatype,
                                                              const MPI_Comm LB_COMM,
                                                              int &nb_elements_recv,
                                                              int &nb_elements_sent,
                                                              double cell_size = 0.007) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);

            std::vector<elements::Element<N>> buffer;
            std::vector<elements::Element<N>> remote_data_gathered;
            // Get the neighbors
            size_t batch_size = 0, data_pointer = 0;
            bool empty_recv_buff = false;

            std::vector<std::vector<elements::Element<N>>> data_to_migrate;
            auto neighbors = partitioning::utils::unzip(
                    partitioning::geometric::get_neighboring_domains<N>(caller_rank, domains, cell_size)).first;
            const int nb_of_round = 10;
            const int next_batch_size = std::ceil(data.size() * wsize / nb_of_round);

            for (int T = 0; T < nb_of_round; T++) {

                std::tie(data_to_migrate, batch_size, data_pointer) = get_next_batch<N>(data, domains, data_pointer,
                                                                                        next_batch_size, LB_COMM,
                                                                                        false);

                auto nb_receiving_neighbors = std::count_if(data_to_migrate.cbegin(),
                                                            data_to_migrate.cend(),
                                                            [](auto data) { return !data.empty(); });

                std::vector<MPI_Request> reqs(nb_receiving_neighbors), snd_reqs(nb_receiving_neighbors), rcv_reqs(
                        wsize);
                std::vector<MPI_Status> statuses(neighbors.size());

                // PREPARATION
                int pe_req_idx = 0;
                for (const size_t &PE : neighbors) {
                    if (PE == (size_t) caller_rank) continue;
                    int send_size = data_to_migrate.at(PE).size();
                    if (send_size) {
                        MPI_Issend(&send_size, 1, MPI_INT, PE, 201, LB_COMM, &snd_reqs[pe_req_idx]);
                        pe_req_idx++;
                    }
                }
                std::map<int, int> receive_data_size_lookup;
                for (size_t PE = 0; PE < wsize; ++PE) {
                    receive_data_size_lookup[PE] = 0;
                    MPI_Irecv(&receive_data_size_lookup[PE], 1, MPI_INT, PE, 201, LB_COMM, &rcv_reqs[PE]);
                }
                if (!snd_reqs.empty())
                    MPI_Waitall(snd_reqs.size(), &snd_reqs.front(), MPI_STATUSES_IGNORE);

                MPI_Barrier(LB_COMM);

                snd_reqs.clear();
                //Clear requests to wrong PEs
                for (auto &req : rcv_reqs) {
                    int flag;
                    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                    if (!flag) MPI_Cancel(&req);
                }
                rcv_reqs.clear();

                auto nb_sending_neighbors = std::count_if(receive_data_size_lookup.cbegin(),
                                                          receive_data_size_lookup.cend(),
                                                          [](auto pe_data) { return pe_data.second > 0; });
                empty_recv_buff = nb_sending_neighbors == 0;

                int cpt = 0;
                nb_elements_sent = 0;
                for (const size_t &neighbor_idx : neighbors) {   //give all my data to neighbors
                    int send_size = data_to_migrate.at(neighbor_idx).size();
                    nb_elements_sent += send_size;
                    if (send_size) {
                        MPI_Isend(&data_to_migrate.at(neighbor_idx).front(), send_size, datatype.elements_datatype,
                                  neighbor_idx, 200, LB_COMM, &reqs[cpt]);
                        cpt++;
                    }
                }

                cpt = 0;
                while (cpt < nb_sending_neighbors && !empty_recv_buff) {// receive the data in any order
                    int source_rank, size;
                    MPI_Probe(MPI_ANY_SOURCE, 200, LB_COMM, &statuses[cpt]);
                    source_rank = statuses[cpt].MPI_SOURCE;
                    MPI_Get_count(&statuses[cpt], datatype.elements_datatype, &size);
                    buffer.resize(size);
                    MPI_Recv(&buffer.front(), size, datatype.elements_datatype, source_rank, 200, LB_COMM,
                             MPI_STATUS_IGNORE);
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
                    cpt++;
                }
                if (!reqs.empty())
                    MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE); // less strict than mpi_barrier

            }

            nb_elements_recv = remote_data_gathered.size();
            return remote_data_gathered;
        }

        template<int N>
        void migrate_zoltan(std::vector<elements::Element<N>> &data, int numImport, int numExport, int *exportProcs,
                            unsigned int *exportGlobalGids,
                            const partitioning::CommunicationDatatype datatype,
                            const MPI_Comm LB_COMM) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);
            std::vector<elements::Element<N> > buffer;
            std::map<int, std::shared_ptr<std::vector<elements::Element<N> > > > data_to_migrate;

            for (int i = 0; i < numExport; ++i)
                if (data_to_migrate.find(exportProcs[i]) == data_to_migrate.end())
                    data_to_migrate[exportProcs[i]] = std::make_shared<std::vector<elements::Element<N>>>();

            for (int i = 0; i < numExport; ++i) {
                auto PE = exportProcs[i];
                auto gid = exportGlobalGids[i];

                //check within the remaining elements which belong to the current PE
                size_t data_id = 0;
                while (data_id < data.size()) {
                    if (gid == (size_t) data[data_id].gid) {
                        //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                        //swap iterator values in constant time
                        std::iter_swap(data.begin() + data_id, data.end() - 1);
                        //get the value and push it in the "to migrate" vector
                        data_to_migrate[PE]->push_back(*(data.end() - 1));
                        //pop the head of the list in constant time
                        data.pop_back();
                    } else data_id++; //if the element must stay with me then check the next one
                }
            }

            std::vector<MPI_Request> reqs(data_to_migrate.size());

            int cpt = 0;
            for (auto const &pe_data : data_to_migrate) {
                int send_size = pe_data.second->size();
                MPI_Isend(&pe_data.second->front(), send_size, datatype.elements_datatype, pe_data.first, 300, LB_COMM,
                          &reqs[cpt]);
                cpt++;
            }
            int collectData = 0;

            while (collectData < numImport) {// receive the data in any order
                int source_rank, size;
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, 300, LB_COMM, &status);
                source_rank = status.MPI_SOURCE;
                MPI_Get_count(&status, datatype.elements_datatype, &size);
                collectData += size;
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, source_rank, 300, LB_COMM, &status);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(data));

            }
            MPI_Waitall(cpt, &reqs.front(), MPI_STATUSES_IGNORE);

        }

        template<int N>
        const std::vector<elements::Element<N>> zoltan_exchange_data(std::vector<elements::Element<N>> &data,
                                                                     Zoltan_Struct *load_balancer,
                                                                     const partitioning::CommunicationDatatype datatype,
                                                                     const MPI_Comm LB_COMM,
                                                                     int &nb_elements_recv,
                                                                     int &nb_elements_sent,
                                                                     double cell_size = 0.000625) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);

            std::vector<elements::Element<N>> buffer;
            std::vector<elements::Element<N>> remote_data_gathered;

            if(wsize == 1) return remote_data_gathered;

	        std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
            size_t data_id = 0;
            std::vector<int> PEs(wsize, -1);
            int num_found, num_known = 0;
            std::vector<int> export_gids, export_lids, export_procs;

            // so much memory could be allocated here... potentially PE * n * DIM * 44 bytes => so linear in N
            // as DIM << PE <<<< n
            while (data_id < data.size()) {
                auto pos_in_double = functional::map<double>(data.at(data_id).position, [](auto p){return (double) p;});
                Zoltan_LB_Box_Assign(load_balancer,
                                     pos_in_double.at(0) - cell_size,
                                     pos_in_double.at(1) - cell_size,
                                     N == 3 ? pos_in_double.at(2) - cell_size : 0.0,
                                     pos_in_double.at(0) + cell_size,
                                     pos_in_double.at(1) + cell_size,
                                     N == 3 ? pos_in_double.at(2) + cell_size : 0.0,
                                     &PEs.front(), &num_found);

                for(int PE_idx = 0; PE_idx < num_found; PE_idx++) {
                    int PE = PEs[PE_idx];
                    if(PE >= 0) {
                        if (PE != caller_rank) {
                            export_gids.push_back(data.at(data_id).gid);
                            export_lids.push_back(data.at(data_id).lid);
                            export_procs.push_back(PE);
                            //get the value and copy it into the "to migrate" vector
                            data_to_migrate.at(PE).push_back(data.at(data_id));
                            num_known++;
                        }
                    }
                }
                data_id++; //if the element must stay with me then check the next one
            }

            ZOLTAN_ID_PTR known_gids = (ZOLTAN_ID_PTR) &export_gids.front();
            ZOLTAN_ID_PTR known_lids = (ZOLTAN_ID_PTR) &export_lids.front();
            ZOLTAN_ID_PTR found_gids, found_lids;

            int *found_procs, *found_parts;

            // Compute who has to send me something via Zoltan.
            int ierr = Zoltan_Invert_Lists(load_balancer, num_known, known_gids, known_lids, &export_procs[0], &export_procs[0],
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

            int nb_reqs = 0;
            // Just check how many requests I have to create
            for (auto buf: data_to_migrate) {
                if (!buf.empty()) nb_reqs++;
            }

            int cpt = 0;

            // Send the data to neighbors
            std::vector<MPI_Request> reqs(nb_reqs);
            nb_elements_sent = 0;
            for (size_t PE = 0; PE < wsize; PE++) {
                int send_size = data_to_migrate.at(PE).size();
                if (send_size) {
                    nb_elements_sent += send_size;
                    MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype.elements_datatype, PE, 400, LB_COMM,
                              &reqs[cpt]);
                    cpt++;
                }
            }
            // Import the data from neighbors
            nb_elements_recv = 0;
            for (int proc_id : import_from_procs) {
                size_t size = num_import_from_procs[proc_id];
                nb_elements_recv += size;
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, proc_id, 400, LB_COMM, MPI_STATUS_IGNORE);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
            }

            MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);
            return remote_data_gathered;
        }
        template<int N>
        void zoltan_migrate_particles(
                std::vector<elements::Element<N>> &data,
                Zoltan_Struct *load_balancer,
                const partitioning::CommunicationDatatype datatype,
                const MPI_Comm LB_COMM) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);

	    if(wsize == 1) return;

            std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);

            size_t data_id = 0;
            int PE;
            int num_known = 0;
            std::vector<int> export_gids, export_lids, export_procs;
            while (data_id < data.size()) {
                auto pos_in_double = functional::map<double>(data.at(data_id).position, [](auto p){return (double) p;});
                Zoltan_LB_Point_Assign(load_balancer, &pos_in_double.front(), &PE);
                if (PE != caller_rank) {
                    export_gids.push_back(data.at(data_id).gid);
                    export_lids.push_back(data.at(data_id).lid);
                    export_procs.push_back(PE);
                    //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                    //swap iterator values in constant time
                    std::iter_swap(data.begin() + data_id, data.end() - 1);
                    //get the value and push it in the "to migrate" vector
                    data_to_migrate.at(PE).push_back(*(data.end() - 1));
                    //pop the head of the list in constant time
                    data.pop_back();
                    num_known++;
                } else data_id++; //if the element must stay with me then check the next one
            }

            ZOLTAN_ID_PTR known_gids = (ZOLTAN_ID_PTR) &export_gids.front();
            ZOLTAN_ID_PTR known_lids = (ZOLTAN_ID_PTR) &export_lids.front();
            ZOLTAN_ID_PTR found_gids, found_lids;

            int *found_procs, *found_parts, num_found;

            int ierr = Zoltan_Invert_Lists(load_balancer, num_known, known_gids, known_lids, &export_procs[0], &export_procs[0],
                                &num_found, &found_gids, &found_lids, &found_procs, &found_parts);

            std::vector<int> num_import_from_procs(wsize);
            std::vector<int> import_from_procs;

            for (size_t i = 0; i < num_found; ++i) {
                num_import_from_procs[found_procs[i]]++;
                if (std::find(import_from_procs.begin(), import_from_procs.end(), found_procs[i]) == import_from_procs.end())
                    import_from_procs.push_back(found_procs[i]);
            }

            /* Let's Migrate ma boi ! */

            if(num_found > 0)
                Zoltan_LB_Free_Part(&found_gids, &found_lids, &found_procs, &found_parts);

            int nb_reqs = 0;
            for (auto buf: data_to_migrate) {
                if (!buf.empty()) nb_reqs++;
            }

            int cpt = 0;
            std::vector<MPI_Request> reqs(nb_reqs);
            for (size_t PE = 0; PE < wsize; PE++) {
                int send_size = data_to_migrate.at(PE).size();
                if (send_size) {
                    MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype.elements_datatype, PE, 300, LB_COMM,
                              &reqs[cpt]);
                    cpt++;
                }
            }
            std::vector<elements::Element<N>> buffer;
            for (int proc_id : import_from_procs) {
                size_t size = num_import_from_procs[proc_id];
                buffer.resize(size);
                MPI_Recv(&buffer.front(), size, datatype.elements_datatype, proc_id, 300, LB_COMM, MPI_STATUS_IGNORE);
                std::move(buffer.begin(), buffer.end(), std::back_inserter(data));
            }

            const int nb_data = data.size();
            for(int i = 0; i < nb_data; ++i) data[i].lid = i;

            MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);

/*

            std::vector<MPI_Status> statuses(data_to_migrate.size());



            cpt=0;
            while(cpt < nb_neighbors) {// receive the data in any order
                int source_rank, size;
                MPI_Probe(MPI_ANY_SOURCE, 300, LB_COMM, &statuses[cpt]);
                source_rank = statuses[cpt].MPI_SOURCE;
                MPI_Get_count(&statuses[cpt], datatype.elements_datatype, &size);


                std::move(buffer.begin(), buffer.end(), std::back_inserter(data));
                cpt++;
            }

*/
        }


        template<int N>
        void migrate_particles(std::vector<elements::Element<N>> &data,
                               const std::vector<partitioning::geometric::Domain<N>> &domains,
                               const partitioning::CommunicationDatatype datatype,
                               const MPI_Comm LB_COMM,
                               std::vector<size_t> neighbors = std::vector<size_t>()) {
            int wsize;
            MPI_Comm_size(LB_COMM, &wsize);
            int caller_rank;
            MPI_Comm_rank(LB_COMM, &caller_rank);
            size_t batch_size = 0, data_pointer = 0;
            bool empty_recv_buff = false;

            std::vector<elements::Element<N>> buffer;
            std::vector<std::vector<elements::Element<N>>> data_to_migrate;

            if (neighbors.empty())
                neighbors = partitioning::utils::unzip(
                        partitioning::geometric::get_neighboring_domains(caller_rank, domains, 0.08)).first;

            const int nb_of_round = 10;
            const int next_batch_size = std::ceil(data.size() / nb_of_round);

            for (int T = 0; T < nb_of_round; T++) {

                std::tie(data_to_migrate, batch_size, data_pointer) = get_next_batch<N>(data, domains, data_pointer,
                                                                                        next_batch_size, LB_COMM, true);

                auto nb_receiving_neighbors = std::count_if(data_to_migrate.cbegin(),
                                                            data_to_migrate.cend(),
                                                            [](auto data) { return !data.empty(); });

                std::vector<MPI_Request> reqs(nb_receiving_neighbors);
                std::vector<MPI_Request> snd_reqs(nb_receiving_neighbors), rcv_reqs(wsize);
                std::vector<MPI_Status> statuses(neighbors.size());
                // PREPARATION
                int pe_req_idx = 0;
                for (const size_t &PE : neighbors) {
                    if (PE == (size_t) caller_rank) continue;
                    int send_size = data_to_migrate.at(PE).size();
                    if (send_size) {
                        MPI_Issend(&send_size, 1, MPI_INT, PE, 301, LB_COMM, &snd_reqs[pe_req_idx]);
                        pe_req_idx++;
                    }
                }

                std::map<int, int> receive_data_size_lookup;
                for (size_t PE = 0; PE < wsize; ++PE) {
                    receive_data_size_lookup[PE] = 0;
                    MPI_Irecv(&receive_data_size_lookup[PE], 1, MPI_INT, PE, 301, LB_COMM, &rcv_reqs[PE]);
                }

                if (!snd_reqs.empty())
                    MPI_Waitall(snd_reqs.size(), &snd_reqs.front(), MPI_STATUSES_IGNORE);

                MPI_Barrier(LB_COMM);

                snd_reqs.clear(); // Clear requests to wrong PEs
                for (auto &req : rcv_reqs) {
                    int flag;
                    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                    if (!flag) MPI_Cancel(&req);
                }
                rcv_reqs.clear();

                auto nb_sending_neighbors = std::count_if(receive_data_size_lookup.cbegin(),
                                                          receive_data_size_lookup.cend(),
                                                          [](auto pe_data) { return pe_data.second > 0; });

                empty_recv_buff = nb_sending_neighbors == 0;

                int cpt = 0;
                for (const size_t &PE : neighbors) {
                    int send_size = data_to_migrate.at(PE).size();
                    if (send_size) {
                        MPI_Isend(&data_to_migrate.at(PE).front(), send_size, datatype.elements_datatype, PE, 300,
                                  LB_COMM, &reqs[cpt]);
                        cpt++;
                    }
                }

                cpt = 0;
                while (cpt < nb_sending_neighbors && !empty_recv_buff) {// receive the data in any order
                    int source_rank, size;
                    MPI_Probe(MPI_ANY_SOURCE, 300, LB_COMM, &statuses[cpt]);
                    source_rank = statuses[cpt].MPI_SOURCE;
                    MPI_Get_count(&statuses[cpt], datatype.elements_datatype, &size);
                    buffer.resize(size);
                    MPI_Recv(&buffer.front(), size, datatype.elements_datatype, source_rank, 300, LB_COMM,
                             MPI_STATUS_IGNORE);
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(data));
                    cpt++;
                }

                if (!reqs.empty())
                    MPI_Waitall(reqs.size(), &reqs.front(), MPI_STATUSES_IGNORE);
            }
        }

        template<int N>
        class GeometricLoadBalancer
                : public LoadBalancer<partitioning::geometric::PartitionsInfo<N>, elements::Element<N>, partitioning::geometric::Domain<N>> {
        public:
            /**
             * Create a load balancer based on geometric partitioning algorithms
             * @param partitioner Partitioning algorithm
             * @param comm Communication group
             */
            GeometricLoadBalancer(
                    const partitioning::Partitioner<
                            partitioning::geometric::PartitionsInfo<N>,
                            std::vector<elements::Element<N>>,
                            std::array<std::pair<double, double>, N>> &partitioner,
                    MPI_Comm comm) :
                    LoadBalancer<
                            partitioning::geometric::PartitionsInfo<N>,
                            elements::Element<N>,
                            std::array<std::pair<double, double>, N>>(partitioner, comm) {}

            void migrate_particles(std::vector<elements::Element<N>> &data,
                                   const std::vector<partitioning::geometric::Domain<N>> &domains) override {
                const size_t wsize = (size_t) this->world_size;
                std::vector<elements::Element<N>> buffer;
                std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
                auto neighbors = partitioning::utils::unzip(
                        partitioning::geometric::get_neighboring_domains(this->caller_rank, domains, 0.0007)).first;

                for (const size_t &PE : neighbors) {//size_t PE = 0; PE < wsize; ++PE) {
                    if (PE == (size_t) this->caller_rank) continue; //do not check with myself
                    //check within the remaining elements which belong to the current PE
                    size_t data_id = 0;
                    while (data_id < data.size()) {
                        if (elements::is_inside<N>(data.at(data_id), domains.at(PE))) {
                            //if the current element has to be moved, then swap with the last and pop it out (dont need to move the pointer also)
                            //swap iterator values in constant time
                            std::iter_swap(data.begin() + data_id, data.end() - 1);
                            //get the value and push it in the "to migrate" vector
                            data_to_migrate.at(PE).push_back(*(data.end() - 1));
                            //pop the head of the list in constant time
                            data.pop_back();
                        } else data_id++; //if the element must stay with me then check the next one
                    }
                }
                std::vector<MPI_Request> reqs(neighbors.size());
                std::vector<MPI_Status> statuses(neighbors.size());
                int cpt = 0, nb_neighbors = neighbors.size();
                for (const size_t &PE : neighbors) {
                    int send_size = data_to_migrate.at(PE).size();
                    MPI_Isend(&data_to_migrate.at(PE).front(), send_size, this->get_element_datatype(), PE, 300,
                              this->LB_COMM, &reqs[cpt]);
                    cpt++;
                }
                cpt = 0;
                while (cpt < nb_neighbors) {// receive the data in any order
                    int source_rank, size;
                    MPI_Probe(MPI_ANY_SOURCE, 300, this->LB_COMM, &statuses[cpt]);
                    source_rank = statuses[cpt].MPI_SOURCE;
                    MPI_Get_count(&statuses[cpt], this->get_element_datatype(), &size);
                    buffer.resize(size);
                    MPI_Recv(&buffer.front(), size, this->get_element_datatype(), source_rank, 300, this->LB_COMM,
                             &statuses[cpt]);
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(data));
                    cpt++;
                }
                MPI_Waitall(cpt, &reqs.front(), &statuses.front());
            }

            const std::vector<elements::Element<N>> exchange_data(const std::vector<elements::Element<N>> &data,
                                                                  const std::vector<partitioning::geometric::Domain<N>> &domains) override {
                std::vector<elements::Element<N>> buffer;
                std::vector<elements::Element<N>> remote_data_gathered;
                // Get the neighbors
                const size_t wsize = (size_t) this->world_size;
                std::vector<std::vector<elements::Element<N>>> data_to_migrate(wsize);
                auto neighbors = partitioning::utils::unzip(
                        partitioning::geometric::get_neighboring_domains(this->caller_rank, domains, 0.0007)).first;

                for (const size_t &PE : neighbors) {//size_t PE = 0; PE < wsize; ++PE) {
                    if (PE == (size_t) this->caller_rank) continue; //do not check with myself
                    //check within the remaining elements which belong to the current PE
                    size_t data_id = 0;
                    while (data_id < data.size()) {
                        if (elements::distance2<N>(domains.at(PE), data.at(data_id)) < 0.0007) {
                            //get the value and push it in the "to migrate" vector
                            data_to_migrate.at(PE).push_back(data.at(data_id));
                        }
                        data_id++; //if the element must stay with me then check the next one
                    }
                }
                std::vector<MPI_Request> reqs(neighbors.size());
                std::vector<MPI_Status> statuses(neighbors.size());
                int cpt = 0, nb_neighbors = neighbors.size();
                for (const size_t &neighbor_idx : neighbors) {   //give all my data to neighbors
                    int send_size = data_to_migrate.at(neighbor_idx).size();
                    MPI_Isend(&data_to_migrate.at(neighbor_idx).front(), send_size, this->get_element_datatype(),
                              neighbor_idx, 200, this->LB_COMM, &reqs[cpt]);
                    cpt++;
                }
                cpt = 0;
                while (cpt < nb_neighbors) {// receive the data in any order
                    int source_rank, size;
                    MPI_Probe(MPI_ANY_SOURCE, 200, this->LB_COMM, &statuses[cpt]);
                    source_rank = statuses[cpt].MPI_SOURCE;
                    MPI_Get_count(&statuses[cpt], this->get_element_datatype(), &size);
                    buffer.resize(size);
                    MPI_Recv(&buffer.front(), size, this->get_element_datatype(), source_rank, 200, this->LB_COMM,
                             &statuses[cpt]);
                    std::move(buffer.begin(), buffer.end(), std::back_inserter(remote_data_gathered));
                    cpt++;
                }
                MPI_Waitall(reqs.size(), &reqs.front(), &statuses.front()); //less strict than mpi_barrier

                return remote_data_gathered;
            }
        };
    }
}

#endif //NBMPI_LOADBALANCER_HPP
