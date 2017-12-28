//
// Created by xetql on 12.12.17.
//

#ifndef NBMPI_SPATIALBISECTION_HPP
#define NBMPI_SPATIALBISECTION_HPP

#include <vector>
#include <algorithm>
#include <memory>
#include <tuple>
#include <ostream>

#include "utils.hpp"
#include "partitioner.hpp"

namespace partitioning { namespace geometric {

    using PartitionID=int;

    template<int N>
    struct Element {
        std::array<double, N> position,  velocity;
        constexpr Element(std::array<double, N> p, std::array<double, N> v) : position(p), velocity(v){}
        constexpr Element() {
            std::fill(velocity.begin(), velocity.end(), 0.0);
            std::fill(position.begin(), position.end(), 0.0);
        }
        static constexpr int size() {
            return N * 2.0;
        }

        /**
         * Equality of two elements regarding the VALUE of their properties
         * @param rhs another element
         * @return true if the position and the velocity of the two elements are equals
         */
        bool operator==(const Element &rhs) const {
            return position == rhs.position && velocity == rhs.velocity;
        }

        bool operator!=(const Element &rhs) const {
            return !(rhs == *this);
        }

        friend std::ostream &operator<<(std::ostream &os, const Element &element) {
            std::string pos = "("+std::to_string(element.position.at(0));
            for(int i = 1; i < N; i++){
                pos += "," + std::to_string(element.position.at(i));
            }
            pos += ")";
            std::string vel = "("+std::to_string(element.velocity.at(0));
            for(int i = 1; i < N; i++){
                vel += "," + std::to_string(element.velocity.at(i));
            }
            vel += ")";
            os << "position: " << pos << " velocity: " << vel;
            return os;
        }

    };

    template<int N>
    struct PartitionsInfo {
        const std::vector<std::pair<PartitionID, Element<N>>> parts;
        PartitionsInfo(const std::vector<std::pair<PartitionID, Element<N>>> &partitions) : parts(partitions){}
    };

    template<int N>
    struct BisectionInfo{
        const std::vector<std::pair<Element<N>, int>> left, right;
        BisectionInfo(const std::vector<std::pair<Element<N>, int>> left, const std::vector<std::pair<Element<N>, int>> right)
                : left(left), right(right) {}
    };

    /**
     * Bisect a dataset in a given dimension at the median point in the dataset
     * Time: Upper bound is O(n.lg(n)) to retrieve the median point
     * @param points
     * @param IDs
     * @param dim
     * @return The two new partitions and their points
     */
    template<int N>
    std::unique_ptr<BisectionInfo<N>> bisect(std::vector<Element<N>> points, const std::vector<int>& IDs, int dim){
        std::vector<std::pair<Element<N>, int>> left, right;
        auto zipped_points = partitioning::utils::zip(points, IDs); //O(n)
        //get the median by sorting in O(n.lg(n))
        //TODO: Median can be retrieved in O(n) c.f Get median of median etc.
        std::sort(zipped_points.begin(), zipped_points.end(),
                  [dim](const std::pair<Element<N>, int> & a, const std::pair<Element<N>, int> & b) -> bool{
                      return a.first.position.at(dim) < b.first.position.at(dim);
                  });
        unsigned int median_idx = points.size() / 2;
        //bind to partition in O(n)
        for(unsigned int i = 0; i < zipped_points.size(); ++i){
            if(i < median_idx) left.push_back(zipped_points.at(i));
            else right.push_back(zipped_points.at(i));
        }
        return std::make_unique<BisectionInfo<N>>(left, right);
    }

    template<int N>
    class SeqSpatialBisection : public Partitioner<PartitionsInfo<N>, std::vector<Element<N>>> {
    public:

        virtual std::unique_ptr<PartitionsInfo<N>> partition_data(
                std::vector<Element<N>> spatial_data,
                int number_of_partitions) const {

            std::vector<std::pair<std::vector<Element<N>>, std::vector<int>>> domain;
            std::vector<int> range(spatial_data.size());

            std::iota(range.begin(), range.end(), 0); //generate ids of particles from 0 to N-1
            domain.push_back(std::make_pair(spatial_data, range)); //the particles and their id.
            std::vector<std::unique_ptr<BisectionInfo<N>>> info;

            unsigned int logPE = std::log(number_of_partitions) / std::log(2);

            //O(lg(p)*2^lg(p-1)*n.lg(n)), P is the number of partitions
            for(unsigned int depth = 0; depth < logPE; ++depth) {
                unsigned int dim = depth % N;
                std::vector<std::unique_ptr<BisectionInfo<N>>> bisections;
                for(std::pair<std::vector<Element<N>>, std::vector<int>> const& section : domain) {
                    bisections.push_back(std::move(bisect<N>(section.first, section.second, dim)));
                }
                domain.clear();
                for(auto const& subsection: bisections){
                    domain.push_back( partitioning::utils::unzip(subsection->left) );
                    domain.push_back( partitioning::utils::unzip(subsection->right) );
                }
            }

            std::vector<std::pair<PartitionID , Element<N>>> partitions(spatial_data.size());
            PartitionID pID = 0;
            //O(p*n) => O(n) p <<< n
            for(auto const& partition: domain){
                auto zipped = partitioning::utils::zip(partition.first, partition.second);
                for(auto const& element : zipped){
                    partitions[element.second] = std::make_pair(pID, element.first);
                }
                pID++;
            }
            return std::move(std::make_unique<PartitionsInfo<N>>(partitions));
        }
    };

}}

#endif //NBMPI_SPATIALBISECTION_HPP
