//
// Created by xetql on 03.09.18.
//

#ifndef NBMPI_FEATURE_CONTAINER_HPP
#define NBMPI_FEATURE_CONTAINER_HPP

class FeatureContainer {
public:
    virtual std::vector<double> get_features() = 0;
    virtual int get_target() = 0;
};

#endif //NBMPI_FEATURE_CONTAINER_HPP
