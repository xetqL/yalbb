//
// Created by xetql on 7/21/21.
//

#ifndef YALBB_STEP_PRODUCER_HPP
#define YALBB_STEP_PRODUCER_HPP
#include <vector>

template<class NumericType>
class StepProducer {
    const std::vector<std::pair<NumericType, unsigned>> steps_repetition;
    unsigned i = 0;
    NumericType step = 0;
    typename decltype(steps_repetition)::const_iterator current_rep;
public:
    explicit StepProducer(std::vector<std::pair<NumericType, unsigned>> steps_rep) :
            steps_repetition(std::move(steps_rep)) {
        current_rep = steps_repetition.begin();
    }

    NumericType next() {
        step += current_rep->first;
        i++;
        if(i >= current_rep->second){
            current_rep++;
            i=0;
        }
        return step;
    }

    bool finished() const {
        return current_rep == steps_repetition.end();
    }
};

#endif //YALBB_STEP_PRODUCER_HPP
