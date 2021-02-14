//
// Created by xetql on 2/14/21.
//

#ifndef NBMPI_BIT_HPP
#define NBMPI_BIT_HPP
bool is_enabled(uint64_t mask, uint64_t i) {
    return mask & 1<<i;
}
bool is_enabled(const std::vector<uint64_t>& mask, uint64_t i){
    auto& cmask = mask.at(i / 64);
    return is_enabled(cmask, i % 64);
}
void enable_bit(uint64_t& mask, uint64_t i) {
    mask = mask | 1<<i;
}
void enable_bit(std::vector<uint64_t>& mask, uint64_t i){
    auto& cmask = mask.at(i / 64);
    enable_bit(cmask, i % 64);
}
#endif //NBMPI_BIT_HPP
