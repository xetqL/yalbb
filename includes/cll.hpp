//
// Created by xetql on 4/30/20.
//

#ifndef NBMPI_CLL_HPP
#define NBMPI_CLL_HPP

#include "utils.hpp"
#include "coordinate_translater.hpp"

constexpr Integer EMPTY = -1;

template<int N, class T, class GetPositionFunc>
void CLL_update(std::initializer_list<std::pair<T*, size_t>>&& element_groups,
                GetPositionFunc getPositionFunc,
                const BoundingBox<N>& bbox, Real rc,
                std::vector<Integer> *head,
                std::vector<Integer> *lscl) {
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);
    Integer c, start_id = 0;
    T* el_ptr;
    size_t n_els, end_el_id, cnt, i;
    for(const auto& span : element_groups){
        el_ptr     = span.first;
        n_els      = span.second;
        end_el_id  = start_id + n_els;
        cnt = 0;
        for (i = start_id; i < end_el_id; ++i, ++cnt) {
            c = CoordinateTranslater::translate_position_into_local_index<N>(*getPositionFunc(el_ptr[cnt]), rc, bbox, lc[0], lc[1]);
            lscl->at(i) = head->at(c);
            head->at(c) = i;
        }
        start_id += n_els;
    }
}

template<int N, class T, class GetPositionFunc>
void CLL_init(std::initializer_list<std::pair<T*, size_t>>&& elements,
              GetPositionFunc getPositionFunc,
              const BoundingBox<N>& bbox, Real rc,
              std::vector<Integer> *head,
              std::vector<Integer> *lscl) {
    Integer lcxyz = get_total_cell_number<N>(bbox, rc);
    std::memset(head->data(), EMPTY, lcxyz * sizeof(Integer));
    CLL_update<N, T, GetPositionFunc>(std::move(elements), getPositionFunc, bbox, rc, head, lscl);
}

template<int N, class T>
inline void CLL_append(const Integer i, const Integer  cell, const T& element, std::vector<Integer>* head, std::vector<Integer>* lscl) {
    lscl->at(i) = head->at(cell);
    head->at(cell) = i;
}

template<class T, class GetPositionFunc, class ComputeForceFunc>
Integer CLL_compute_forces3d(std::vector<Real>* acc,
                             const T *elements, Integer n_elements,
                             const T *remote_elements,
                             GetPositionFunc getPosFunc,
                             const BoundingBox<3>& bbox, Real rc,
                             const std::vector<Integer> *head, const std::vector<Integer> *lscl,
                             ComputeForceFunc computeForceFunc) {
    auto lc = get_cell_number_by_dimension<3>(bbox, rc);
    Integer c, c1, ic[3], ic1[3], j;
    T source, receiver;
    Integer cmplx = n_elements;
    std::fill(acc->begin(), acc->begin()+n_elements, (Real) 0.0);
    for (size_t i = 0; i < n_elements; ++i) {
        const auto& pos = *getPosFunc(const_cast<T&>(elements[i]));
        c = position_to_local_cell_index<3>(pos, rc, bbox, lc[0], lc[1]);
        receiver = elements[i];
        for (auto d = 0; d < 3; ++d)
            ic[d] = c / lc[d];
        for (ic1[0] = ic[0] - 1; ic1[0] < (ic[0]+1); ic1[0]++) {
            for (ic1[1] = ic[1] - 1; ic1[1] < ic[1] + 1; ic1[1]++) {
                for (ic1[2] = ic[2] - 1; ic1[2] < ic[2] + 1; ic1[2]++) {
                    /* this is for bounce back, to avoid heap-buffer over/under flow */
                    if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) || (ic1[2] < 0 || ic1[2] >= lc[2])) continue;
                    c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                    j = head->at(c1);
                    while(j != EMPTY) {
                        if(i < j) {
                            source = j < n_elements ? elements[j] : remote_elements[j - n_elements];
                            std::array<Real, 3> force = computeForceFunc(receiver, source);
                            for (int dim = 0; dim < 3; ++dim) {
                                acc->at(3*i + dim) += force[dim];
                            }
                            cmplx++;
                        }
                        j = lscl->at(j);
                    }
                }
            }
        }
    }
    return cmplx;
}

template<int N, class T, class GetPositionFunc, class ComputeForceFunc>
Integer CLL_compute_forces(std::vector<Real>* acc,
                           const std::vector<T>& loc_el,
                           const std::vector<T>& rem_el,
                           GetPositionFunc getPosFunc,
                           const BoundingBox<N>& bbox, Real rc,
                           const std::vector<Integer> *head, const std::vector<Integer> *lscl,
                           ComputeForceFunc computeForceFunc) {
    if constexpr(N==3) {
        return CLL_compute_forces3d(acc, loc_el.data(), loc_el.size(), rem_el.data(), getPosFunc, bbox, rc, head, lscl, computeForceFunc);
    }else {
        return 0;
    }
}

#endif //NBMPI_CLL_HPP
