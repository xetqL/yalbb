//
// Created by xetql on 4/30/20.
//

#ifndef NBMPI_CLL_HPP
#define NBMPI_CLL_HPP

#include "utils.hpp"
#include "coordinate_translater.hpp"
#include "bit.hpp"

constexpr Integer EMPTY = -1;

template<int N, class T, class GetPositionFunc>
inline void CLL_update(Integer acc,
                       std::initializer_list<std::pair<T*, size_t>>&& elements,
                       GetPositionFunc getPositionFunc,
                       const BoundingBox<N>& bbox, Real rc,
                       std::vector<Integer> *head,
                       std::vector<Integer> *lscl) {
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);
    Integer c;
    for (const auto &span : elements) {
        auto el_ptr = span.first;
        auto n_els = span.second;
        for (size_t i = 0; i < n_els; ++i) {
            auto &pos = *getPositionFunc(&el_ptr[i]);
            if (is_within<N>(bbox, pos)) {
                c = CoordinateTranslater::translate_position_into_local_index<N>(pos, rc, bbox, lc[0], lc[1]);
                lscl->at(i + acc) = head->at(c);
                head->at(c) = i + acc;
            }
        }
        acc += n_els;
    }
}

template<int N, class T, class GetPositionFunc>
inline void CLL_update(Integer acc,
                       std::initializer_list<std::pair<T*, size_t>>&& elements,
                       GetPositionFunc getPositionFunc,
                       const BoundingBox<N>& bbox, Real rc,
                       std::vector<Integer> *head,
                       std::vector<Integer> *lscl,
                       std::vector<Integer> *non_empty_boxes) {
    auto lc = get_cell_number_by_dimension<N>(bbox, rc);
    Integer c;
    const auto nmask = std::ceil(head->size() / 64) * 64;
    std::vector<uint64_t> boxes_non_empty_status(nmask, 0);
    for (const auto &span : elements) {
        auto el_ptr = span.first;
        auto n_els = span.second;
        for (size_t i = 0; i < n_els; ++i) {
            auto &pos = *getPositionFunc(&el_ptr[i]);
            if (is_within<N>(bbox, pos)) {
                c = CoordinateTranslater::translate_position_into_local_index<N>(pos, rc, bbox, lc[0], lc[1]);
                enable_bit(boxes_non_empty_status, c);
                lscl->at(i + acc) = head->at(c);
                head->at(c) = i + acc;
            }
        }
        acc += n_els;
    }
    auto nbits = head->size();
    non_empty_boxes->reserve(nbits / 2);
    for(Integer i = 0; i < nbits; ++i){
        if(is_enabled(boxes_non_empty_status, i))
            non_empty_boxes->push_back(i);
    }
    non_empty_boxes->shrink_to_fit();
}

template<int N, class T, class GetPositionFunc>
inline void CLL_init(std::initializer_list<std::pair<T*, size_t>>&& elements,
                     GetPositionFunc getPositionFunc,
                     const BoundingBox<N>& bbox, Real rc,
                     std::vector<Integer> *head,
                     std::vector<Integer> *lscl,
                     std::vector<Integer> *non_empty_boxes) {
    apply_resize_strategy(head, get_total_cell_number<N>(bbox, rc));
    std::fill(head->begin(), head->end(), EMPTY);
    CLL_update<N, T, GetPositionFunc>(0, std::move(elements), getPositionFunc, bbox, rc, head, lscl, non_empty_boxes);
}

template<int N, class T, class GetPositionFunc>
inline void CLL_init(std::initializer_list<std::pair<T*, size_t>>&& elements,
              GetPositionFunc getPositionFunc,
              const BoundingBox<N>& bbox, Real rc,
              std::vector<Integer> *head,
              std::vector<Integer> *lscl) {
    apply_resize_strategy(head, get_total_cell_number<N>(bbox, rc));
    std::fill(head->begin(), head->end(), EMPTY);
    CLL_update<N, T, GetPositionFunc>(0, std::move(elements), getPositionFunc, bbox, rc, head, lscl);
}

template<int N, class T>
inline void CLL_append(const Integer i, const Integer cell, const T& element, std::vector<Integer>* head, std::vector<Integer>* lscl) {
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
    Integer c1, ic1[3], j;
    std::array<Integer, 3> ic;
    const T *source, *receiver;
    Integer cmplx = n_elements;
    for (size_t i = 0; i < n_elements; ++i) {
        const auto& pos = *getPosFunc(const_cast<T*>(&elements[i]));
        receiver = &elements[i];
        for(int d = 0; d < 3; ++d)
            ic[d] = (pos[d]-bbox[2*d]) / rc;

        for (ic1[0] = ic[0] - 1; ic1[0] <= (ic[0]+1); ic1[0]++) {
            for (ic1[1] = ic[1] - 1; ic1[1] <= (ic[1] + 1); ic1[1]++) {
                for (ic1[2] = ic[2] - 1; ic1[2] <= (ic[2] + 1); ic1[2]++) {
                    /* this is for bounce back, to avoid heap-buffer over/under flow */
                    if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) || (ic1[2] < 0 || ic1[2] >= lc[2])) continue;
                    c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                    j = head->at(c1);
                    while(j != EMPTY) {
                        if(i != j) {
                            source = j < n_elements ? &elements[j] : &remote_elements[j - n_elements];
                            std::array<Real, 3> force = computeForceFunc(receiver, source);
                            for (int dim = 0; dim < 3; ++dim) {
                                acc->at(3 * i + dim) += force[dim];
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

template<class T, class GetPositionFunc, class BinaryOp>
Integer CLL_foreach_interaction(const T *elements, Integer n_elements,
                    const T *remote_elements,
                    GetPositionFunc getPosFunc,
                    const BoundingBox<3>& bbox, Real rc,
                    const std::vector<Integer> *head, const std::vector<Integer> *lscl,
                    BinaryOp binary_op) {
    auto lc = get_cell_number_by_dimension<3>(bbox, rc);
    Integer c1, ic1[3], j;
    std::array<Integer, 3> ic;
    const T *source, *receiver;
    Integer cmplx = n_elements;
    for (size_t i = 0; i < n_elements; ++i) {
        const auto& pos = *getPosFunc(const_cast<T*>(&elements[i]));
        receiver = &elements[i];

        for(int d = 0; d < 3; ++d) ic[d] = (pos[d]-bbox[2*d]) / rc;

        for (ic1[0] = ic[0] - 1; ic1[0] <= (ic[0]+1); ic1[0]++) {
            for (ic1[1] = ic[1] - 1; ic1[1] <= (ic[1] + 1); ic1[1]++) {
                for (ic1[2] = ic[2] - 1; ic1[2] <= (ic[2] + 1); ic1[2]++) {
                    /* this is for bounce back, to avoid heap-buffer over/under flow */
                    if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1]) || (ic1[2] < 0 || ic1[2] >= lc[2])) continue;
                    c1 = (ic1[0]) + (lc[0] * ic1[1]) + (lc[0] * lc[1] * ic1[2]);
                    j = head->at(c1);
                    while(j != EMPTY) {
                        if(i != j) {
                            source = j < n_elements ? &elements[j] : &remote_elements[j - n_elements];
                            binary_op(receiver, source);
                        }
                        j = lscl->at(j);
                    }
                }
            }
        }
    }
    return cmplx;
}

template<class T, class GetPositionFunc, class ComputeForceFunc>
Integer CLL_compute_forces2d(std::vector<Real>* acc,
                             const T *elements, Integer n_elements,
                             const T *remote_elements,
                             GetPositionFunc getPosFunc,
                             const BoundingBox<2>& bbox, Real rc,
                             const std::vector<Integer> *head, const std::vector<Integer> *lscl,
                             ComputeForceFunc computeForceFunc) {
    auto lc = get_cell_number_by_dimension<2>(bbox, rc);
    Integer c1, ic1[2], j;
    std::array<Integer, 2> ic;
    const T *source, *receiver;
    Integer cmplx = n_elements;
    for (size_t i = 0; i < n_elements; ++i) {
        const auto& pos = *getPosFunc(const_cast<T*>(&elements[i]));
        receiver = &elements[i];

        for(int d = 0; d < 2; ++d)
            ic[d] = (pos[d]-bbox[2*d]) / rc;

        for (ic1[0] = ic[0] - 1; ic1[0] <= (ic[0]+1); ic1[0]++) {
            for (ic1[1] = ic[1] - 1; ic1[1] <= (ic[1] + 1); ic1[1]++) {
                /* this is for bounce back, to avoid heap-buffer over/under flow */
                if((ic1[0] < 0 || ic1[0] >= lc[0]) || (ic1[1] < 0 || ic1[1] >= lc[1])) continue;

                c1 = (ic1[0]) + (lc[0] * ic1[1]);
                j = head->at(c1);
                while(j != EMPTY) {
                    source = j < n_elements ? &elements[j] : &remote_elements[j - n_elements];
                    if(i != j) {
                        std::array<Real, 2> force = computeForceFunc(receiver, source);
                        for (int dim = 0; dim < 2; ++dim) {
                            acc->at(2 * i + dim) += force[dim];
                        }
                        cmplx++;
                    }
                    j = lscl->at(j);
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
    //std::memset(acc->data(), 0.0f, sizeof(Real) * acc->size());
    if constexpr(N==3) {
        return CLL_compute_forces3d(acc, loc_el.data(), loc_el.size(), rem_el.data(), getPosFunc, bbox, rc, head, lscl, computeForceFunc);
    } else {
        return CLL_compute_forces2d(acc, loc_el.data(), loc_el.size(), rem_el.data(), getPosFunc, bbox, rc, head, lscl, computeForceFunc);
    }
}

#endif //NBMPI_CLL_HPP
