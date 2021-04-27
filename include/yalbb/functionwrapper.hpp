//
// Created by xetql on 11/27/20.
//
#pragma once

template<class GetPosPtrFunc, class GetVelPtrFunc, class UnaryForceFunc, class BinaryForceFunc, class BoxIntersectionFunc, class PointAssignationFunc, class LoadBalancingFunc>
class FunctionWrapper {
    GetPosPtrFunc posPtrFunc;
    GetVelPtrFunc velPtrFunc;
    UnaryForceFunc unaryForceFunc;
    BinaryForceFunc forceFunc;
    BoxIntersectionFunc boxIntersectionFunc;
    PointAssignationFunc pointAssignationFunc;
    LoadBalancingFunc loadBalancingFunc;
public:
    FunctionWrapper(GetPosPtrFunc posPtrFunc, GetVelPtrFunc velPtrFunc, UnaryForceFunc unaryForceFunc, BinaryForceFunc forceFunc,
                    BoxIntersectionFunc boxIntersectionFunc, PointAssignationFunc pointAssignationFunc,
                    LoadBalancingFunc loadBalancingFunc) : posPtrFunc(posPtrFunc), velPtrFunc(velPtrFunc), unaryForceFunc(unaryForceFunc),
                                                           forceFunc(forceFunc),
                                                           boxIntersectionFunc(boxIntersectionFunc),
                                                           pointAssignationFunc(pointAssignationFunc),
                                                           loadBalancingFunc(loadBalancingFunc) {}

    const GetPosPtrFunc &getPosPtrFunc() const {
        return posPtrFunc;
    }

    void setPosPtrFunc(const GetPosPtrFunc &posPtrFunc) {
        FunctionWrapper::posPtrFunc = posPtrFunc;
    }

    const GetVelPtrFunc &getVelPtrFunc() const {
        return velPtrFunc;
    }

    void setVelPtrFunc(const GetVelPtrFunc &velPtrFunc) {
        FunctionWrapper::velPtrFunc = velPtrFunc;
    }

    BinaryForceFunc getForceFunc() const {
        return forceFunc;
    }

    void setForceFunc(BinaryForceFunc forceFunc) {
        FunctionWrapper::forceFunc = forceFunc;
    }

    BoxIntersectionFunc getBoxIntersectionFunc() const {
        return boxIntersectionFunc;
    }

    void setBoxIntersectionFunc(BoxIntersectionFunc boxIntersectionFunc) {
        FunctionWrapper::boxIntersectionFunc = boxIntersectionFunc;
    }

    PointAssignationFunc getPointAssignationFunc() const {
        return pointAssignationFunc;
    }

    void setPointAssignationFunc(PointAssignationFunc pointAssignationFunc) {
        FunctionWrapper::pointAssignationFunc = pointAssignationFunc;
    }

    LoadBalancingFunc getLoadBalancingFunc() const {
        return loadBalancingFunc;
    }

    void setLoadBalancingFunc(LoadBalancingFunc loadBalancingFunc) {
        FunctionWrapper::loadBalancingFunc = loadBalancingFunc;
    }

    UnaryForceFunc getUnaryForceFunc() const {
        return unaryForceFunc;
    }

    void setUnaryForceFunc(UnaryForceFunc unaryForceFunc) {
        FunctionWrapper::unaryForceFunc = unaryForceFunc;
    }
};
