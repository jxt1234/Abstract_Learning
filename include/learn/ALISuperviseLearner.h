#ifndef INCLUDE_LEARN_ALIFLOATLEARNER_H
#define INCLUDE_LEARN_ALIFLOATLEARNER_H
#include "core/ALFloatDataChain.h"
#include "core/ALIExpander.h"
#include "math/ALFloatMatrix.h"

class ALIMatrixPredictor:public ALRefCount
{
public:
    ALIMatrixPredictor(){}
    virtual ~ALIMatrixPredictor(){}
    /*Assert X->height() == Y->height()*/
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const = 0;
    
    /*For Classify*/
    virtual void vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const {}
    virtual const ALFloatMatrix* vGetPossiableValues() const {return NULL;}
    virtual void vPrint(std::ostream& output) const{}
};

class ALDummyMatrixPredictor:public ALIMatrixPredictor
{
public:
    ALDummyMatrixPredictor(){}
    virtual ~ALDummyMatrixPredictor(){}
    /*Assert X->height() == Y->height()*/
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        auto h = X->height();
        ALASSERT(Y->width()>0);
        ALASSERT(h == Y->height());
        for (size_t i=0; i<h; ++i)
        {
            auto y = Y->vGetAddr(i);
            y[0] = 0.0;
        }
    }
    virtual void vPrint(std::ostream& output) const{}
};
class ALISuperviseLearner:public ALRefCount
{
public:
    ALISuperviseLearner(){}
    virtual ~ALISuperviseLearner(){}
    /*Assert X->height() == Y->height()*/
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const = 0;
};
#endif
