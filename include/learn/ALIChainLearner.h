#ifndef INCLUDE_LEARN_ALICHAINLEARNER_H
#define INCLUDE_LEARN_ALICHAINLEARNER_H
#include "core/ALFloatDataChain.h"
#include "core/ALIExpander.h"
#include "ALIUnSuperLearner.h"
class ALIChainLearner:public ALRefCount
{
public:
    struct Error
    {
        ALFLOAT sum;
        int num;
    };
    ALIChainLearner(){}
    virtual ~ALIChainLearner(){}
    virtual ALFloatPredictor* vLearn(const ALLabeldData* data) const = 0;
    static Error computeError(const ALLabeldData* data, const ALFloatPredictor* p);
    static ALIChainLearner* createFromBasic(ALSp<ALISuperviseLearner> basic, ALSp<ALIExpander> Xe);
};
#endif
