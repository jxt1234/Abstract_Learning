#ifndef INCLUDE_LEARN_ALIUNSUPERLEARNER_H
#define INCLUDE_LEARN_ALIUNSUPERLEARNER_H
#include "ALISuperviseLearner.h"
class ALIUnSuperLearner:public ALRefCount
{
    public:
        ALIUnSuperLearner(){}
        virtual ~ALIUnSuperLearner(){}
        virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X) const = 0;
};
#endif
