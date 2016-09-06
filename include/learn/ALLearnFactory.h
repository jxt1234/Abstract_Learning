#ifndef INCLUDE_LEARN_ALLEARNFACTORY_H
#define INCLUDE_LEARN_ALLEARNFACTORY_H
#include "learn/ALISuperviseLearner.h"
#include "learn/ALIChainLearner.h"
#include "core/ALIExpander.h"
#include "math/ALFloatMatrix.h"
#include "ALHead.h"
class ALLearnFactory
{
public:
    static ALISuperviseLearner* createLearner();
    static ALFLOAT crossValidate(ALIChainLearner* l, const ALLabeldData* c, int div = 10);
    static ALFLOAT crossValidateForClassify(ALIChainLearner* l, const ALLabeldData* c, int div=10);
    static ALFLOAT crossValidate(ALISuperviseLearner* l, const ALFloatMatrix* X, const ALFloatMatrix* Y, int div = 10);
    static ALFLOAT crossValidateForClassify(ALISuperviseLearner* l, const ALFloatMatrix* X, const ALFloatMatrix* Y, int div = 10);
};

#endif
