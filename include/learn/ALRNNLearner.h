#ifndef LEARN_ALRNNLEARNER_H
#define LEARN_ALRNNLEARNER_H
#include "ALIChainLearner.h"
#include "math/ALIGradientDecent.h"
#include "cJSON/cJSON.h"
#include "core/ALARStructure.h"
#include "core/ALBasicExpander.h"
class ALRNNLearner : public ALIChainLearner
{
    public:
        class LayerStruct;
        ALRNNLearner(const cJSON* info);
        virtual ~ALRNNLearner();
        virtual ALFloatPredictor* vLearn(const ALLabeldData* data) const override;
    private:
        ALSp<ALIGradientDecent> mGDMethod;
        ALSp<ALIGradientDecent::DerivativeFunction> mDetFunction;
        unsigned int mIteration;
        unsigned int mBatchSize;
        ALARStructure mAR;
        ALSp<ALIExpander> mExpander;
        LayerStruct* mLayerPredict;
};
#endif
