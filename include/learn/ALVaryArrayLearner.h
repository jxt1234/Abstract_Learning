#ifndef LEARN_ALVARYARRAYLEARNER_H
#define LEARN_ALVARYARRAYLEARNER_H
#include "data/ALVaryArrayMatrix.h"
#include "cJSON/cJSON.h"
#include "math/ALIGradientDecent.h"
class ALVaryArrayLearner : public ALRefCount
{
public:
    class LayerStruct;
    ALVaryArrayLearner(cJSON* json);
    virtual ~ ALVaryArrayLearner();
    void train(const ALVaryArray* array, const ALFloatMatrix* label = NULL);
    void predict(const ALVaryArray* array, ALFloatMatrix* dst);
private:
    ALSp<ALIGradientDecent> mGDMethod;
    ALSp<ALIGradientDecent::DerivativeFunction> mDetFunction;
    unsigned int mIteration;
    unsigned int mBatchSize;
    ALFLOAT mAlpha = 0.35;
    size_t mTime = 0;
    size_t mPropWidth = 0;
    LayerStruct* mLayerPredict;
    ALSp<ALFloatMatrix> mCoeffecient;
};
#endif
