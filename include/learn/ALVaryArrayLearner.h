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
        void train(const ALVaryArray* array);
        ALINT predict(const ALVaryArray::Array& array);
    private:
        ALSp<ALIGradientDecent> mGDMethod;
        ALSp<ALIGradientDecent::DerivativeFunction> mDetFunction;
        unsigned int mIteration;
        unsigned int mBatchSize;
        size_t mTime;
        size_t mNumber;
        LayerStruct* mLayerPredict;
        ALSp<ALFloatMatrix> mCoeffecient;
};
#endif
