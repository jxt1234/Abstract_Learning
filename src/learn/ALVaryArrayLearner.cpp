#include "cnn/LayerWrapFactory.h"
#include "cnn/CNNDerivativeFunction.h"
#include "learn/ALVaryArrayLearner.h"
using namespace ALCNN;
struct ALVaryArrayLearner::LayerStruct
{
    ALSp<LayerWrap> pFirstLayer;
};
ALVaryArrayLearner::ALVaryArrayLearner(cJSON* description)
{
    ALASSERT(NULL!=description);
    cJSON* layer = NULL;
    mIteration = 1000;
    for (auto c = description->child; NULL!=c; c=c->next)
    {
        if (strcmp(c->string, "train-batch") == 0)
        {
            mBatchSize = c->valueint;
        }
        else if (strcmp(c->string, "layers")==0)
        {
            layer = c->child;
        }
        else if (strcmp(c->string, "iteration")==0)
        {
            mIteration = c->valueint;
        }
        else if (strcmp(c->string, "time")==0)
        {
            mTime = c->valueint;
        }
        else if (strcmp(c->string, "width")==0)
        {
            mNumber = c->valueint;
        }
        else if (strcmp(c->string, "prop")==0)
        {
            mPropWidth = c->valueint;
        }
    }
    if (0 == mPropWidth)
    {
        mPropWidth = mNumber;
    }
    ALASSERT(NULL!=layer);
    ALSp<LayerWrap> firstLayer = LayerWrapFactory::create(layer);
    auto ow = mPropWidth;
    mGDMethod = ALIGradientDecent::create(ALIGradientDecent::SGD, mBatchSize);
    mDetFunction = new CNNDerivativeFunction(firstLayer, ow);
    mLayerPredict = new LayerStruct;
    mLayerPredict->pFirstLayer = firstLayer;
    auto parameterSize = mLayerPredict->pFirstLayer->getParameterSize();
    mCoeffecient = ALFloatMatrix::create(parameterSize, 1);
}
ALVaryArrayLearner::~ALVaryArrayLearner()
{
    delete mLayerPredict;
}
void ALVaryArrayLearner::train(const ALVaryArray* array, const ALFloatMatrix* label)
{
    size_t time = mTime;
    ALSp<ALFloatMatrix> labelExpand;
    if (NULL == label)
    {
        time = mTime+1;//Last for predict
    }
    else
    {
        ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(label);
        ALSp<ALFloatMatrix> Y_Expand = ALFloatMatrix::create(mPropWidth, label->height());
        ALFloatMatrix::typeExpand(Y_Expand.get(), YT.get());
        labelExpand = Y_Expand;
    }
    ALSp<ALFloatMatrix> varyMatrix = new ALVaryArrayMatrix(array, time, mNumber, labelExpand.get());
    mDetFunction->vInitParameters(mCoeffecient.get());
    mGDMethod->vOptimize(mCoeffecient.get(), varyMatrix.get(), mDetFunction.get(), 0.35, mIteration);
    mLayerPredict->pFirstLayer->mapParameters(mCoeffecient.get(), 0);
}

ALINT ALVaryArrayLearner::predict(const ALVaryArray::Array& array)
{
    return 0;
}
