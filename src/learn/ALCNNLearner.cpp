#include "learn/ALCNNLearner.h"
#include "cnn/CNNPredictor.h"
#include "cnn/CNNDerivativeFunction.h"
#include <string.h>
#include <fstream>
#include "cnn/LayerWrapFactory.h"
using namespace ALCNN;

struct ALCNNLearner::LayerStruct
{
    ALSp<LayerWrap> pFirstLayer;
};


ALCNNLearner::ALCNNLearner(const cJSON* description)
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
    }
    ALASSERT(NULL!=layer);
    ALSp<LayerWrap> firstLayer = LayerWrapFactory::create(layer);
    auto ow = firstLayer->getLastLayer()->outputWidth();
    mGDMethod = ALIGradientDecent::create(ALIGradientDecent::SGD, mBatchSize);
    mDetFunction = new CNNDerivativeFunction(firstLayer, ow);
    mProp = ALFloatMatrix::create(ow, 1);
    auto p = mProp->vGetAddr();
    for (size_t i=0; i<ow; ++i)
    {
        p[i] = i;
    }
    mLayerPredict = new LayerStruct;
    mLayerPredict->pFirstLayer = firstLayer;
}
ALCNNLearner::~ ALCNNLearner()
{
    delete mLayerPredict;
}

ALIMatrixPredictor* ALCNNLearner::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(Y->height() == X->height());
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    ALSp<ALFloatMatrix> prop = mProp;
    
    /*Prepare Data*/
    ALSp<ALFloatMatrix> Y_Expand = ALFloatMatrix::create(prop->width(), Y->height());
    ALFloatMatrix::typeExpand(Y_Expand.get(), YT.get());
    ALSp<ALFloatMatrix> Merge = ALFloatMatrix::unionHorizontal(Y_Expand.get(), X);
    
    /*Optimize parameters*/
    auto parameterSize = mLayerPredict->pFirstLayer->getParameterSize();
    ALSp<ALFloatMatrix> coefficient = ALFloatMatrix::create(parameterSize, 1);
    /*Init parameters randomly*/
    mDetFunction->vInitParameters(coefficient.get());
    mGDMethod->vOptimize(coefficient.get(), Merge.get(), mDetFunction.get(), 0.35, mIteration);
    mLayerPredict->pFirstLayer->mapParameters(coefficient.get(), 0);
    
    return new CNNPredictor(mLayerPredict->pFirstLayer, prop, coefficient);
}
ALGradientMethod* ALCNNLearner::getGDMethod() const
{
    ALGradientMethod* result = new ALGradientMethod;
    result->gd = mGDMethod;
    result->det = mDetFunction;
    result->alpha = 0.35;
    result->iteration = mIteration;
    result->type = ALGradientMethod::CLASSIFY;
    result->typeNumber = (int)mProp->width();
    return result;
}
ALIMatrixPredictor* ALCNNLearner::load(const ALFloatMatrix* P)
{
    ALASSERT(NULL!=P);
    ALASSERT(1==P->height());
    ALSp<ALFloatMatrix> parameters = ALFloatMatrix::create(P->width(), 1);
    ALFloatMatrix::copy(parameters.get(), P);
    mLayerPredict->pFirstLayer->mapParameters(parameters.get(), 0);
    return new CNNPredictor(mLayerPredict->pFirstLayer, mProp, parameters);
}
