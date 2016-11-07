#include "learn/ALCNNLearner.h"
#include "cnn/CNNPredictor.h"
#include "cnn/CNNDerivativeFunction.h"
#include "cnn/MeanPoolLayer.h"
#include "cnn/CNNLayer.h"
#include "cnn/SoftMaxLayer.h"
#include "cnn/ReluLayer.hpp"
#include "cnn/MaxPoolLayer.hpp"
#include "cnn/InnerProductLayer.hpp"
#include <fstream>
#include <string.h>
#include "cnn/LayerFactoryRegistor.hpp"
using namespace ALCNN;

struct ALCNNLearner::LayerStruct
{
    ALSp<LayerWrap> pFirstLayer;
};

static void _readLayerParameters(cJSON* layer, LayerParameters& p, std::string& type)
{
    ALASSERT(NULL!=layer);
    p.mIntValues.clear();
    for (auto c = layer->child; NULL!=c; c=c->next)
    {
        auto name = c->string;
        if (!strcmp(name, "type"))
        {
            type = c->valuestring;
            continue;
        }
        if (!strcmp(name, "input"))
        {
            p.uInputSize = c->valueint;
            continue;
        }
        if (!strcmp(name, "output"))
        {
            p.uOutputSize = c->valueint;
            continue;
        }
        if (!strcmp(name, "input_3D"))
        {
            auto ac = c->child;
            ALASSERT(NULL!=ac);
            p.mMatrixInfo.iWidth = ac->valueint;
            ac = ac->next;
            ALASSERT(NULL!=ac);
            p.mMatrixInfo.iHeight = ac->valueint;
            ac = ac->next;
            ALASSERT(NULL!=ac);
            p.mMatrixInfo.iDepth = ac->valueint;
            continue;
        }
        p.mIntValues.insert(std::make_pair(name, c->valueint));
    }
}

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
    ALSp<LayerWrap> firstLayer;
    ALSp<LayerWrap> currentLayer;
    ALSp<LayerWrap> lastLayer;
    ALSp<LayerWrap> nextLayer;
    
    LayerParameters parameters;
    std::string type = "";
    /*Construct first layer*/
    _readLayerParameters(layer, parameters, type);
    firstLayer = new LayerWrap(LayerFactory::get()->create(type.c_str(), parameters));
    currentLayer = firstLayer;
    
    mInputSize = parameters.uInputSize;
    
    for (layer=layer->next;layer!=NULL; layer=layer->next)
    {
        _readLayerParameters(layer, parameters, type);
        nextLayer = new LayerWrap(LayerFactory::get()->create(type.c_str(), parameters));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
    }
    lastLayer = currentLayer;
    mGDMethod = ALIGradientDecent::create(ALIGradientDecent::SGD, mBatchSize);
    mDetFunction = new CNNDerivativeFunction(firstLayer, lastLayer, parameters.uOutputSize);
    mProp = ALFloatMatrix::create(parameters.uOutputSize, 1);
    auto p = mProp->vGetAddr();
    for (size_t i=0; i<parameters.uOutputSize; ++i)
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
    ALASSERT(X->width() == mInputSize);
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
