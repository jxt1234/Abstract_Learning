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

ALCNNLearner::ALCNNLearner(const cJSON* description, unsigned int iteration)
{
    ALASSERT(NULL!=description);
    cJSON* layer = NULL;
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
    mIteration = iteration;
    
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
    ALSp<ALFloatMatrix> prop = ALFloatMatrix::genTypes(YT.get());
    
    /*Prepare Data*/
    ALSp<ALFloatMatrix> Y_Expand = ALFloatMatrix::create(prop->width(), Y->height());
    ALFloatMatrix::zero(Y_Expand.get());
    auto yh = Y->height();
    auto yp = prop->vGetAddr();
    auto yw = prop->width();
    for (int i=0; i<yh; ++i)
    {
        auto y = Y->vGetAddr(i);
        auto ye = Y_Expand->vGetAddr(i);
        for (int k=0; k<yw; ++k)
        {
            if (ZERO(yp[k]-y[0]))
            {
                ye[k] = 1.0f;
                break;
            }
        }
    }
    ALSp<ALFloatMatrix> Merge = ALFloatMatrix::unionHorizontal(Y_Expand.get(), X);
    
    /*Optimize parameters*/
    auto parameterSize = mLayerPredict->pFirstLayer->getParameterSize();
    ALSp<ALFloatMatrix> coefficient = ALFloatMatrix::create(parameterSize, 1);
    /*Init parameters randomly*/
    auto c = coefficient->vGetAddr();
    for (int i=0; i<parameterSize; ++i)
    {
        c[i] = 0.1*ALRandom::rate()-0.05;
    }
    mGDMethod->vOptimize(coefficient.get(), Merge.get(), mDetFunction.get(), 0.35, mIteration);
    mLayerPredict->pFirstLayer->setParameters(coefficient.get(), 0);
    
    return new CNNPredictor(mLayerPredict->pFirstLayer, prop);
}
