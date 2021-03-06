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
        else if (strcmp(c->string, "prop")==0)
        {
            mPropWidth = c->valueint;
        }
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
    ALSp<ALFloatMatrix> varyMatrix = new ALVaryArrayMatrix(array, time, labelExpand.get());
    mDetFunction->vInitParameters(mCoeffecient.get());
    mGDMethod->vOptimize(mCoeffecient.get(), varyMatrix.get(), mDetFunction.get(), mAlpha, mIteration);
    mLayerPredict->pFirstLayer->mapParameters(mCoeffecient.get(), 0);
}

void ALVaryArrayLearner::predict(const ALVaryArray* array, ALFloatMatrix* dst)
{
    ALASSERT(NULL!=array);
    ALASSERT(NULL!=dst);
    ALASSERT(dst->height() == array->size());
    ALSp<ALFloatMatrix> varyMatrix = new ALVaryArrayMatrix(array, mTime, NULL);
    auto h = varyMatrix->height();
    auto iw = varyMatrix->width();
    auto ow = dst->width();
    ALSp<ALFloatMatrix> expandLine = ALFloatMatrix::create(varyMatrix->width(), 1);
    auto _expand = expandLine->vGetAddr();
    for (size_t i=0; i<h; ++i)
    {
        auto v = varyMatrix->vGetAddr(i);
        ::memcpy(_expand, v, iw*sizeof(ALFLOAT));
        ALSp<ALFloatMatrix> p_line = mLayerPredict->pFirstLayer->forward(expandLine);
        auto _dst = dst->vGetAddr(i);
        ::memcpy(_dst, p_line->vGetAddr(), ow*sizeof(ALFLOAT));
    }
}
