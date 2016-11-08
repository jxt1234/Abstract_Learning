#include "learn/ALRNNLearner.h"
#include "cnn/CNNDerivativeFunction.h"
#include <string.h>
#include <fstream>
#include "cnn/LayerWrapFactory.h"
using namespace ALCNN;
struct ALRNNLearner::LayerStruct
{
    ALSp<LayerWrap> pFirstLayer;
};

class ALRNNPredictor : public ALFloatPredictor
{
    public:
        ALRNNPredictor(ALSp<LayerWrap> layer, ALSp<ALIExpander> expander, ALSp<ALFloatMatrix> p)
        {
            mExpander = expander;
            mLength = expander->vLength();
            mLayerPredict = layer;
            mParameters = p;
        }
        //Not thread-safe
        virtual ALFLOAT vPredict(const ALFloatData* data) const override
        {
            ALSp<ALFloatMatrix> temp = ALFloatMatrix::create(mLength, 1);
            auto res = mExpander->vExpand(data, temp->vGetAddr());
            if (!res)
            {
                return 0.0f;
            }
            ALSp<ALFloatMatrix> pred = mLayerPredict->forward(temp);
            ALASSERT(1==pred->width());
            return *(pred->vGetAddr());
        }
        virtual void vPrint(std::ostream& out) const override
        {
            //TODO
        }
    private:
        mutable ALSp<LayerWrap> mLayerPredict;
        ALSp<ALIExpander> mExpander;
        ALSp<ALFloatMatrix> mParameters;
        size_t mLength;
};

ALRNNLearner::ALRNNLearner(const cJSON* description)
{
    ALASSERT(NULL!=description);
    cJSON* layer = NULL;
    mIteration = 1000;
    mAR.c = 0;
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
            mAR.l = c->valueint;
        }
        else if (strcmp(c->string, "width")==0)
        {
            mAR.w = c->valueint;
        }
    }
    ALASSERT(NULL!=layer);
    ALSp<LayerWrap> firstLayer = LayerWrapFactory::create(layer);
    auto ow = 1;
    mGDMethod = ALIGradientDecent::create(ALIGradientDecent::SGD, mBatchSize);
    mDetFunction = new CNNDerivativeFunction(firstLayer, ow);
    mExpander = new ALARExpander(mAR);
    mLayerPredict = new LayerStruct;
    mLayerPredict->pFirstLayer = firstLayer;
}
ALRNNLearner::~ALRNNLearner()
{
    delete mLayerPredict;
}
ALFloatPredictor* ALRNNLearner::vLearn(const ALLabeldData* data) const
{
    /*Prepare Data*/
    ALSp<ALFloatMatrix> X;
    ALSp<ALFloatMatrix> YT;
    ALIExpander::expandXY(mExpander.get(), data, X, YT);
    ALASSERT(NULL!=X.get() && NULL!=YT.get());//TODO
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::transpose(YT.get());
    ALSp<ALFloatMatrix> Merge = ALFloatMatrix::unionHorizontal(Y.get(), X.get());
    X = NULL;
    Y = NULL;
    /*Optimize parameters*/
    auto parameterSize = mLayerPredict->pFirstLayer->getParameterSize();
    ALSp<ALFloatMatrix> coefficient = ALFloatMatrix::create(parameterSize, 1);
    /*Init parameters randomly*/
    mDetFunction->vInitParameters(coefficient.get());
    mGDMethod->vOptimize(coefficient.get(), Merge.get(), mDetFunction.get(), 0.35, mIteration);
    mLayerPredict->pFirstLayer->mapParameters(coefficient.get(), 0);
    return new ALRNNPredictor(mLayerPredict->pFirstLayer, mExpander, coefficient);
}
