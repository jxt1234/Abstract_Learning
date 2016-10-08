#include "learn/ALCNNLearner.h"
#include "cnn/CNNPredictor.h"
#include "cnn/CNNDerivativeFunction.h"
#include "MeanPoolLayer.h"
#include "CNNLayer.h"
using namespace ALCNN;

ALCNNLearner::ALCNNLearner(const ALIMatrix4DOp::Matrix4D& inputDescripe)
{
    mGDMethod = ALIGradientDecent::create(ALIGradientDecent::SGD);
    ALASSERT(inputDescripe.iHeight == inputDescripe.iWidth);
    mInputDescribe = inputDescripe;
}
ALCNNLearner::~ ALCNNLearner()
{
    
}

ALIMatrixPredictor* ALCNNLearner::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALSp<ALFloatMatrix> prop = ALFloatMatrix::genTypes(Y);
    
    auto& inputDescripe = mInputDescribe;
    /*Construct Layer*/
    ALSp<LayerWrap> net;
    ALSp<LayerWrap> currentLayer;
    {
        //First CNN
        int kernelSize = 5;
        int filterNumber = 6;
        net = new LayerWrap(new CNNLayer(inputDescripe.iWidth, inputDescripe.iDepth, kernelSize, filterNumber, 1));
        currentLayer = net;
        auto currentWidth = (inputDescripe.iWidth-kernelSize)+1;
        auto currentDepth = filterNumber;
        
        //Pool
        ALSp<LayerWrap> nextLayer = new LayerWrap(new MeanPoolLayer(2, currentWidth, currentWidth, currentDepth));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
        currentWidth = currentWidth/2;
        
        //Second CNN
        kernelSize = 5;
        filterNumber = 12;
        nextLayer = new LayerWrap(new CNNLayer(currentWidth, currentDepth, kernelSize, filterNumber, 1));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
        currentDepth = filterNumber;
        currentWidth = currentWidth - kernelSize + 1;
        
        //Second Pool
        nextLayer = new LayerWrap(new MeanPoolLayer(2, currentWidth, currentWidth, currentDepth));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
        currentWidth = currentWidth/2;
        
        /*Last Layer*/
        kernelSize = currentWidth;
        filterNumber = (int)prop->width();
        nextLayer = new LayerWrap(new CNNLayer(currentWidth, currentDepth, kernelSize, filterNumber, 1));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
    }
    ALSp<ALIGradientDecent::DerivativeFunction> det = new CNNDerivativeFunction(net, (int)prop->width());
    
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
    auto parameterSize = net->getParameterSize();
    ALSp<ALFloatMatrix> coefficient = ALFloatMatrix::create(parameterSize, 1);
    auto c = coefficient->vGetAddr();
    for (int i=0; i<parameterSize; ++i)
    {
        c[i] = ALRandom::rate();
    }
    mGDMethod->vOptimize(coefficient.get(), Merge.get(), det.get(), 0.1, 1000);
    net->setParameters(coefficient.get(), 0);
    
    return new CNNPredictor(net, prop);
}
