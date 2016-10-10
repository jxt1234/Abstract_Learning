#include "learn/ALCNNLearner.h"
#include "cnn/CNNPredictor.h"
#include "cnn/CNNDerivativeFunction.h"
#include "cnn/MeanPoolLayer.h"
#include "cnn/CNNLayer.h"
#include "cnn/SoftMaxLayer.h"
#include <fstream>
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
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    ALSp<ALFloatMatrix> prop = ALFloatMatrix::genTypes(YT.get());
    
    /*For Debug*/
//    std::ofstream dump1("/Users/jiangxiaotang/Documents/Abstract_Learning/1");
//    std::ofstream dump2("/Users/jiangxiaotang/Documents/Abstract_Learning/2");
//    std::ofstream dump3("/Users/jiangxiaotang/Documents/Abstract_Learning/3");
//    std::ofstream dump4("/Users/jiangxiaotang/Documents/Abstract_Learning/4");
//    std::ofstream dump5("/Users/jiangxiaotang/Documents/Abstract_Learning/5");
    
    auto& inputDescripe = mInputDescribe;
    /*Construct Layer*/
    ALSp<LayerWrap> firstLayer;
    ALSp<LayerWrap> currentLayer;
    ALSp<LayerWrap> lastLayer;
    if (false)
    {
        auto currentWidth = inputDescripe.iWidth;
        auto kernelSize = inputDescripe.iWidth;
        auto currentDepth = inputDescripe.iDepth;
        auto filterNumber = (int)prop->width();
        firstLayer = new LayerWrap(new CNNLayer(currentWidth, currentDepth, kernelSize, filterNumber, 1));
        
        
        lastLayer = new LayerWrap(new SoftMaxLayer((int)prop->width()));
        
        firstLayer->connectOutput(lastLayer);
        lastLayer->connectInput(firstLayer.get());
    }
    else
    {
        //First CNN
        int kernelSize = 5;
        int filterNumber = 6;
        firstLayer = new LayerWrap(new CNNLayer(inputDescripe.iWidth, inputDescripe.iDepth, kernelSize, filterNumber, 1));
        currentLayer = firstLayer;
        auto currentWidth = (inputDescripe.iWidth-kernelSize)+1;
        auto currentDepth = filterNumber;
        //currentLayer->setForwardDebug(&dump1);
        
        //Pool
        ALSp<LayerWrap> nextLayer = new LayerWrap(new MeanPoolLayer(2, currentWidth, currentWidth, currentDepth));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
        currentWidth = currentWidth/2;
        //currentLayer->setForwardDebug(&dump2);

        
        //Second CNN
        kernelSize = 5;
        filterNumber = 12;
        nextLayer = new LayerWrap(new CNNLayer(currentWidth, currentDepth, kernelSize, filterNumber, 1));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
        currentDepth = filterNumber;
        currentWidth = currentWidth - kernelSize + 1;
        //currentLayer->setForwardDebug(&dump3);

        
        //Second Pool
        nextLayer = new LayerWrap(new MeanPoolLayer(2, currentWidth, currentWidth, currentDepth));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
        currentWidth = currentWidth/2;
        //currentLayer->setForwardDebug(&dump4);

        
        /*Last Layer*/
        kernelSize = currentWidth;
        filterNumber = (int)prop->width();
        nextLayer = new LayerWrap(new CNNLayer(currentWidth, currentDepth, kernelSize, filterNumber, 1));
        currentLayer->connectOutput(nextLayer);
        nextLayer->connectInput(currentLayer.get());
        currentLayer = nextLayer;
        //currentLayer->setForwardDebug(&dump5);

        lastLayer = new LayerWrap(new SoftMaxLayer((int)prop->width()));
        
        currentLayer->connectOutput(lastLayer);
        lastLayer->connectInput(currentLayer.get());
    }
    ALSp<ALIGradientDecent::DerivativeFunction> det = new CNNDerivativeFunction(firstLayer, lastLayer, (int)prop->width());
    
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
    auto parameterSize = firstLayer->getParameterSize();
    ALSp<ALFloatMatrix> coefficient = ALFloatMatrix::create(parameterSize, 1);
    /*Init parameters randomly*/
    auto c = coefficient->vGetAddr();
    for (int i=0; i<parameterSize; ++i)
    {
        c[i] = 0.1*ALRandom::rate();
    }
    mGDMethod->vOptimize(coefficient.get(), Merge.get(), det.get(), 0.85, 10000);
    firstLayer->setParameters(coefficient.get(), 0);
    
    return new CNNPredictor(firstLayer, prop);
}
