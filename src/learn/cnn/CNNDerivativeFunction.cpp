#include "CNNDerivativeFunction.h"
namespace ALCNN {
    CNNDerivativeFunction::CNNDerivativeFunction(ALSp<LayerWrap> net, int outputSize)
    {
        mNet = net;
        mOutputSize = outputSize;
    }
    CNNDerivativeFunction::~CNNDerivativeFunction()
    {
    }

    ALFloatMatrix* CNNDerivativeFunction::vCompute(ALFloatMatrix* coefficient, const ALFloatMatrix* Merge) const
    {
        mNet->resetBatchSize((int)Merge->height());
        ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(Merge, mOutputSize, 0, Merge->width()-1, Merge->height()-1);
        ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(Merge, 0, 0, mOutputSize-1, Merge->height()-1);
        mNet->setParameters(coefficient, 0);

        mNet->forward(X);
        auto YP = mNet->getOutput();
        ALASSERT(YP->width() == Y->width());
        ALASSERT(YP->height() == Y->height());

        ALSp<ALFloatMatrix> YDiff = ALFloatMatrix::create(YP->width(), YP->height());
        auto yh = YP->height();
        auto yw = YP->width();
        for (int i=0; i<yh; ++i)
        {
            auto dst = YDiff->vGetAddr(i);
            auto srcP = YP->vGetAddr(i);
            auto srcO = Y->vGetAddr(i);
            for (int j=0; j<yw; ++j)
            {
                dst[j] = srcP[j]*(1.0-srcP[j])*(srcO[j]-srcP[j]);
            }
        }
        mNet->backward(YDiff);
        ALFloatMatrix* resultDiff = ALFloatMatrix::create(coefficient->width(), coefficient->height());
        mNet->readParametersDiff(resultDiff, 0);
        return resultDiff;
    }
}
