#include "CNNPredictor.h"
namespace ALCNN {
    void CNNPredictor::vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        ALASSERT(false);
    }
    void CNNPredictor::vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        mNet->resetBatchSize((int)X->height());
        ALSp<ALFloatMatrix> X_C = ALFloatMatrix::create(X->width(), X->height());
        ALFloatMatrix::copy(X_C.get(), X);
        mNet->forward(X_C);
        ALSp<ALFloatMatrix> Y_P = mNet->getOutput();
        ALFloatMatrix::copy(Y, Y_P.get());
    }
    const ALFloatMatrix* CNNPredictor::vGetPossiableValues() const
    {
        return mProbability.get();
    }
    
    CNNPredictor::CNNPredictor(ALSp<LayerWrap> net, ALSp<ALFloatMatrix> prob)
    {
        mNet = net;
        mProbability = prob;
    }
    CNNPredictor::~ CNNPredictor()
    {
    }

}