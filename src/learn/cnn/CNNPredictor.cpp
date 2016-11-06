#include "CNNPredictor.h"
#include <fstream>
namespace ALCNN {
    void CNNPredictor::vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        ALASSERT(X->height() == Y->height());
        ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(mProbability->width(), Y->height());
        this->vPredictProbability(X, YP.get());
        ALSp<ALFloatMatrix> YT = ALFloatMatrix::getTypes(YP.get(), mProbability.get());
        ALFloatMatrix::transpose(YT.get(), Y);
    }
    void CNNPredictor::vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        ALASSERT(X->height() == Y->height());
        ALSp<ALFloatMatrix> X_C = ALFloatMatrix::create(X->width(), X->height());
        ALFloatMatrix::copy(X_C.get(), X);
        ALSp<ALFloatMatrix> Y_P = mNet->forward(X_C);
        ALFloatMatrix::copy(Y, Y_P.get());
    }
    const ALFloatMatrix* CNNPredictor::vGetPossiableValues() const
    {
        return mProbability.get();
    }
    
    CNNPredictor::CNNPredictor(ALSp<LayerWrap> net, ALSp<ALFloatMatrix> prob, ALSp<ALFloatMatrix> parameters)
    {
        mNet = net;
        mProbability = prob;
        mParameters = parameters;
    }
    CNNPredictor::~ CNNPredictor()
    {
    }

}
