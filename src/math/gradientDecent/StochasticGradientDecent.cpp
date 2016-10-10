#include "StochasticGradientDecent.h"
void StochasticGradientDecent::vOptimize(ALFloatMatrix* coefficient, const ALFloatMatrix* X, const DerivativeFunction* delta, double alpha, int iteration) const
{
    ALASSERT(NULL!=coefficient);
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=delta);
    ALASSERT(iteration >= 1);
    if (mBatchSize >= X->height())
    {
        mDegeneration.vOptimize(coefficient, X, delta, alpha, iteration);
        return;
    }
    for (int i=0; i<iteration; ++i)
    {
        ALSp<ALFloatMatrix> selectX = ALFloatMatrix::randomeSelectMatrix(X, mBatchSize);
        ALSp<ALFloatMatrix> deltaC = delta->vCompute(coefficient, selectX.get());
        ALSp<ALFloatMatrix> C = ALFloatMatrix::linear(coefficient, 1.0, deltaC.get(), -alpha);
        ALFloatMatrix::copy(coefficient, C.get());
    }
}
