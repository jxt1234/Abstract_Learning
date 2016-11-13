#include "StochasticGradientDecent.h"
void StochasticGradientDecent::vOptimize(ALFloatMatrix* coefficient, const ALFloatMatrix* X, const DerivativeFunction* delta, double alpha, int iteration) const
{
    ALASSERT(alpha>0);
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
        //ALFORCEAUTOTIME;
        
        //ALSp<ALFloatMatrix> selectX = ALFloatMatrix::randomSelectMatrix(X, mBatchSize, true);
        
        /*For Debug*/
        ALSp<ALFloatMatrix> selectX = ALFloatMatrix::create(X->width(), mBatchSize);
        ALSp<ALFloatMatrix> selectXCrop = ALFloatMatrix::createCropVirtualMatrix(X, 0, 0, X->width()-1, mBatchSize-1);
        ALFloatMatrix::copy(selectX.get(), selectXCrop.get());
        
        ALSp<ALFloatMatrix> deltaC = delta->vCompute(coefficient, selectX.get());
        ALFloatMatrix::linear(coefficient, coefficient, 1.0, deltaC.get(), -alpha);
        ALFloatMatrix::checkAndSet(coefficient, 0.0f);
    }
}
