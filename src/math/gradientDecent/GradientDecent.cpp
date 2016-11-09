#include "GradientDecent.h"

void GradientDecent::vOptimize(ALFloatMatrix* coefficient, const ALFloatMatrix* X, const DerivativeFunction* delta, double alpha, int iteration) const
{
    ALASSERT(NULL!=coefficient);
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=delta);
    ALASSERT(mBatchSize > 0);
    ALASSERT(iteration >= 1);
    iteration = iteration/mBatchSize;
    if (iteration <= 0)
    {
        iteration = 1;
    }
    auto batchNumber = X->height()/mBatchSize;
    for (int i=0; i<iteration; ++i)
    {
        for (size_t j=0; j<=batchNumber; ++j)
        {
            auto x_sta = j*mBatchSize;
            auto x_fin = (j+1)*mBatchSize-1;
            if (x_fin >= X->height())
            {
                x_fin = X->height()-1;
            }
            if (x_fin <= x_sta)
            {
                continue;
            }
            ALSp<ALFloatMatrix> XSelect = ALFloatMatrix::createCropVirtualMatrix(X, 0, x_sta, X->width()-1, x_fin);
            ALSp<ALFloatMatrix> deltaC = delta->vCompute(coefficient, XSelect.get());
            ALFloatMatrix::linear(coefficient, coefficient, 1.0, deltaC.get(), -alpha);
        }
    }
}
