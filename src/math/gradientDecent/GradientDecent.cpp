#include "GradientDecent.h"

void GradientDecent::vOptimize(ALFloatMatrix* coefficient, const ALFloatMatrix* X, const DerivativeFunction* delta, double alpha, int iteration) const
{
    ALASSERT(NULL!=coefficient);
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=delta);
    ALASSERT(iteration >= 1);
    for (int i=0; i<iteration; ++i)
    {
        ALSp<ALFloatMatrix> deltaC = delta->vCompute(coefficient, X);
        ALSp<ALFloatMatrix> C = ALFloatMatrix::linear(coefficient, 1.0, deltaC.get(), -alpha);
        ALFloatMatrix::copy(coefficient, C.get());
    }
}
