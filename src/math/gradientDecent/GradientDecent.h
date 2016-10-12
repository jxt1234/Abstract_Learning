#ifndef MATH_GRADIENTDECENT_GRADIENTDECENT_H
#define MATH_GRADIENTDECENT_GRADIENTDECENT_H
#include "math/ALIGradientDecent.h"

class GradientDecent :public ALIGradientDecent
{
public:
    GradientDecent(int batchSize=50):mBatchSize(batchSize){}
    virtual ~ GradientDecent() {}
    
    virtual void vOptimize(ALFloatMatrix* coefficient, const ALFloatMatrix* X, const DerivativeFunction* delta, double alpha, int iteration) const override;
private:
    int mBatchSize;
};
#endif
