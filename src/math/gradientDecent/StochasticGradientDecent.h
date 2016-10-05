#ifndef MATH_GRADIENTDECENT_STOCHASTICGRADIENTDECENT_H
#define MATH_GRADIENTDECENT_STOCHASTICGRADIENTDECENT_H
#include "math/ALIGradientDecent.h"
#include "GradientDecent.h"
class StochasticGradientDecent :public ALIGradientDecent
{
public:
    StochasticGradientDecent(int batchSize = 5) {mBatchSize = batchSize;}
    virtual ~ StochasticGradientDecent() {}
    
    virtual void vOptimize(ALFloatMatrix* coefficient, const ALFloatMatrix* X, const DerivativeFunction* delta, double alpha, int iteration) const override;
private:
    int mBatchSize;
    GradientDecent mDegeneration;
};
#endif
