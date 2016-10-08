#ifndef INCLUDE_LEARN_ALLOGICALREGRESS_H
#define INCLUDE_LEARN_ALLOGICALREGRESS_H
#include "ALISuperviseLearner.h"
#include "core/ALIExpander.h"
#include "math/ALFloatMatrix.h"
class ALLogicalRegress:public ALISuperviseLearner
{
public:
    ALLogicalRegress(int iter=1000, ALFLOAT alpha = 0.1);
    virtual ~ALLogicalRegress();
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const override;
    static ALSp<ALFloatMatrix> learn(const ALFloatMatrix* X, const ALFloatMatrix* Y, size_t maxiter, ALFLOAT alpha);
    static ALSp<ALFloatMatrix> predict(const ALFloatMatrix* X, const ALFloatMatrix* W);
private:
    int mMaxIter;
    ALFLOAT mAlpha;
};
#endif
