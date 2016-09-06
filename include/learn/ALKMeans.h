#ifndef INCLUDE_LEARN_ALKMEANS_H
#define INCLUDE_LEARN_ALKMEANS_H
#include "ALIUnSuperLearner.h"
#include "core/ALIExpander.h"
#include "math/ALFloatMatrix.h"
class ALKMeans:public ALIUnSuperLearner
{
public:
    ALKMeans(size_t class_number, size_t iter=1000);
    virtual ~ALKMeans();
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X) const override;
    static ALSp<ALFloatMatrix> learn(const ALFloatMatrix* X, size_t number, size_t maxiter=1000);
    static void predict(const ALFloatMatrix* X, const ALFloatMatrix* Center, ALFloatMatrix* R);
private:
    size_t mNumber;
    size_t mIter;
};
#endif
