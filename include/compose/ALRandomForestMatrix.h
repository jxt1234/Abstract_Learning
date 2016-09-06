#ifndef INCLUDE_LEARN_ALRANDOMFORESTMATRIX_H
#define INCLUDE_LEARN_ALRANDOMFORESTMATRIX_H
#include "learn/ALISuperviseLearner.h"
class ALRandomForestMatrix:public ALISuperviseLearner
{
public:
    ALRandomForestMatrix(int mTree=50, bool isdiscrete=false);
    virtual ~ALRandomForestMatrix();
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const override;
private:
    int mTree;
    bool mDiscrete;
};
#endif
