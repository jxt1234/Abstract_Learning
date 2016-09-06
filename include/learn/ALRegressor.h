#ifndef INCLUDE_LEARN_ALREGRESSOR_H
#define INCLUDE_LEARN_ALREGRESSOR_H
#include "core/ALARStructure.h"
#include "ALISuperviseLearner.h"
/*TODO Support other type*/
class ALRegressor:public ALISuperviseLearner
{
public:
    friend class ALNetRegressor;
    ALRegressor(){}
    virtual ~ALRegressor(){}
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const override;
private:
    ALSp<ALFloatMatrix> getMatrix(ALIMatrixPredictor* p);
    ALIMatrixPredictor* _regressMulti(const ALFloatMatrix* X, const ALFloatMatrix* YT) const;
};
#endif
