#ifndef INCLUDE_LEARN_ALGMM_H
#define INCLUDE_LEARN_ALGMM_H
#include "ALIUnSuperLearner.h"
class ALGMM:public ALIUnSuperLearner
{
public:
    ALGMM(int centernumber);
    virtual ~ALGMM();
    
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X) const override;
private:
    int mCenters;
};
#endif
