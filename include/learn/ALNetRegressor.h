#ifndef INCLUDE_LEARN_ALNETREGRESSOR_H
#define INCLUDE_LEARN_ALNETREGRESSOR_H
#include "ALISuperviseLearner.h"
class ALNetRegressor:public ALISuperviseLearner
{
public:
    ALNetRegressor(int number=2, bool useoffset=true);
    virtual ~ALNetRegressor();
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const;
private:
    int mNumber;
    bool mOffset;
};
#endif
