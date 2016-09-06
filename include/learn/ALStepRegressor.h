#ifndef INCLUDE_LEARN_ALSTEPREGRESSOR_H
#define INCLUDE_LEARN_ALSTEPREGRESSOR_H
#include "core/ALIExpander.h"
#include "core/ALFloatData.h"
#include "learn/ALIChainLearner.h"


class ALStepRegressor:public ALIChainLearner
{
public:
    ALStepRegressor(int b) {mMaxBack = b;}
    virtual ~ALStepRegressor() {}
    
    ALIExpander* train(const ALLabeldData* data) const;
    virtual ALFloatPredictor* vLearn(const ALLabeldData* data) const override;
private:
    int mMaxBack;
};

#endif
