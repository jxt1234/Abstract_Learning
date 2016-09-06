#ifndef INCLUDE_LEARN_ALNAIVEBAYESIANLEARNER_H
#define INCLUDE_LEARN_ALNAIVEBAYESIANLEARNER_H
#include "ALISuperviseLearner.h"

class ALNaiveBayesianLearner:public ALISuperviseLearner
{
public:
    ALNaiveBayesianLearner(bool discrete=true, bool needNormalize=false, ALFLOAT divide = 0.5);
    virtual ~ALNaiveBayesianLearner();
    /*Assert X->height() == Y->height()*/
    /*X is assume as 0-1 matrix*/
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const override;
private:
    bool mDiscrete;
    bool mNeedNormalize;
    ALFLOAT mDivide;
};
#endif
