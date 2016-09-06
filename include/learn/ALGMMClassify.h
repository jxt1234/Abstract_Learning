#ifndef INCLUDE_LEARN_ALGMMCLASSIFY_H
#define INCLUDE_LEARN_ALGMMCLASSIFY_H
#include "ALISuperviseLearner.h"
class ALGMMClassify:public ALISuperviseLearner
{
public:
    ALGMMClassify(int centers = 3);
    virtual ~ALGMMClassify();
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const;
private:
    int mCenters;
};
#endif
