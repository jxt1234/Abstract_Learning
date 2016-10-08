#ifndef INCLUDE_LEARN_ALCNNLEARNER_H
#define INCLUDE_LEARN_ALCNNLEARNER_H
#include "ALLearnFactory.h"
#include "math/ALIGradientDecent.h"
#include "math/ALIMatrix4DOp.h"
class ALCNNLearner : public ALISuperviseLearner
{
public:
    ALCNNLearner(const ALIMatrix4DOp::Matrix4D& inputDescribe);
    virtual ~ ALCNNLearner();
    
    virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const;

private:
    ALSp<ALIGradientDecent> mGDMethod;
    ALIMatrix4DOp::Matrix4D mInputDescribe;
};

#endif
