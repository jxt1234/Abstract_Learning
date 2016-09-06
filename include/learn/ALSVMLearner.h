#ifndef INCLUDE_LEARN_ALSVMLEARNER_H
#define INCLUDE_LEARN_ALSVMLEARNER_H
#include "ALISuperviseLearner.h"
#include "core/ALIExpander.h"
/*TODO Add More Kernel Type*/
/*Only support -1:1 divide, and only use rbf kernel*/
class ALSVMParameter:public ALRefCount
{
public:
    ALFLOAT Gamma;
    ALFLOAT Bound;
    size_t iternumber;
    ALSVMParameter():Gamma(0.05), Bound(512), iternumber(2){}
    ALSVMParameter(const ALSVMParameter& p) = default;
    ALSVMParameter& operator=(const ALSVMParameter& p) = default;
    ~ALSVMParameter() = default;
};
class ALSVMLearner:public ALISuperviseLearner
{
    public:
        ALSVMLearner(ALSVMParameter* par=NULL);
        virtual ~ALSVMLearner();
        virtual ALIMatrixPredictor* vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const override;
    private:
        ALSVMParameter* mPar;
};
#endif
