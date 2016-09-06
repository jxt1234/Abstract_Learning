#ifndef INCLUDE_LEARN_ALSVMPREDITOR_H
#define INCLUDE_LEARN_ALSVMPREDITOR_H
#include "core/ALFloatData.h"
#include "learn/ALISuperviseLearner.h"
#include "ALSVM.h"
class ALSVMPreditor:public ALIMatrixPredictor
{
public:
    ALSVMPreditor(ALSp<ALSVM> svm);
    virtual ~ALSVMPreditor();
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override;
    virtual void vPrint(std::ostream& output) const override
    {
        mSVM->save(output);
    }
private:
    ALSp<ALSVM> mSVM;
};
#endif
