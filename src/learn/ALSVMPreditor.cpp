#include "learn/ALSVMPreditor.h"
#include <string.h>

ALSVMPreditor::ALSVMPreditor(ALSp<ALSVM> svm)
{
    mSVM = svm;
}

ALSVMPreditor::~ALSVMPreditor()
{
}

void ALSVMPreditor::vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
{
    mSVM->predict(Y, X);
}
