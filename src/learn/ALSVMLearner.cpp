#include "learn/ALSVMLearner.h"
#include "learn/ALSVMPreditor.h"
#include "learn/ALSVMKernelFactory.h"
#include "core/ALNormalizer.h"
#include "core/ALBasicExpander.h"
using namespace std;

ALSVMLearner::ALSVMLearner(ALSVMParameter* par)
{
    if (NULL == par)
    {
        mPar = new ALSVMParameter;
    }
    else
    {
        par->addRef();
        mPar = par;
    }
}
ALSVMLearner::~ALSVMLearner()
{
    mPar->decRef();
}

ALIMatrixPredictor* ALSVMLearner::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(X->height() == Y->height());
    ALSp<ALSVM::Kernel> k = ALSVMKernelFactory::createRBF(mPar->Gamma);
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    ALSp<ALSVM> svm = ALSVM::train(YT.get(), X, k, mPar->Bound, mPar->iternumber);
    if (NULL == svm.get())
    {
        return new ALDummyMatrixPredictor;
    }
    ALIMatrixPredictor* result = new ALSVMPreditor(svm);
    return result;
}
