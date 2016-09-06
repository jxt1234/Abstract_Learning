#include "learn/ALRegressor.h"
#include "math/ALFloatMatrix.h"
#include "core/ALBasicExpander.h"
#include <iostream>

using namespace std;

class ALProductMatrixPredictor:public ALIMatrixPredictor
{
public:
    friend class ALRegressor;
    ALProductMatrixPredictor(ALSp<ALFloatMatrix> p):mP(p){}
    virtual ~ALProductMatrixPredictor(){}
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
    {
        ALASSERT(X->width()>=mP->width());
        auto k = mP->width();
        auto h = X->height();
        auto w = mP->width();
        ALFloatMatrix::productBasicT(Y->vGetAddr(), Y->width(), X->vGetAddr(), X->width(), mP->vGetAddr(), mP->width(), w, h, k);
    }
    virtual void vPrint(std::ostream& output) const override
    {
        output << "<ALProductMatrixPredictor>"<<endl;
        ALFloatMatrix::print(mP.get(), output);
        output << "</ALProductMatrixPredictor>"<<endl;
    }
private:
    ALSp<ALFloatMatrix> mP;
};
ALSp<ALFloatMatrix> ALRegressor::getMatrix(ALIMatrixPredictor* p)
{
    ALProductMatrixPredictor* _p = (ALProductMatrixPredictor*)p;
    return _p->mP;
}

ALIMatrixPredictor* ALRegressor::_regressMulti(const ALFloatMatrix* X, const ALFloatMatrix* YT) const
{
    ALASSERT(NULL!=X && NULL!=YT);
    ALASSERT(X->height() == YT->width());
    ALASSERT(X->height()>0);

    //Compute
    ALSp<ALFloatMatrix> XT = ALFloatMatrix::transpose(X);
    ALSp<ALFloatMatrix> XTX= ALFloatMatrix::sts(XT.get(), true);
    /*Result = (XTX)-1 * XT * Y, P, Q for mid compute*/
    ALSp<ALFloatMatrix> P = ALFloatMatrix::inverse(XTX.get());
    ALSp<ALFloatMatrix> Q = ALFloatMatrix::product(P.get(), XT.get());
    P = ALFloatMatrix::productT(Q.get(), YT);
    return new ALProductMatrixPredictor(P);
}

ALIMatrixPredictor* ALRegressor::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::create(Y->height(), Y->width());
    ALFloatMatrix::transposeBasic(Y->vGetAddr(), Y->width(), YT->vGetAddr(), YT->width(), Y->width(), Y->height());
    return _regressMulti(X, YT.get());
}

