#include "learn/ALLogicalRegress.h"
#include "math/ALIGradientDecent.h"
#include <math.h>
static ALFLOAT computeThetha(const ALFLOAT* x, const ALFLOAT* _w, size_t w)
{
    ALFLOAT thetha = 0.0;
    //_w[0];
    for (int j=0; j<w; ++j)
    {
        thetha += _w[j]*x[j];
    }
    thetha = 1.0/(1.0 + exp(-thetha));
    return thetha;
}

class LogGradientComputer : public ALIGradientDecent::DerivativeFunction
{
public:
    LogGradientComputer(size_t size):mSize(size){}
    virtual ~ LogGradientComputer(){}
    
    virtual size_t vInitParameters(ALFloatMatrix* coefficient) const override
    {
        if (NULL!=coefficient)
        {
            ALASSERT(coefficient->width() == mSize);
            ALFloatMatrix::zero(coefficient);
        }
        return mSize;
    }

    /*X is merged as [Y, X]*/
    virtual ALFloatMatrix* vCompute(ALFloatMatrix* coefficient, const ALFloatMatrix* X) const override
    {
        ALASSERT(NULL!=coefficient);
        ALASSERT(NULL!=X);
        ALASSERT(coefficient->height() == 1);
        ALASSERT(coefficient->width() == mSize);
        ALASSERT(X->width() == coefficient->width()+1);
        auto weight = coefficient->vGetAddr();
        auto h = X->height();
        auto w = coefficient->width();
        ALAUTOSTORAGE(error, ALFLOAT, (int)h);
        for (int i=0; i<h; ++i)
        {
            auto y = X->vGetAddr(i);
            auto x = y+1;
            error[i] = computeThetha(x, weight, w) - *y;
        }
        ALFloatMatrix* detC = ALFloatMatrix::create(coefficient->width(), 1);
        ALFloatMatrix::zero(detC);
        auto det = detC->vGetAddr();

        for (size_t i=0; i<h; ++i)
        {
            ALFLOAT* x = X->vGetAddr(i)+1;
            for (size_t j=0; j<w; ++j)
            {
                det[j] += (error[i])*x[j];
            }
        }
        
        return detC;
    }
private:
    size_t mSize;
};

ALLogicalRegress::ALLogicalRegress(int iter, ALFLOAT alpha)
{
    mMaxIter = iter;
    mAlpha = alpha;
    if (mAlpha < 0)
    {
        mAlpha = -mAlpha;
    }
}
ALLogicalRegress::~ALLogicalRegress()
{
}
ALIMatrixPredictor* ALLogicalRegress::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALSp<ALFloatMatrix> w = learn(X, Y, mMaxIter, mAlpha);
    class LogicalJudger:public ALIMatrixPredictor
    {
    public:
        LogicalJudger(ALSp<ALFloatMatrix> W):mW(W)
        {
        }
        virtual ~LogicalJudger(){}
        virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
        {
            ALASSERT(X->height() == Y->height());
            ALASSERT(Y->width()>=1);
            for (auto i =0; i<X->height(); ++i)
            {
                auto x = X->vGetAddr(i);
                auto y = Y->vGetAddr(i);
                y[0] = computeThetha(x, mW->vGetAddr(), X->width());
            }
        }
        virtual void vPrint(std::ostream& output) const override
        {
            output << "<LogicModel>\n";
            ALFloatMatrix::print(mW.get(), output);
            output << "</LogicModel>\n";
        }
    private:
        ALSp<ALFloatMatrix> mW;
    };
    return new LogicalJudger(w);
}

ALSp<ALFloatMatrix> ALLogicalRegress::learn(const ALFloatMatrix* X, const ALFloatMatrix* Y, size_t maxiter, ALFLOAT alpha)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(maxiter>=1);
    ALASSERT(X->height() == Y->height());
    auto w = X->width();
    ALSp<ALFloatMatrix> W = ALFloatMatrix::create(w, 1);
    ALFloatMatrix::zero(W.get());
    ALSp<ALIGradientDecent> gd = ALIGradientDecent::create(ALIGradientDecent::SGD);
    LogGradientComputer delta(w);
    ALSp<ALFloatMatrix> merge = ALFloatMatrix::unionHorizontal(Y,X);
    gd->vOptimize(W.get(), merge.get(), &delta, alpha, (int)maxiter);
    return W;
}
ALSp<ALFloatMatrix> ALLogicalRegress::predict(const ALFloatMatrix* X, const ALFloatMatrix* W)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=W);
    ALASSERT(X->width() == W->width()-1);
    auto w = X->width();
    auto h = X->height();
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::create(1, h);
    ALFLOAT* _w = W->vGetAddr();
    for (int i=0; i<h; ++i)
    {
        ALFLOAT* x = X->vGetAddr(i);
        *(Y->vGetAddr(i)) = computeThetha(x, _w, w);
    }
    return Y;
}
