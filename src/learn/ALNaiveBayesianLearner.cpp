#include "learn/ALNaiveBayesianLearner.h"
#include "math/ALStatistics.h"
ALNaiveBayesianLearner::ALNaiveBayesianLearner(bool discrete, bool needNormalize, ALFLOAT divide)
{
    mDiscrete = discrete;//TODO
    mNeedNormalize = needNormalize;//TODO
    mDivide = divide;
}
ALNaiveBayesianLearner::~ALNaiveBayesianLearner()
{
}



class ALBayesianMatrixPredictor:public ALIMatrixPredictor
{
public:
    ALBayesianMatrixPredictor(ALSp<ALFloatMatrix> prop, ALSp<ALFloatMatrix> values, ALFLOAT divide, ALSp<ALFloatMatrix> YP)
    {
        ALASSERT(1 == values->height());
        ALASSERT(prop->height()%2==0);
        ALASSERT(YP->width() == prop->width());
        ALASSERT(prop->width() == values->width());
        mProbabilities = prop;
        mValues = values;
        mDivide = divide;
        mYP = YP;
    }
    virtual ~ALBayesianMatrixPredictor()
    {
    }
    /*Assert X->height() == Y->height()*/
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        ALASSERT(NULL!=X);
        ALASSERT(NULL!=Y);
        ALASSERT(X->height() == Y->height());
        auto w = X->width();
        auto h = X->height();
        auto pw = mProbabilities->width();
        auto values = mValues->vGetAddr();
        ALAUTOSTORAGE(props, ALFLOAT, pw);
        for (int i=0; i<h; ++i)
        {
            auto x = X->vGetAddr(i);
            for (int j=0; j<pw; ++j)
            {
                props[j] = 1.0f;
            }
            for (int j=0; j<w; ++j)
            {
                auto p0 = mProbabilities->vGetAddr(2*j);
                auto p1 = mProbabilities->vGetAddr(2*j+1);
                ALFLOAT* _p = p0;
                if (x[j] > mDivide)
                {
                    _p = p1;
                }
                for (int k=0; k<pw; ++k)
                {
                    props[k]*=_p[k];
                }
            }
            
            int maxPos = -1;
            ALFLOAT maxValue = -1.0f;
            for (int j=0; j<pw; ++j)
            {
                if (props[j]>maxValue)
                {
                    maxPos = j;
                    maxValue = props[j];
                }
            }
            ALASSERT(-1!=maxPos);
            auto y = Y->vGetAddr(i);
            y[0] = values[maxPos];
        }
    }
    
    /*For Classify*/
    virtual void vPredictProbability(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        ALASSERT(NULL!=X);
        ALASSERT(NULL!=Y);
        ALASSERT(X->height() == Y->height());
        ALASSERT(Y->width() == mValues->width());
        ALASSERT(NULL!=X);
        ALASSERT(NULL!=Y);
        ALASSERT(X->height() == Y->height());
        auto w = X->width();
        auto h = X->height();
        auto pw = mProbabilities->width();
        auto yp = mYP->vGetAddr();
        for (int i=0; i<h; ++i)
        {
            auto x = X->vGetAddr(i);
            auto y = Y->vGetAddr(i);
            auto props = y;
            for (int j=0; j<pw; ++j)
            {
                props[j] = yp[j];
            }
            for (int j=0; j<w; ++j)
            {
                auto p0 = mProbabilities->vGetAddr(2*j);
                auto p1 = mProbabilities->vGetAddr(2*j+1);
                ALFLOAT* _p = p0;
                if (x[j] > mDivide)
                {
                    _p = p1;
                }
                for (int k=0; k<pw; ++k)
                {
                    props[k]*=_p[k];
                }
            }
            ALFLOAT sumProp = 0.0f;
            for (int k=0; k<pw; ++k)
            {
                sumProp += props[k];
            }
            for (int k=0; k<pw; ++k)
            {
                props[k]/=sumProp;
            }
        }
    }
    virtual const ALFloatMatrix* vGetPossiableValues() const
    {
        return mValues.get();
    }
    virtual void vPrint(std::ostream& output) const
    {
        ALFloatMatrix::print(mProbabilities.get(), output);
    }
private:
    ALSp<ALFloatMatrix> mProbabilities;
    ALSp<ALFloatMatrix> mValues;
    ALFLOAT mDivide;
    ALSp<ALFloatMatrix> mYP;
};


static ALIMatrixPredictor* _createDiscrete(const ALFloatMatrix* X, const ALFloatMatrix* YT, ALFLOAT divide)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=YT);
    ALASSERT(X->height() == YT->width());
    ALASSERT(1 == YT->height());
    ALSp<ALFloatMatrix> types = ALFloatMatrix::genTypes(YT);
    auto w = X->width();
    auto h = X->height();
    auto t = types->vGetAddr();
    auto t_num = types->width();
    
    int stride = (int)(2*w+1);//First is totalNumber 2*w+1 for < divide, 2*w+2 for > divide
    ALAUTOSTORAGE(typeCount, size_t, stride*t_num);
    ::memset(typeCount, 0, sizeof(size_t)*stride*t_num);
    ALAUTOSTORAGE(xCount, size_t, 2*w);
    ::memset(xCount, 0, sizeof(size_t)*2*w);
    
    auto y = YT->vGetAddr();
    for (int i=0; i<h; ++i)
    {
        int t_i = -1;
        auto yi = y[i];
        for (int j=0; j<t_num; ++j)
        {
            if (ZERO(yi-t[j]))
            {
                t_i = j;
                break;
            }
        }
        ALASSERT(t_i>=0);
        auto typeCountPos = typeCount + stride*t_i;
        typeCountPos[0] = typeCountPos[0]+1;
        auto x = X->vGetAddr(i);
        for (int j=0; j<w; ++j)
        {
            if (x[j]<divide)
            {
                xCount[2*j]++;
                typeCountPos[2*j+1]++;
            }
            else
            {
                xCount[2*j+1]++;
                typeCountPos[2*j+2]++;
            }
        }
    }
    
    ALSp<ALFloatMatrix> props = ALFloatMatrix::create(types->width(), 2*w);
    for (int i=0; i<w; ++i)
    {
        auto p0 = props->vGetAddr(2*i);
        auto p1 = props->vGetAddr(2*i+1);
        for (int j=0; j<t_num; ++j)
        {
            auto numbers = typeCount[stride*j];
            if (numbers > 0)
            {
                p1[j] = (ALFLOAT)typeCount[stride*j+2*i+2] / (ALFLOAT)(ALFLOAT)typeCount[stride*j];
                p0[j] = (ALFLOAT)typeCount[stride*j+2*i+1] / (ALFLOAT)typeCount[stride*j];
            }
            else
            {
                p0[j] = 0;
                p1[j] = 0;
            }
        }
    }
    ALSp<ALFloatMatrix> YP = ALFloatMatrix::create(t_num, 1);
    auto yp = YP->vGetAddr();
    for (int i=0; i<t_num; ++i)
    {
        yp[i] = (ALFLOAT)typeCount[stride*i] / (ALFLOAT)h;
    }
    return new ALBayesianMatrixPredictor(props, types, divide, YP);
}

ALIMatrixPredictor* ALNaiveBayesianLearner::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(X->height() == Y->height());
    ALASSERT(1 == Y->width());
    ALSp<ALFloatMatrix> YT = ALFloatMatrix::transpose(Y);
    if (mDiscrete)
    {
        return _createDiscrete(X, YT.get(), mDivide);
    }
    //TODO
    ALASSERT(false);
    return NULL;
}
