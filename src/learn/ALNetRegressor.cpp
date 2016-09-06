#include "learn/ALNetRegressor.h"
#include "learn/ALRegressor.h"
#include "utils/ALDebug.h"
#include "math/ALStatistics.h"
#include <vector>

static size_t power_int(size_t x, size_t y)
{
    size_t powers = 1;
    for (int i=0; i<y; ++i)
    {
        powers *= x;
    }
    return powers;
}

class ALNetRegressorPredictor:public ALIMatrixPredictor
{
public:
    ALNetRegressorPredictor(ALSp<ALFloatMatrix> xdividers, bool offset=false)
    {
        ALASSERT(NULL!=xdividers.get());
        auto powers = power_int(xdividers->width()+1, xdividers->height());
        for (int i=0; i<powers; ++i)
        {
            mParameters.push_back(NULL);
        }
        mXDividers = xdividers;
        mOffset = offset;
    }
    virtual ~ALNetRegressorPredictor(){}
    
    void setParameters(ALSp<ALFloatMatrix> parameters, size_t magic)
    {
        ALASSERT(magic < mParameters.size() && NULL!=parameters.get());
        ALASSERT(1 == parameters->height());
        mParameters[magic] = parameters;
    }
    
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        ALASSERT(NULL!=X);
        ALASSERT(NULL!=Y);
        ALASSERT(X->height() == Y->height());
        ALASSERT(X->width() == mXDividers->height());
        auto h = X->height();
        auto w = X->width();
        for (int i=0; i<h; ++i)
        {
            auto x = X->vGetAddr(i);
            auto y = Y->vGetAddr(i);
            auto magic = computeMagic(x);
            if (NULL == mParameters[magic].get())
            {
                y[0] = 0.0;
                continue;
            }
            auto p = mParameters[magic]->vGetAddr();
            ALFLOAT sum = 0.0;
            for (int j=0; j<w; ++j)
            {
                sum += p[j]*x[j];
            }
            if (mOffset)
            {
                sum += p[w];
            }
            y[0] = sum;
        }
    }
    virtual void vPrint(std::ostream& output) const
    {
        output << "<ALNetRegressorPredictor>\n";
        output << "<Offset>"<<mOffset << "</Offset>\n";
        output << "<Number>"<<mXDividers->width()+1 <<"</Number>\n";
        output << "<Dividers>\n";
        ALFloatMatrix::print(mXDividers.get(), output);
        output << "</Dividers>\n";
        output << "<Parameters>\n";
        for (int i=0; i<mParameters.size(); ++i)
        {
            if (NULL!=mParameters[i].get())
            {
                ALFloatMatrix::print(mParameters[i].get(), output);
            }
        }
        output << "</Parameters>\n";
        output << "</ALNetRegressorPredictor>\n";
    }
    size_t computeMagic(ALFLOAT* x) const
    {
        auto h = mXDividers->height();
        auto w = mXDividers->width();
        size_t magic = 0;
        for (int i=0; i<h; ++i)
        {
            magic*=(w+1);
            auto xi = x[i];
            auto xd = mXDividers->vGetAddr(i);
            int j=0;
            for (j=0; j<w; ++j)
            {
               if (xi < xd[j])
               {
                   break;
               }
            }
            magic+=j;
        }
        return magic;
    }
private:
    ALSp<ALFloatMatrix> mXDividers;
    std::vector<ALSp<ALFloatMatrix>> mParameters;
    bool mOffset;
};

ALNetRegressor::ALNetRegressor(int number, bool useOffset)
{
    ALASSERT(number >= 2);
    mNumber = number;
    mOffset = useOffset;
}
ALNetRegressor::~ALNetRegressor()
{
    
}

ALIMatrixPredictor* ALNetRegressor::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(X->height() == Y->height());
    ALSp<ALFloatMatrix> stats = ALStatistics::statistics(X);
    /*Make dividers*/
    int divide_number = mNumber-1;
    ALSp<ALFloatMatrix> dividers = ALFloatMatrix::create(X->width(), divide_number);
    auto w = X->width();
    auto _max = stats->vGetAddr(2);
    auto _min = stats->vGetAddr(1);
    float interval = 1.0/(mNumber);
    for (int i=0; i<divide_number; ++i)
    {
        auto _d = dividers->vGetAddr(i);
        for (int j=0; j<w; ++j)
        {
            _d[j] = interval*(i+1)*(_max[j]-_min[j])+_min[j];
        }
    }
    ALNetRegressorPredictor* totalPredictor = new ALNetRegressorPredictor(ALFloatMatrix::transpose(dividers.get()), mOffset);
    /*Create regressor begin*/
    auto n = power_int(mNumber, X->width());
    std::vector<ALSp<ALFloatMatrix>> XSamples;
    std::vector<ALSp<ALFloatMatrix>> YSamples;
    std::vector<size_t> numbers;
    for (int i=0; i<n; ++i)
    {
        XSamples.push_back(NULL);
        YSamples.push_back(NULL);
        numbers.push_back(0);
    }
    //Select samples
    auto h = X->height();
    /*Count numbers*/
    for (int i=0; i<h; ++i)
    {
        auto x = X->vGetAddr(i);
        auto magic = totalPredictor->computeMagic(x);
        numbers[magic]++;
    }
    for (int i=0; i<n; ++i)
    {
        if (0 < numbers[i])
        {
            if (mOffset)
            {
                XSamples[i] = ALFloatMatrix::create(X->width()+1, numbers[i]);
                for (int j=0; j<numbers[i]; ++j)
                {
                    XSamples[i]->vGetAddr(j)[X->width()] = 1;
                }
            }
            else
            {
                XSamples[i] = ALFloatMatrix::create(X->width(), numbers[i]);
            }
            YSamples[i] = ALFloatMatrix::create(Y->width(), numbers[i]);
        }
    }
    /*Fill*/
    std::vector<size_t> curs(n, 0);
    for (int i=0; i<h; ++i)
    {
        auto x = X->vGetAddr(i);
        auto y = Y->vGetAddr(i);
        auto magic = totalPredictor->computeMagic(x);
        size_t cur = curs[magic];
        curs[magic]++;
        auto dst_x = XSamples[magic]->vGetAddr(cur);
        auto dst_y = YSamples[magic]->vGetAddr(cur);
        ::memcpy(dst_x, x, X->width()*sizeof(ALFLOAT));
        ::memcpy(dst_y, y, Y->width()*sizeof(ALFLOAT));
    }
    ALRegressor reg;
    for (int i=0; i<n; ++i)
    {
        if (numbers[i] > 0)
        {
            ALSp<ALIMatrixPredictor> m = reg.vLearn(XSamples[i].get(), YSamples[i].get());
            totalPredictor->setParameters(ALFloatMatrix::transpose(reg.getMatrix(m.get()).get()), i);
        }
    }
    /*Create regressor end*/
    return totalPredictor;
}
