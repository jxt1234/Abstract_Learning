#include <math.h>
#include "learn/ALSpectralClustering.h"
#include "math/ALStatistics.h"
#include "learn/ALKMeans.h"
ALSpectralClustering::ALSpectralClustering(ALSp<ALIExpander> X, ALFLOAT sigma, size_t k, size_t class_number)
{
    mX = X;
    mSigma = sigma;
    mK = k;
    mNum = class_number;
}
ALSpectralClustering::~ALSpectralClustering()
{
}

ALIMatrixPredictor*  ALSpectralClustering::vLearn(const ALFloatMatrix* X) const
{
    /*TODO*/
    return NULL;
}

ALFloatMatrix* ALSpectralClustering::relativeGauss(ALFloatMatrix* X) const
{
    ALAUTOTIME;
    /*TODO normalize X firstly*/
    ALASSERT(NULL!=X);
    auto w = X->width();
    auto h = X->height();
    ALFloatMatrix* R = ALFloatMatrix::create(h, h);
    for (size_t i=0; i<h; ++i)
    {
        R->vGetAddr(i)[i] = 1.0;
        auto xi = X->vGetAddr(i);
        for (size_t j=i+1; j<h; ++j)
        {
            auto xj = X->vGetAddr(j);
            ALFLOAT sum = 0.0;
            for (size_t k=0; k<w; ++k)
            {
                auto diff = xi[k]-xj[k];
                sum += (diff*diff);
            }
            sum = exp(-sum/2.0/mSigma);
            R->vGetAddr(j)[i] = sum;
            R->vGetAddr(i)[j] = sum;
        }
    }
    return R;
}
ALFloatMatrix* ALSpectralClustering::classify(ALFloatMatrix* X) const
{
    ALASSERT(NULL!=X);
    ALSp<ALFloatMatrix> S = relativeGauss(X);
    ALSp<ALFloatMatrix> Root;
    ALSp<ALFloatMatrix> Vector;
    ALStatistics::characteristic_compute(S.get(), Root, Vector);
    auto r = Root->vGetAddr();
    auto k = mK;
    if (k > Root->width())
    {
       k = Root->width();
    }
    ALAutoStorage<ALFLOAT> __kr(k);
    ALAutoStorage<size_t> __kn(k);
    auto kr = __kr.get();
    auto n = __kn.get();
    for (size_t i=0; i<k; ++i)
    {
        kr[i] = r[i];
        n[i] = i;
    }
    for (size_t i=k; i<Root->width(); ++i)
    {
        size_t max_n = 0;
        ALFLOAT max_r = kr[0];
        for (size_t j=1; j<k; ++j)
        {
            if (kr[j] > max_r)
            {
                max_r = kr[j];
                max_n = j;
            }
        }
        if (r[i] < max_r)
        {
            n[max_n] = i;
            kr[max_n] = r[i];
        }
    }
    Vector = ALFloatMatrix::transpose(Vector.get());
    ALSp<ALFloatMatrix> VNew = ALFloatMatrix::create(Vector->width(), k);
    for (size_t i=0; i<k; ++i)
    {
        auto v = Vector->vGetAddr(n[i]);
        auto vnew = VNew->vGetAddr(i);
        ::memcpy(vnew, v, sizeof(ALFLOAT)*VNew->width());
    }
    VNew = ALFloatMatrix::transpose(VNew.get());
    ALSp<ALFloatMatrix> Center = ALKMeans::learn(VNew.get(), mNum);
    ALFloatMatrix* result = ALFloatMatrix::create(1, VNew->height());
    ALKMeans::predict(VNew.get(), Center.get(), result);
    return result;
}
