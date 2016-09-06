#include "learn/ALGMM.h"
#include "learn/ALKMeans.h"
#include <vector>
#include <math.h>
#include <fstream>
//static const ALFLOAT PI = 3.141592654;

class ALGMMModel:public ALIMatrixPredictor
{
public:
    ALGMMModel(ALSp<ALFloatMatrix> center, const std::vector<ALSp<ALFloatMatrix>>& covs, const std::vector<ALFLOAT>& coefs)
    {
        ALASSERT(center->height() == covs.size());
        ALASSERT(covs.size() == coefs.size());
        //ALFLOAT basic_parameter = 1.0/pow(2.0*PI, center->width()/2.0);
        for (auto c : covs)
        {
            ALASSERT(c->width() == c->height() && c->width() == center->width());
        }
        mCenters = center;
        for (int i=0; i<covs.size(); ++i)
        {
            auto c = covs[i];
            auto size = c->width();
            ALSp<ALFloatMatrix> inv_cov = ALFloatMatrix::create(size, size);
            ALFLOAT cov_det = ALFloatMatrix::inverse_basic(c.get(), inv_cov.get());
            mInverseCovs.push_back(inv_cov);
            //mInverseCovs.push_back(c);//For Debug
            //mCoefs.push_back(basic_parameter*coefs[i]/sqrt(cov_det));
            mCoefs.push_back(coefs[i]/sqrt(cov_det+0.00000001));
        }
    }
    virtual ~ALGMMModel()
    {
    }
    virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const
    {
        ALASSERT(NULL!=X);
        ALASSERT(NULL!=Y);
        ALASSERT(X->height() == Y->height());
        ALASSERT(1 == Y->width());
        ALASSERT(X->width() == mCenters->width());
        auto h = X->height();
        auto w = X->width();
        ALAUTOSTORAGE(diff, ALFLOAT, w);
        for (int i=0; i<h; ++i)
        {
            ALFLOAT sum = 0.0;
            ::memset(diff, 0, sizeof(ALFLOAT)*w);
            auto data_ = X->vGetAddr(i);
            for (int j=0; j<mInverseCovs.size(); ++j)
            {
                ALFLOAT multi = 0.0;
                auto center_ = mCenters->vGetAddr(j);
                auto invcov_ = mInverseCovs[j]->vGetAddr(0);
                for (int k=0; k<w; ++k)
                {
                    diff[k] = data_[k] - center_[k];
                }
                for (int x=0; x<w; ++x)
                {
                    ALFLOAT csum = 0.0;
                    for (int y=0; y<w; ++y)
                    {
                        csum += invcov_[x*w+y]*diff[y];
                    }
                    multi += csum*diff[x];
                }
                sum += mCoefs[j]*exp(-0.5f*multi);
            }
            *(Y->vGetAddr(i)) = sum;
        }
        
    }
    virtual void vPrint(std::ostream& output) const
    {
        output << "<ALGMM>\n";
        output << "<Centers>\n";
        ALFloatMatrix::print(mCenters.get(), output);
        output << "</Centers>\n";
        output << "<InverseCovs>\n";
        for (auto cov : mInverseCovs)
        {
            ALFloatMatrix::print(cov.get(), output);
            output << "\n";
        }
        output << "</InverseCovs>\n";
        output << "<Coef>\n";
        for (auto c : mCoefs)
        {
            output << c << " ";
        }
        output << "</Coef>\n";
        output << "</ALGMM>\n";
    }
private:
    ALSp<ALFloatMatrix> mCenters;
    std::vector<ALSp<ALFloatMatrix>> mInverseCovs;
    std::vector<ALFLOAT> mCoefs;
};


ALGMM::ALGMM(int centernumber)
{
    mCenters = centernumber;
}
ALGMM::~ALGMM()
{
}

ALIMatrixPredictor* ALGMM::vLearn(const ALFloatMatrix* data) const
{
    ALASSERT(NULL!=data);
    if(mCenters > data->height()/10)
    {
        FUNC_PRINT(mCenters);
        FUNC_PRINT(data->height());
        return new ALDummyMatrixPredictor;
    }
    ALSp<ALFloatMatrix> centerPoints = ALKMeans::learn(data, mCenters);
    /*Compute the cov*/
    auto w = data->width();
    auto h = data->height();
    int centers = mCenters;
    std::vector<ALSp<ALFloatMatrix>> cov_matrixs;
    for (int i=0; i<centers; ++i)
    {
        cov_matrixs.push_back(ALFloatMatrix::create(w, w));
    }
    for (int i=0; i<centers; ++i)
    {
        ALFloatMatrix::zero(cov_matrixs[i].get());
    }
    ALSp<ALFloatMatrix> mask = ALFloatMatrix::create(1, h);
    ALKMeans::predict(data, centerPoints.get(), mask.get());
    auto mask_ = mask->vGetAddr();
    ALAUTOSTORAGE(counts, int, centers);
    ::memset(counts, 0, centers*sizeof(int));
    for (int i=0; i<h; ++i)
    {
        auto data_ = data->vGetAddr(i);
        int class_num = mask_[i];
        auto targetCov = cov_matrixs[class_num];
        counts[class_num]+=1;
        auto center = centerPoints->vGetAddr(class_num);
        for (int j=0; j<w; ++j)
        {
            auto targetCov_line = targetCov->vGetAddr(j);
            for (int k=j; k<w; ++k)
            {
                targetCov_line[k] += (data_[j]-center[j])*(data_[k]-center[k]);
            }
        }
    }
    /*The coes is comput as c[i]/total*/
    std::vector<ALFLOAT> coefs;
    for (int i=0; i<centers; ++i)
    {
        if (0 == counts[i])
        {
            ALFloatMatrix::zero(cov_matrixs[i].get());
            continue;
        }
        ALFLOAT count = counts[i];
        auto _cov = cov_matrixs[i]->vGetAddr(0);
        coefs.push_back(count / (float)h);
        for (int j=0; j<w; ++j)
        {
            for (int k=j; k<w; ++k)
            {
                ALFLOAT temp = _cov[j*w+k]/count;
                _cov[j*w+k] = temp;
                _cov[k*w+j] = temp;
            }
        }
    }
    
    return new ALGMMModel(centerPoints, cov_matrixs, coefs);
}

