#include "learn/ALMatrixNormalizer.h"
#include "math/ALStatistics.h"
ALMatrixNormalizer::ALMatrixNormalizer(const ALFloatMatrix* train, ALFLOAT rate)
{
    ALASSERT(NULL!=train);
    ALSp<ALFloatMatrix> min_max = ALStatistics::statistics(train);
    auto w = min_max->width();
    auto max_ = min_max->vGetAddr(2);
    auto min_ = min_max->vGetAddr(1);
    std::vector<int> validPos;
    for (size_t i=0; i<w; ++i)
    {
        if(!(ZERO(max_[i]-min_[i])))
        {
            validPos.push_back(i);
        }
    }
    if (validPos.size() != w)
    {
        mSelect = new ALMatrixSelector(validPos);
    }
    mK = ALFloatMatrix::create(validPos.size(), 1);
    mB = ALFloatMatrix::create(validPos.size(), 1);
    auto k = mK->vGetAddr(0);
    auto b = mB->vGetAddr(0);
    for (size_t i=0; i<validPos.size(); ++i)
    {
        int pos = validPos[i];
        ALASSERT(!(ZERO(max_[pos]-min_[pos])));
        k[i] = 1.0f/(max_[pos]-min_[pos]);
        b[i] = -(min_[pos])/(max_[pos]-min_[pos]);
    }
}
ALMatrixNormalizer::~ALMatrixNormalizer()
{
    
}

ALFloatMatrix* ALMatrixNormalizer::vTransform(const ALFloatMatrix* origin_) const
{
    ALASSERT(NULL!=origin_);
    const ALFloatMatrix* origin = origin_;
    ALSp<ALFloatMatrix> originP;
    if (NULL!=mSelect.get())
    {
        originP = mSelect->vTransform(origin_);
        origin = originP.get();
    }
    ALASSERT(origin->width() == mK->width() && origin->width() == mB->width());
    auto w = origin->width();
    auto h = origin->height();
    ALFloatMatrix* result = ALFloatMatrix::create(w, h);
    auto k = mK->vGetAddr(0);
    auto b = mB->vGetAddr(0);
    for (size_t y=0; y<h; ++y)
    {
        auto origin_ = origin->vGetAddr(y);
        auto result_ = result->vGetAddr(y);
        for (size_t x=0; x<w; ++x)
        {
            result_[x] = origin_[x]*k[x]+b[x];
        }
    }
    return result;
}
