#include "SoftMaxLayer.h"
#include <math.h>
namespace ALCNN {
    SoftMaxLayer::SoftMaxLayer(int outputSize)
    {
        ALASSERT(outputSize>=1);
        mOutputSize = outputSize;
    }
    SoftMaxLayer::~SoftMaxLayer()
    {
    }
    
    
    ALFloatMatrix* SoftMaxLayer::vInitOutput(int batchSize) const
    {
        return ALFloatMatrix::create(mOutputSize, batchSize);
    }
    bool SoftMaxLayer::vCheckInput(const ALFloatMatrix* input) const
    {
        return input->width() == mOutputSize;
    }
    void SoftMaxLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const
    {
        ALASSERT(NULL!=before);
        ALASSERT(NULL!=after);
        ALASSERT(before->width() == after->width());
        ALASSERT(before->height() == after->height());
        auto w = before->width();
        auto h = before->height();
        for (int i=0; i<h; ++i)
        {
            auto src = before->vGetAddr(i);
            auto dst = after->vGetAddr(i);
            for (int j=0; j<w; ++j)
            {
                dst[j] = 1.0/(1.0+::exp(-src[j]));
            }
        }
    }
    void SoftMaxLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const
    {
        ALASSERT(NULL!=after);
        ALASSERT(NULL!=after_diff);
        ALASSERT(NULL!=before_diff);
        ALASSERT(after->width() == after_diff->width());
        ALASSERT(after->height() == after_diff->height());
        auto w = after->width();
        auto h = after->height();
        
        for (int i=0; i<h; ++i)
        {
            auto srcDiff = before_diff->vGetAddr(i);
            auto dst = after->vGetAddr(i);
            auto dstDiff = after_diff->vGetAddr(i);
            for (int j=0; j<w; ++j)
            {
                srcDiff[j] = dst[j]*(1.0f-dst[j])*dstDiff[j];
            }
        }
    }
}

