//
//  ReluLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 14/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "ReluLayer.hpp"
#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    void ReluLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        ALASSERT(NULL!=before);
        ALASSERT(NULL!=after);
        ALASSERT(after->height() == before->height());
        ALASSERT(after->width() == before->width());
        auto h = after->height();
        auto w = after->width();
        for (int i=0; i<h; ++i)
        {
            auto dst = after->vGetAddr(i);
            auto src = before->vGetAddr(i);
            for (int j=0; j<w; ++j)
            {
                dst[j] = src[j];
                if (src[j]<0)
                {
                    dst[j] = mSlope;
                }
            }
        }
    }
    void ReluLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALASSERT(NULL!=before);
        ALASSERT(NULL!=after);
        ALASSERT(after->height() == before->height());
        ALASSERT(after->width() == before->width());
        auto h = after->height();
        auto w = after->width();
        for (int i=0; i<h; ++i)
        {
            auto dstDiff = after_diff->vGetAddr(i);
            auto srcDiff = before_diff->vGetAddr(i);
            auto src = before->vGetAddr(i);
            for (int j=0; j<w; ++j)
            {
                if (src[j]<0)
                {
                    srcDiff[j] = mSlope;
                }
                else
                {
                    srcDiff[j] = dstDiff[j];
                }
            }
        }
    }
    
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new ReluLayer(p.uInputSize);
    };

    static LayerFactoryRegister __reg(gCreateFunction, "ReLU");
}
