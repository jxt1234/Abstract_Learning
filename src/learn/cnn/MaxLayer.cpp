//
//  MaxLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 10/11/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "MaxLayer.hpp"

#include <fstream>
#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    MaxLayer::MaxLayer(size_t iw, size_t ow):ILayer(iw, ow, 0, 0, 0, 0)
    {
        ALASSERT(iw > ow);
        ALASSERT(iw%ow == 0);
        mStride = iw/ow;
    }
    MaxLayer::~MaxLayer()
    {
    }
    void MaxLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        auto function = [](ALFLOAT* dst, ALFLOAT* src, size_t w) {
            for (size_t i=0; i<w; ++i)
            {
                if (src[i] > dst[i])
                {
                    dst[i] = src[i];
                }
            }
        };
        auto ow = after->width();
        auto h = after->height();
        ALSp<ALFloatMatrix> crop = ALFloatMatrix::createCropVirtualMatrix(before, 0, 0, ow-1, h-1);
        ALFloatMatrix::copy(after, crop.get());
        for (size_t i=1; i<mStride; ++i)
        {
            ALSp<ALFloatMatrix> crop = ALFloatMatrix::createCropVirtualMatrix(before, i*ow, 0,( i+1)*ow-1, h-1);
            ALFloatMatrix::runLineFunction(after, crop.get(), function);
        }
    }
    void MaxLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        if (NULL == before_diff)
        {
            return;
        }
        auto ow = after_diff->width();
        auto h = after_diff->height();
        ALFloatMatrix::zero( before_diff);
        for (size_t j=0; j<h; ++j)
        {
            auto src_diff = before_diff->vGetAddr(j);
            auto dst_diff = after_diff->vGetAddr(j);
            auto dst = after->vGetAddr(j);
            auto src = before->vGetAddr(j);
            for (size_t i=0; i<mStride; ++i)
            {
                if (ZERO(src[i*ow+j]-dst[j]))
                {
                    src_diff[i*ow+j] = dst_diff[j];
                    break;
                }
            }
        }
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new MaxLayer(p.uInputSize, p.uOutputSize);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "Max");
}
