//
//  EmbeddingLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 10/11/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "EmbeddingLayer.hpp"
#include <fstream>
#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    EmbeddingLayer::EmbeddingLayer(size_t iw, size_t ow, size_t time, size_t number):ILayer(iw, ow+1, ow/time, number, 0, 0)
    {
        ALASSERT(iw%time == 0);
        ALASSERT(ow%time == 0);
        mNumber = number;
    }
    EmbeddingLayer::~EmbeddingLayer()
    {
    }
    void EmbeddingLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        auto info = getInfo();
        auto time = info.iw;
        auto ow = (info.ow-1) / time;
        ALASSERT((info.ow-1)%time == 0);
        auto h = after->height();
        ALASSERT(parameters->width() == ow);
        ALASSERT(parameters->height() == mNumber);
        ALFloatMatrix::zero(after);
        for (size_t y=0; y<h; ++y)
        {
            auto dst = after->vGetAddr(y);
            auto src = before->vGetAddr(y);
            size_t real_t = src[0];
            dst[0] = real_t;
            ALASSERT(time >= real_t);
            for (size_t t=0; t<real_t; ++t)
            {
                auto _dst = dst + t*ow + 1;
                size_t index = src[t+1];
                auto _src = parameters->vGetAddr(index);
                ::memcpy(_dst, _src, ow*sizeof(ALFLOAT));
            }
        }
    }
    void EmbeddingLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALASSERT(NULL == before_diff);
        auto info = getInfo();
        auto time = info.iw;
        auto ow = (info.ow-1) / time;
        ALASSERT((info.ow-1)%time == 0);
        auto h = after->height();
        ALFloatMatrix::zero(parameters_diff);

        for (size_t y=0; y<h; ++y)
        {
            auto dst_diff = after_diff->vGetAddr(y);
            auto src = before->vGetAddr(y);
            size_t real_t = src[0];
            ALASSERT(time >= real_t);

            for (size_t t=0; t<real_t; ++t)
            {
                auto _dst_diff = dst_diff + t*ow + 1;
                size_t index = src[t+1];
                auto _p_diff = parameters_diff->vGetAddr(index);
                for (size_t i=0; i<ow; ++i)
                {
                    _p_diff[i] += _dst_diff[i];
                }
            }
        }

    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new EmbeddingLayer(p.uInputSize, p.uOutputSize, p.get("time"), p.get("number"));
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "Embedding");
};