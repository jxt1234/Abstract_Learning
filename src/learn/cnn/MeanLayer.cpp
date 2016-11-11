#include "MeanLayer.h"
#include <fstream>
#include "LayerFactoryRegistor.hpp"
#define DUMP(x) {std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/dump/."#x); ALFloatMatrix::print(x.get(), output);}
#define DUMP2(x) {std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/dump/."#x); ALFloatMatrix::print(x, output);}

namespace ALCNN {
    MeanLayer::MeanLayer(size_t iw, size_t ow):ILayer(iw+1, ow, 0, 0, 0, 0)
    {
        ALASSERT(iw > ow);
        ALASSERT(iw%ow == 0);
        mStride = iw/ow;
    }
    MeanLayer::~MeanLayer()
    {
    }
    void MeanLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        auto function = [](ALFLOAT* dst, ALFLOAT* src, size_t w) {
            ALFLOAT rate = 1.0/src[0];
            size_t t = src[0];
            for (size_t i=0; i<w; ++i)
            {
                dst[i] = 0;
                for (size_t k=0; k<t; ++k)
                {
                    dst[i] += src[1+i+k*w]*rate;
                }
            }
        };
        ALFloatMatrix::runLineFunction(after, before, function);
    }
    void MeanLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        if (NULL == before_diff)
        {
            return;
        }
        ALFloatMatrix::zero(before_diff);
        auto ow = after_diff->width();
        auto h = after_diff->height();
        for (size_t i=0; i<h; ++i)
        {
            auto src_diff = before_diff->vGetAddr(i);
            auto src = before->vGetAddr(i);
            src_diff[0] = src[0];
            auto dst_diff = after_diff->vGetAddr(i);
            size_t time = src[0];
            ALFLOAT rate = 1.0/src[0];
            for (size_t j=0; j<ow; ++j)
            {
                for (size_t t=0; t<time; ++t)
                {
                    src_diff[1+j+t*ow] = dst_diff[j]*rate;
                }
            }
        }
//        DUMP2(before_diff);
//        DUMP2(before);
//        DUMP2(after_diff);
//        DUMP2(after);
        
        ow = 0;
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new MeanLayer(p.uInputSize, p.uOutputSize);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "Mean");
}
