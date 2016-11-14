#include "SelectLastLayer.h"
#include <fstream>
#include "LayerFactoryRegistor.hpp"

namespace ALCNN {
    SelectLastLayer::SelectLastLayer(size_t iw, size_t ow):ILayer(iw+1, ow, 0, 0, 0, 0)
    {
        ALASSERT(iw > ow);
        ALASSERT(iw%ow == 0);
        mOw = ow;
    }
    SelectLastLayer::~SelectLastLayer()
    {
    }
    void SelectLastLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        auto function = [&](ALFLOAT* dst, ALFLOAT* src, size_t w) {
            ALASSERT(mOw == w);
            size_t t = src[0];
            auto _src = src + mOw*(t-1);
            ::memcpy(dst, _src, w*sizeof(ALFLOAT));
        };
        ALFloatMatrix::runLineFunction(after, before, function);
    }
    void SelectLastLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALASSERT(NULL!=before_diff);
        ALFloatMatrix::zero(before_diff);
        auto h = after_diff->height();
        for (size_t i=0; i<h; ++i)
        {
            auto src_diff = before_diff->vGetAddr(i);
            auto src = before->vGetAddr(i);
            src_diff[0] = src[0];
            auto dst_diff = after_diff->vGetAddr(i);
            size_t time = src[0];
            ::memcpy(src_diff+1+(time-1)*mOw, dst_diff, mOw*sizeof(ALFLOAT));
        }
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new SelectLastLayer(p.uInputSize, p.uOutputSize);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "Last");
}
