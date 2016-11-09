#include "MeanLayer.h"
#include <fstream>
#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    MeanLayer::MeanLayer(size_t iw, size_t ow):ILayer(iw, ow, 0, 0, 0, 0)
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
        ALFLOAT rate = 1.0/mStride;
        auto function = [rate](ALFLOAT* dst, ALFLOAT* src, size_t w) {
            for (size_t i=0; i<w; ++i)
            {
                dst[i] += src[i]*rate;
            }
        };
        auto ow = after->width();
        auto h = after->height();
        ALFloatMatrix::zero(after);
        for (size_t i=0; i<mStride; ++i)
        {
            ALSp<ALFloatMatrix> crop = ALFloatMatrix::createCropVirtualMatrix(before, i*ow, 0,( i+1)*ow-1, h-1);
            ALFloatMatrix::runLineFunction(after, crop.get(), function);
        }
    }
    void MeanLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        if (NULL == before_diff)
        {
            return;
        }
        ALFLOAT rate = 1.0/mStride;
        auto function = [rate](ALFLOAT* dst, ALFLOAT* src, size_t w) {
            for (size_t i=0; i<w; ++i)
            {
                dst[i] = src[i]*rate;
            }
        };
        auto ow = after_diff->width();
        auto h = after_diff->height();
        for (size_t i=0; i<mStride; ++i)
        {
            ALSp<ALFloatMatrix> crop = ALFloatMatrix::createCropVirtualMatrix(before_diff, i*ow, 0,( i+1)*ow-1, h-1);
            ALFloatMatrix::runLineFunction(crop.get(), after_diff, function);
        }
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new MeanLayer(p.uInputSize, p.uOutputSize);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "Mean");
}
