#include "SoftMaxLayer.h"
#include <math.h>
#include "math/ALStatistics.h"
#include <fstream>
#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    SoftMaxLayer::SoftMaxLayer(int inputSize, int outputSize)
    {
        ALASSERT(outputSize>=1);
        ALASSERT(inputSize>=1);
        ALASSERT(inputSize == outputSize);
        mOutputSize = outputSize;
        mInputSize = inputSize;
    }
    SoftMaxLayer::~SoftMaxLayer()
    {
    }
    ALFloatMatrix* SoftMaxLayer::vInitParameters() const
    {
        return NULL;
    }
    
    ALFloatMatrix* SoftMaxLayer::vInitOutput(int batchSize) const
    {
        return ALFloatMatrix::create(mOutputSize, batchSize);
    }
    bool SoftMaxLayer::vCheckInput(const ALFloatMatrix* input) const
    {
        return input->width() == mInputSize;
    }
    void SoftMaxLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=before);
        ALASSERT(NULL!=after);
        ALASSERT(before->width() == mInputSize);
        ALASSERT(after->width() == mOutputSize);
        ALASSERT(before->height() == after->height());
        ALSp<ALFloatMatrix> dot = ALFloatMatrix::create(before->width(), before->height());
        ALFloatMatrix::copy(dot.get(), before);
        auto w = dot->width();
        auto h = dot->height();
        
        /*Pretreat dot, for compute precision, like caffe*/
        for (int i=0; i<h; ++i)
        {
            auto _dot = dot->vGetAddr(i);
            ALFLOAT maxNumber = _dot[0];
            for (int j=1; j<w; ++j)
            {
                if (_dot[j] > maxNumber)
                {
                    maxNumber = _dot[j];
                }
            }
            for (int j=0; j<w; ++j)
            {
                _dot[j] -= maxNumber;
            }
        }
        
        /*Compute*/
        for (int i=0; i<h; ++i)
        {
            auto src = dot->vGetAddr(i);
            auto dst = after->vGetAddr(i);
            ALFLOAT dstSum = 0.0f;
            for (int j=0; j<w; ++j)
            {
                dst[j] = ::exp(src[j]);
                dstSum+=dst[j];
            }
            for (int j=0; j<w; ++j)
            {
                dst[j]/=dstSum;
            }
        }
    }
    void SoftMaxLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=after);
        ALASSERT(NULL!=after_diff);
        ALASSERT(after->width() == after_diff->width());
        ALASSERT(after->height() == after_diff->height());
        
        auto batchSize = after->height();
        /*Compute input diff*/
        if (NULL == before_diff)
        {
            return;
        }
        auto w = before->width();
        for (int z = 0; z<batchSize; ++z)
        {
            auto ydet = after_diff->vGetAddr(z);
            auto y = after->vGetAddr(z);
            auto x = before_diff->vGetAddr(z);
            for (int i=0; i<w; ++i)
            {
                x[i] = (y[i]-y[i]*y[i])*ydet[i];
            }
        }
        if (0)
        {
            std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/temp_input_diff_soft_max.txt");
            ALFloatMatrix::print(before_diff, output);
        }
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new SoftMaxLayer(p.uInputSize, p.uOutputSize);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "SoftMax");
}

