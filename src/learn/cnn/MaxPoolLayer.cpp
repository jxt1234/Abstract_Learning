//
//  MaxPoolLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 15/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "MaxPoolLayer.hpp"
#include <fstream>
#include "LayerFactoryRegistor.hpp"

namespace ALCNN {
    MaxPoolLayer::MaxPoolLayer(int stride, int width, int height, int depth):ILayer(width*height*depth, width*height*depth/stride/stride, 0, 0, width*height*depth/stride/stride, 1)
    {
        ALASSERT(stride>=2);
        ALASSERT(width>0);
        ALASSERT(height>0);
        ALASSERT(depth>0);
        ALASSERT(width%stride==0);
        ALASSERT(height%stride==0);
        mInput.iExpand = 0;
        mInput.iHeight = height;
        mInput.iWidth = width;
        mInput.iDepth = depth;
        
        mOutput.iDepth = depth;
        mOutput.iExpand = 0;
        mOutput.iWidth = width/stride;
        mOutput.iHeight = height/stride;
        mStride = stride;
    }
    MaxPoolLayer::~ MaxPoolLayer()
    {
    }
    
    void MaxPoolLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=after);
        ALASSERT(NULL!=before);
        ALASSERT(before->height() == after->height());
        ALASSERT(before->width() == mInput.getTotalWidth());
        ALASSERT(after->width() == mOutput.getTotalWidth());
        auto batchSize = after->height();
        ALAUTOSTORAGE(srcLines, ALFLOAT*, mStride);
        auto h = mOutput.iHeight;
        auto w = mOutput.iWidth;
        for (int z=0; z<batchSize; ++z)
        {
            for (int p=0; p<mInput.iDepth; ++p)
            {
                auto output_p = after->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight;
                auto input_p = before->vGetAddr(z)+p*mInput.iWidth*mInput.iHeight;
                auto cache_p = cache->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight;

                for (int i=0; i<h; ++i)
                {
                    auto dst = output_p + mOutput.iWidth*i;
                    auto cache_dst = cache_p + mOutput.iWidth*i;
                    for (int k=0; k<mStride; ++k)
                    {
                        srcLines[k] = input_p + (mStride*i+k)*mInput.iWidth;
                    }
                    for (int j=0; j<w; ++j)
                    {
                        ALFLOAT maxNumber = srcLines[0][0];
                        size_t maxX = 0;
                        size_t maxY = 0;
                        for (int x=0; x<mStride; ++x)
                        {
                            for (int y=0; y<mStride; ++y)
                            {
                                auto s = srcLines[x][y+mStride*j];
                                if (maxNumber < s)
                                {
                                    maxNumber = s;
                                    maxX = x;
                                    maxY = y;
                                }
                            }
                        }
                        dst[j] = maxNumber;
                        cache_dst[j] = maxX + mStride*maxY;
                    }
                }
            }
        }
    }
    void MaxPoolLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=after_diff);
        ALASSERT(NULL!=after);
        ALASSERT(after->height() == after_diff->height());
        ALASSERT(before_diff->width() == mInput.getTotalWidth());
        ALASSERT(after_diff->width() == mOutput.getTotalWidth());
        auto batchSize = after_diff->height();
        ALFloatMatrix::zero(before_diff);
        auto h = mOutput.iHeight;
        auto w = mOutput.iWidth;
        for (int z=0; z<batchSize; ++z)
        {
            for (int p=0; p<mInput.iDepth; ++p)
            {
                auto output_p = after_diff->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight;
                auto input_diff_p = before_diff->vGetAddr(z)+p*mInput.iWidth*mInput.iHeight;
                auto cache_p = cache->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight;

                for (int i=0; i<h; ++i)
                {
                    auto dst = output_p + i*mOutput.iWidth;
                    auto cache_dst = cache_p + mOutput.iWidth*i;
                    for (int j=0; j<w; ++j)
                    {
                        auto mask = (size_t)(cache_dst[j]);
                        auto maxX = mask % mStride;
                        auto maxY = (mask - maxX)/mStride;
                        *(input_diff_p+(mStride*i+maxY)*mInput.iWidth + mStride*j+maxX) = dst[j];
                    }
                }
            }
        }
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new MaxPoolLayer(p.get("stride"), p.mMatrixInfo.iWidth, p.mMatrixInfo.iHeight, p.mMatrixInfo.iDepth);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "MaxPool");

}
