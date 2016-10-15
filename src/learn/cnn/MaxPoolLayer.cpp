//
//  MaxPoolLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 15/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "MaxPoolLayer.hpp"
#include <fstream>

namespace ALCNN {
    MaxPoolLayer::MaxPoolLayer(int stride, int width, int height, int depth)
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
    
    ALFloatMatrix* MaxPoolLayer::vInitOutput(int batchSize) const
    {
        ALASSERT(batchSize>0);
        return ALFloatMatrix::create(mOutput.getTotalWidth(), batchSize);
    }
    bool MaxPoolLayer::vCheckInput(const ALFloatMatrix* input) const
    {
        return mInput.getTotalWidth() == input->width();
    }
    void MaxPoolLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=after);
        ALASSERT(NULL!=before);
        ALASSERT(before->height() == after->height());
        ALASSERT(before->width() == mInput.getTotalWidth());
        ALASSERT(after->width() == mOutput.getTotalWidth());
        auto batchSize = after->height();
        ALAUTOSTORAGE(srcLines, ALFLOAT*, mStride);
        for (int z=0; z<batchSize; ++z)
        {
            for (int p=0; p<mInput.iDepth; ++p)
            {
                ALSp<ALFloatMatrix> input = ALFloatMatrix::createRefMatrix(before->vGetAddr(z)+p*mInput.iWidth*mInput.iHeight, mInput.iWidth, mInput.iHeight);
                ALSp<ALFloatMatrix> output = ALFloatMatrix::createRefMatrix(after->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight, mOutput.iWidth, mOutput.iHeight);
                auto h = mOutput.iHeight;
                auto w = mOutput.iWidth;
                for (int i=0; i<h; ++i)
                {
                    auto dst = output->vGetAddr(i);
                    for (int k=0; k<mStride; ++k)
                    {
                        srcLines[k] = input->vGetAddr(mStride*i+k);
                    }
                    for (int j=0; j<w; ++j)
                    {
                        ALFLOAT maxNumber = srcLines[0][0];
                        for (int x=0; x<mStride; ++x)
                        {
                            for (int y=0; y<mStride; ++y)
                            {
                                auto s = srcLines[x][y+mStride*j];
                                if (maxNumber < s)
                                {
                                    maxNumber = s;
                                }
                            }
                        }
                        dst[j] = maxNumber;
                    }
                }
            }
        }
    }
    void MaxPoolLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=after_diff);
        ALASSERT(NULL!=after);
        ALASSERT(after->height() == after_diff->height());
        ALASSERT(before_diff->width() == mInput.getTotalWidth());
        ALASSERT(after_diff->width() == mOutput.getTotalWidth());
        auto batchSize = after_diff->height();
        ALAUTOSTORAGE(srcLines, ALFLOAT*, mStride);
        ALAUTOSTORAGE(srcDiffLines, ALFLOAT*, mStride);
        ALFloatMatrix::zero(before_diff);
        for (int z=0; z<batchSize; ++z)
        {
            for (int p=0; p<mInput.iDepth; ++p)
            {
                ALSp<ALFloatMatrix> input = ALFloatMatrix::createRefMatrix(before->vGetAddr(z)+p*mInput.iWidth*mInput.iHeight, mInput.iWidth, mInput.iHeight);
                ALSp<ALFloatMatrix> input_diff = ALFloatMatrix::createRefMatrix(before_diff->vGetAddr(z)+p*mInput.iWidth*mInput.iHeight, mInput.iWidth, mInput.iHeight);
                ALSp<ALFloatMatrix> output = ALFloatMatrix::createRefMatrix(after_diff->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight, mOutput.iWidth, mOutput.iHeight);
                auto h = mOutput.iHeight;
                auto w = mOutput.iWidth;
                for (int i=0; i<h; ++i)
                {
                    auto dst = output->vGetAddr(i);
                    for (int k=0; k<mStride; ++k)
                    {
                        srcLines[k] = input->vGetAddr(mStride*i+k);
                        srcDiffLines[k] = input_diff->vGetAddr(mStride*i+k);
                    }
                    for (int j=0; j<w; ++j)
                    {
                        ALFLOAT maxNumber = srcLines[0][0];
                        int maxX = 0;
                        int maxY = 0;
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
                        srcDiffLines[maxX][maxY+mStride*j] = dst[j];
                    }
                }
            }
        }
    }
}