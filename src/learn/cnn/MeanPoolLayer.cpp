#include "MeanPoolLayer.h"

namespace ALCNN {
    MeanPoolLayer::MeanPoolLayer(int stride, int width, int height, int depth)
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
    MeanPoolLayer::~ MeanPoolLayer()
    {
    }
    
    ALFloatMatrix* MeanPoolLayer::vInitOutput(int batchSize) const
    {
        ALASSERT(batchSize>0);
        return ALFloatMatrix::create(mOutput.getTotalWidth(), batchSize);
    }
    bool MeanPoolLayer::vCheckInput(const ALFloatMatrix* input) const
    {
        return mInput.getTotalWidth() == input->width();
    }
    void MeanPoolLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const
    {
        ALASSERT(NULL!=after);
        ALASSERT(NULL!=before);
        ALASSERT(before->height() == after->height());
        ALASSERT(before->width() == mInput.getTotalWidth());
        ALASSERT(after->width() == mOutput.getTotalWidth());
        auto batchSize = after->height();
        for (int z=0; z<batchSize; ++z)
        {
            for (int p=0; p<mInput.iDepth; ++p)
            {
                ALSp<ALFloatMatrix> input = ALFloatMatrix::createRefMatrix(before->vGetAddr(z)+p*mInput.iWidth*mInput.iHeight, mInput.iWidth, mInput.iHeight);
                ALSp<ALFloatMatrix> output = ALFloatMatrix::createRefMatrix(after->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight, mOutput.iWidth, mOutput.iHeight);
                ALAUTOSTORAGE(srcLines, ALFLOAT*, mStride);
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
                        ALFLOAT sum = 0.0;
                        for (int x=0; x<mStride; ++x)
                        {
                            for (int y=0; y<mStride; ++y)
                            {
                                sum += srcLines[x][y+mStride*j];
                            }
                        }
                        dst[j] = sum / (ALFLOAT)(mStride*mStride);
                    }
                }
            }
        }
    }
    void MeanPoolLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const
    {
        ALASSERT(NULL!=after_diff);
        ALASSERT(NULL!=after);
        ALASSERT(after->height() == after_diff->height());
        ALASSERT(before_diff->width() == mInput.getTotalWidth());
        ALASSERT(after_diff->width() == mOutput.getTotalWidth());
        auto batchSize = after_diff->height();
        for (int z=0; z<batchSize; ++z)
        {
            for (int p=0; p<mInput.iDepth; ++p)
            {
                ALSp<ALFloatMatrix> input = ALFloatMatrix::createRefMatrix(before_diff->vGetAddr(z)+p*mInput.iWidth*mInput.iHeight, mInput.iWidth, mInput.iHeight);
                ALSp<ALFloatMatrix> output = ALFloatMatrix::createRefMatrix(after_diff->vGetAddr(z)+p*mOutput.iWidth*mOutput.iHeight, mOutput.iWidth, mOutput.iHeight);
                ALAUTOSTORAGE(srcLines, ALFLOAT*, mStride);
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
                        ALFLOAT ave = dst[j] / (ALFLOAT)(mStride*mStride);
                        for (int x=0; x<mStride; ++x)
                        {
                            for (int y=0; y<mStride; ++y)
                            {
                                srcLines[x][y+mStride*j] = ave;
                            }
                        }
                    }
                }
            }
        }
    }
}