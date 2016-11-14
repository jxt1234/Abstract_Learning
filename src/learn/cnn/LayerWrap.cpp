#include "LayerWrap.h"
#include <fstream>
#define DUMP(x) {std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/dump/."#x); ALFloatMatrix::print(x.get(), output);}
#define DUMP2(x) {std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/dump/."#x); ALFloatMatrix::print(x, output);}

namespace ALCNN {
    LayerWrap::LayerWrap(ALSp<ILayer> layer)
    {
        ALASSERT(layer.get()!=NULL);
        mLayer = layer;
        mBefore = NULL;
        mNext = NULL;
        mCacheSize = layer->getInfo().cw * layer->getInfo().ch;
    }
    
    LayerWrap::~LayerWrap()
    {
    }
    
    size_t LayerWrap::getParameterSize() const
    {
        auto info = mLayer->getInfo();
        if (NULL == mNext.get())
        {
            return info.pw*info.ph;
        }
        return info.pw*info.ph+mNext->getParameterSize();
    }

    
    void LayerWrap::mapParameters(const ALFloatMatrix* p, size_t offset)
    {
        ALASSERT(NULL!=p);
        ALASSERT(offset <= p->width());
        ALASSERT(offset>=0);
        auto p_cur = p->vGetAddr()+offset;
        auto info = mLayer->getInfo();
        mParameters = ALFloatMatrix::createRefMatrix(p_cur, info.pw, info.ph);
        auto currentOffset = info.pw*info.ph;
        if (NULL != mNext.get())
        {
            mNext->mapParameters(p, offset+currentOffset);
        }
    }
    void LayerWrap::mapParametersDiff(const ALFloatMatrix* p, size_t offset)
    {
        ALASSERT(NULL!=p);
        ALASSERT(offset <= p->width());
        ALASSERT(offset>=0);
        auto p_cur = p->vGetAddr()+offset;
        auto info = mLayer->getInfo();
        mParameterDiff = ALFloatMatrix::createRefMatrix(p_cur, info.pw, info.ph);
        auto currentOffset = info.pw*info.ph;
        if (NULL != mNext.get())
        {
            mNext->mapParametersDiff(p, offset+currentOffset);
        }
    }

    ALSp<ALFloatMatrix> LayerWrap::forward(ALSp<ALFloatMatrix> input)
    {
        if ((mCache.get() == NULL || mCache->height() != input->height()) && mCacheSize > 0)
        {
            mCache = ALFloatMatrix::create(mCacheSize, input->height());
        }
        if (mOutput.get() == NULL || mOutput->height() != input->height())
        {
            mOutput = ALFloatMatrix::create(mLayer->getInfo().ow, input->height());
        }
        mInput = input;
        ALASSERT(NULL!=mOutput.get());
        mLayer->vForward(input.get(), mOutput.get(), mParameters.get(), mCache.get());
        
        if(ALFloatMatrix::checkNAN(mOutput.get()))
        {
            DUMP(input);
            DUMP(mOutput);
            if (NULL!=mParameters.get())
            {
                DUMP(mParameters);
            }
            if (NULL!=mCache.get())
            {
                DUMP(mCache);
            }
            ALASSERT(false);
        }
        
        if (NULL != mForwardDump)
        {
            ALFloatMatrix::print(mOutput.get(), *mForwardDump);
            mForwardDump->flush();
        }
        if (NULL!=mNext.get())
        {
            return mNext->forward(mOutput);
        }
        return mOutput;
    }
    
    void LayerWrap::backward(ALSp<ALFloatMatrix> error)
    {
        ALASSERT(error->height() == mOutput->height());
        ALASSERT(mInput.get()!=NULL);
        if (NULL!=mBefore)
        {
            ALSp<ALFloatMatrix> inputError = ALFloatMatrix::create(mInput->width(), mInput->height());
            mLayer->vBackward(error.get(), mOutput.get(), mParameters.get(), mInput.get(), inputError.get(), mParameterDiff.get(), mCache.get());
            mBefore->backward(inputError);
        }
        else
        {
            mLayer->vBackward(error.get(), mOutput.get(), mParameters.get(), mInput.get(), NULL, mParameterDiff.get(), mCache.get());
        }
    }

    void LayerWrap::connectInput(LayerWrap* input)
    {
        ALASSERT(NULL == mBefore);
        ALASSERT(NULL!=input);
        mBefore = input;
    }
    void LayerWrap::connectOutput(ALSp<LayerWrap> output)
    {
        ALASSERT(NULL == mNext.get());
        ALASSERT(NULL!=output.get());
        mNext = output;
    }
    LayerWrap* LayerWrap::getLastLayer()
    {
        if (NULL == mNext.get())
        {
            return this;
        }
        return mNext->getLastLayer();
    }
}
