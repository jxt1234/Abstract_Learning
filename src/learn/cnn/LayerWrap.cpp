#include "LayerWrap.h"
namespace ALCNN {
    LayerWrap::LayerWrap(ALSp<ILayer> layer)
    {
        mLayer = layer;
        mParameters = layer->vInitParameters();
        mParameterDiff = layer->vInitParameters();
        mBefore = NULL;
        mNext = NULL;
    }
    
    LayerWrap::~LayerWrap()
    {
    }
    
    void LayerWrap::resetBatchSize(int batchSize)
    {
    }
    
    int LayerWrap::getParameterSize() const
    {
        int size = 0;
        if (mParameters.get() != NULL)
        {
            size = (int)(mParameters->width()*mParameters->height());
        }
        if (NULL!=mNext.get())
        {
            return size + mNext->getParameterSize();
        }
        return size;
    }

    
    void LayerWrap::setParameters(const ALFloatMatrix* p, int offset)
    {
        ALASSERT(NULL!=p);
        ALASSERT(offset <= p->width());
        int currentOffset = 0;
        do
        {
            if (NULL == mParameters.get())
            {
                break;
            }
            ALASSERT(offset>=0);
            auto p_cur = p->vGetAddr()+offset;
            int cur = 0;
            for (int i=0; i<mParameters->height(); ++i)
            {
                auto p_dst = mParameters->vGetAddr(i);
                for (int j=0; j<mParameters->width(); ++j)
                {
                    p_dst[j] = p_cur[cur++];
                }
            }
            currentOffset = (int)(mParameters->width()*mParameters->height());
        } while(0);
        if (NULL != mNext.get())
        {
            mNext->setParameters(p, offset+currentOffset);
        }
    }
    void LayerWrap::readParametersDiff(const ALFloatMatrix* p, int offset)
    {
        ALASSERT(NULL!=p);
        ALASSERT(offset <= p->width());
        int currentOffset = 0;
        do
        {
            if (NULL == mParameters.get())
            {
                break;
            }
            ALASSERT(offset>=0);
            auto p_cur = p->vGetAddr()+offset;
            int cur = 0;
            for (int i=0; i<mParameterDiff->height(); ++i)
            {
                auto p_dst = mParameterDiff->vGetAddr(i);
                for (int j=0; j<mParameterDiff->width(); ++j)
                {
                    p_cur[cur++] = p_dst[j];
                }
            }
            currentOffset = (int)(mParameters->width()*mParameters->height());
        } while(0);
        if (NULL != mNext.get())
        {
            mNext->readParametersDiff(p, offset+currentOffset);
        }
    }

    ALSp<ALFloatMatrix> LayerWrap::forward(ALSp<ALFloatMatrix> input)
    {
        ALASSERT(mLayer->vCheckInput(input.get()));
        mOutput= mLayer->vInitOutput((int)input->height());
        mInput = input;
        mLayer->vForward(input.get(), mOutput.get(), mParameters.get());
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
        ALSp<ALFloatMatrix> inputError = ALFloatMatrix::create(mInput->width(), mInput->height());
        mLayer->vBackward(error.get(), mOutput.get(), mParameters.get(), mInput.get(), inputError.get(), mParameterDiff.get());
        if (NULL!=mBefore)
        {
            mBefore->backward(inputError);
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
}