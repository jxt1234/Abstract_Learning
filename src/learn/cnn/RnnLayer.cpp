//
//  RnnLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 14/11/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "RnnLayer.hpp"
#include "LayerFactoryRegistor.hpp"
#include <math.h>
#include <fstream>


namespace ALCNN {
    static auto gMaskFunction = [](ALFLOAT* dst, ALFLOAT* src, ALFLOAT* mask, size_t w) {
        for (size_t i=0; i<w; ++i)
        {
            dst[i] = dst[i]*mask[i] + (1.0-mask[i])*src[i];
        }
    };
    static auto gAddFunction = [](ALFLOAT* dst, ALFLOAT* src, size_t w) {
        for (size_t i=0; i<w; ++i)
        {
            dst[i] += src[i];
        }
    };

    static void _setMaskMatrix(ALFloatMatrix* mask, const ALFloatMatrix* time, size_t t)
    {
        ALASSERT(NULL != time);
        ALASSERT(NULL != mask);
        ALASSERT(time->height()==1);
        ALASSERT(time->width() == mask->height());
        auto size = time->width();
        auto _t = time->vGetAddr();
        auto w = mask->width();
        for (size_t i=0; i<size; ++i)
        {
            auto dst = mask->vGetAddr(i);
            if (t < _t[i])
            {
                for (size_t j=0; j<w; ++j)
                {
                    dst[j] = 1.0;
                }
            }
            else
            {
                ::memset(dst, 0, sizeof(ALFLOAT)*w);
            }
        }
    }
    
    struct RNNWeightMatrix
    {
        ALSp<ALFloatMatrix> W;
        ALSp<ALFloatMatrix> U;
        ALSp<ALFloatMatrix> B;
        
        size_t mIw;
        size_t mOw;
        
        static size_t computeSize(size_t iw, size_t ow)
        {
            auto weight_ih = iw*ow;
            auto weight_ii = ow*ow;
            auto weight_bias = ow;
            return weight_ih + weight_ii + weight_bias;
        }
        
        RNNWeightMatrix(const ALFloatMatrix* parameters, size_t iw, size_t ow)
        {
            auto p = parameters->vGetAddr();
            W = ALFloatMatrix::createRefMatrix(p, iw, ow);
            p+= iw*ow;
            U = ALFloatMatrix::createRefMatrix(p, ow, ow);
            p+= ow*ow;
            B = ALFloatMatrix::createRefMatrix(p, ow, 1);
            mIw = iw;
            mOw = ow;
            ALASSERT(parameters->width() == computeSize(iw, ow));
        }
        ~ RNNWeightMatrix() {}
        void addBias(const ALFloatMatrix* x_t, const ALFloatMatrix* h_t_1, const ALFloatMatrix* w_u_b)
        {
            //return;
            ALASSERT(x_t!=NULL);
            ALASSERT(w_u_b!=NULL);
            ALSp<ALFloatMatrix> w_diff = ALFloatMatrix::create(mIw, mOw);
            ALFloatMatrix::productTA(w_diff.get(), w_u_b, x_t);
            ALFloatMatrix::linear(W.get(), W.get(), 1.0, w_diff.get(), 1.0);
            
            ALFloatMatrix::runReduceFunction(B.get(), w_u_b, gAddFunction);
            
            if (NULL != h_t_1)
            {
                ALSp<ALFloatMatrix> u_diff = ALFloatMatrix::create(mOw, mOw);
                ALFloatMatrix::productTA(u_diff.get(), w_u_b, h_t_1);
                ALFloatMatrix::linear(U.get(), U.get(), 1.0, u_diff.get(), 1.0);
            }
        }

    };

    void RNNLayer::vForward(const ALFloatMatrix* _before, ALFloatMatrix* _after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        auto batchSize = _after->height();
        RNNWeightMatrix weight(parameters, mInputSize, mOutputSize);
        ALSp<ALFloatMatrix> time = ALFloatMatrix::createCropVirtualMatrix(_before, 0, 0, 0, _before->height()-1);
        ALSp<ALFloatMatrix> targetTime = ALFloatMatrix::createCropVirtualMatrix(_after, 0, 0, 0, _after->height()-1);
        ALFloatMatrix::copy(targetTime.get(), time.get());
        
        time = ALFloatMatrix::transpose(time.get());
        ALSp<ALFloatMatrix> mask = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> _beforeWrap = ALFloatMatrix::createCropVirtualMatrix(_before, 1, 0, _before->width()-1, _before->height()-1);
        ALSp<ALFloatMatrix> _afterWrap = ALFloatMatrix::createCropVirtualMatrix(_after, 1, 0, _after->width()-1, _after->height()-1);
        auto before = _beforeWrap.get();
        auto after = _afterWrap.get();

        ALSp<ALFloatMatrix> h_t_1 = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(h_t_1.get());
        ALSp<ALFloatMatrix> c_t_1 = ALFloatMatrix::create(mOutputSize, batchSize);

        for (size_t t=0; t<mTime; ++t)
        {
            _setMaskMatrix(mask.get(), time.get(), t);
            ALSp<ALFloatMatrix> x_t = ALFloatMatrix::createCropVirtualMatrix(before, t*mInputSize, 0, (t+1)*mInputSize-1, batchSize-1);
            ALSp<ALFloatMatrix> h_t = ALFloatMatrix::createCropVirtualMatrix(after, t*mOutputSize, 0, (t+1)*mOutputSize-1, batchSize-1);
            if (t > 0)
            {
                h_t_1 = ALFloatMatrix::createCropVirtualMatrix(after, (t-1)*mOutputSize, 0, (t)*mOutputSize-1, batchSize-1);
            }
            
            ALFloatMatrix::productT(h_t.get(), x_t.get(), weight.W.get());
            
            ALFloatMatrix::linearVector(h_t.get(), h_t.get(), 1.0, weight.B.get(), 1.0);
            ALFloatMatrix::product(c_t_1.get(), h_t_1.get(), weight.U.get());
            ALFloatMatrix::linear(h_t.get(), h_t.get(), 1.0, c_t_1.get(), 1.0);
            ALFloatMatrix::runLineFunctionBi(h_t.get(), h_t_1.get(), mask.get(), gMaskFunction);
        }
    }
    void RNNLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        RNNWeightMatrix weight(parameters, mInputSize, mOutputSize);
        ALFloatMatrix::zero(parameters_diff);
        RNNWeightMatrix weightDiff(parameters_diff, mInputSize, mOutputSize);
        auto batchSize = before->height();
        //Init Cache
        ALSp<ALFloatMatrix> w_u_b = ALFloatMatrix::create(mOutputSize, batchSize);
        
        ALSp<ALFloatMatrix> h_diff_t = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(h_diff_t.get());
        
        ALSp<ALFloatMatrix> time = ALFloatMatrix::createCropVirtualMatrix(before, 0, 0, 0, before->height()-1);
        time = ALFloatMatrix::transpose(time.get());
        ALSp<ALFloatMatrix> mask = ALFloatMatrix::create(mOutputSize, batchSize);
        
        for (int t=(int)mTime-1; t>=0; --t)
        {
            _setMaskMatrix(mask.get(), time.get(), t);
            ALSp<ALFloatMatrix> x_t = ALFloatMatrix::createCropVirtualMatrix(before, t*mInputSize+1, 0, (t+1)*mInputSize, batchSize-1);
            ALSp<ALFloatMatrix> h_t = ALFloatMatrix::createCropVirtualMatrix(after, t*mOutputSize+1, 0, (t+1)*mOutputSize, batchSize-1);
            ALSp<ALFloatMatrix> h_diff_t_outside = ALFloatMatrix::createCropVirtualMatrix(after_diff, t*mOutputSize+1, 0, (t+1)*mOutputSize, batchSize-1);
            ALFloatMatrix::linear(w_u_b.get(), h_diff_t.get(), 1.0, h_diff_t_outside.get(), 1.0);

            if (NULL!=before_diff)
            {
                //Compute x_diff_t
                ALSp<ALFloatMatrix> x_diff_t = ALFloatMatrix::createCropVirtualMatrix(before_diff, t*mInputSize+1, 0, (t+1)*mInputSize, batchSize-1);
                ALFloatMatrix::product(x_diff_t.get(), w_u_b.get(), weight.W.get());
            }
            //Compute weight bias
            ALSp<ALFloatMatrix> h_t_1;
            if (t>0)
            {
                h_t_1 = ALFloatMatrix::createCropVirtualMatrix(after, (t-1)*mOutputSize+1, 0, (t)*mOutputSize, batchSize-1);
            }
            weightDiff.addBias(x_t.get(), h_t_1.get(), w_u_b.get());
            
            //Compute h_diff_t and c_diff_t_1
            if (t > 0)
            {
                ALFloatMatrix::productT(h_diff_t.get(), w_u_b.get(), weight.U.get());
                ALFloatMatrix::productDot(h_diff_t.get(), h_diff_t.get(), mask.get());
            }
        }
    }
    
    //iw and ow is expanded by time
    RNNLayer::RNNLayer(size_t iw, size_t ow, size_t t):ILayer(iw+1, ow+1, RNNWeightMatrix::computeSize(iw/t, ow/t), 1, 0, 0)
    {
        ALASSERT(iw%t==0);
        ALASSERT(ow%t==0);
        mTime = t;
        mInputSize = iw/t;
        mOutputSize = ow/t;
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new RNNLayer(p.uInputSize, p.uOutputSize, p.get("time"));
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "RNN");
};
