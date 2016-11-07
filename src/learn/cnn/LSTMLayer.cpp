//
//  LSTMLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 02/11/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "LSTMLayer.hpp"
#include "LayerFactoryRegistor.hpp"
#include <math.h>

namespace ALCNN {
    static auto gSigmod = [](ALFLOAT* dst, ALFLOAT* src, size_t w){
        for (size_t i=0; i<w; ++i)
        {
            dst[i] = 1.0/(1.0+exp(-src[i]));
        }
    };
    static auto gTanh = [](ALFLOAT* dst, ALFLOAT* src, size_t w){
        for (size_t i=0; i<w; ++i)
        {
            dst[i] = tanh(src[i]);
        }
    };
    static auto gSigmodDet = [](ALFLOAT* dst, ALFLOAT* src, size_t w){
        for (size_t i=0; i<w; ++i)
        {
            auto y = src[i];
            dst[i] = y*(1-y);
        }
    };


    static size_t _computeParamterSize(size_t iw, size_t ow)
    {
        auto weight_ih = iw * ow * 4;
        auto weight_ii = ow * 4 * ow;
        auto weight_bias = 4*ow;
        return weight_ih + weight_ii + weight_bias;
    }
    
    struct WeightMatrix
    {
        ALSp<ALFloatMatrix> W;
        ALSp<ALFloatMatrix> U;
        ALSp<ALFloatMatrix> B;
        ALSp<ALFloatMatrix> V;//TODO
        
        WeightMatrix(const ALFloatMatrix* parameters, size_t iw, size_t ow)
        {
            auto p = parameters->vGetAddr();
            W = ALFloatMatrix::createRefMatrix(p, iw, ow*4);
            p+= iw*ow*4;
            U = ALFloatMatrix::createRefMatrix(p, ow*4, ow);
            p+= ow*4*ow;
            B = ALFloatMatrix::createRefMatrix(p, 4*ow, 1);
        }
        ~ WeightMatrix() {}
    };
    
    struct Cache
    {
        struct tCache
        {
            ALSp<ALFloatMatrix> i;
            ALSp<ALFloatMatrix> c;
            ALSp<ALFloatMatrix> f;
            ALSp<ALFloatMatrix> o;
            ALSp<ALFloatMatrix> merge;
        };
        ALSp<ALFloatMatrix> total;
        Cache(const ALFloatMatrix* cache, size_t ow, size_t t):mOw(ow), mT(t), mBatchSize(cache->height())
        {
            ALASSERT(cache->width() == ow*4*t);
            ALASSERT(cache->continues());
            //Reshape
            auto origin = cache->vGetAddr();
            auto w = ow;
            auto h = cache->height()*t;
            auto size = w*h;
            ALASSERT(size*4 == cache->width()*cache->height());
            total = ALFloatMatrix::createRefMatrix(origin, w*4, h);
        }
        ~ Cache(){}
        tCache get(size_t t) const
        {
            ALASSERT(t<mT);
            ALASSERT(t>=0);
            tCache cache;
            cache.merge = ALFloatMatrix::createCropVirtualMatrix(total.get(), 0, t*mBatchSize, 4*mOw-1, (t+1)*mBatchSize-1);
            cache.i = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), 0, 0, mOw-1, mBatchSize-1);
            ALSp<ALFloatMatrix> c_t = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), mOw, 0, mOw*2-1, mBatchSize-1);
            ALSp<ALFloatMatrix> f_t = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), 0, mOw*2, mOw*3-1, mBatchSize-1);
            ALSp<ALFloatMatrix> o_t = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), mOw*3, 0, mOw*4-1, mBatchSize-1);
            return cache;
        }
        size_t mOw;
        size_t mT;
        size_t mBatchSize;
    };
    
    void LSTMLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=parameters);
        ALASSERT(NULL!=cache);
        ALASSERT(cache->width() == getInfo().cw*getInfo().ch);
        ALASSERT(parameters->height() == 1);
        WeightMatrix weight(parameters, before->width(), after->width());
        auto batchSize = before->height();
        Cache cacheMatrix(cache, mOutputSize, mTime);
        
        ALASSERT(batchSize == after->height());
        /*Reshape Before matrix*/
        ALSp<ALFloatMatrix> h_t_1 = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(h_t_1.get());
        ALSp<ALFloatMatrix> c_t_1 = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(c_t_1.get());

        ALSp<ALFloatMatrix> u_h_t_1 = ALFloatMatrix::create(mOutputSize*4, batchSize);
        for (size_t t=0; t<mTime; ++t)
        {
            auto tCache = cacheMatrix.get(t);
            ALSp<ALFloatMatrix> x_t = ALFloatMatrix::createCropVirtualMatrix(before, t*mInputSize, 0, (t+1)*mInputSize-1, batchSize-1);
            ALSp<ALFloatMatrix> h_t = ALFloatMatrix::createCropVirtualMatrix(after, t*mOutputSize, 0, (t+1)*mOutputSize-1, batchSize-1);

            ALFloatMatrix::productT(tCache.merge.get(), x_t.get(), weight.W.get());
            ALFloatMatrix::linearVector(tCache.merge.get(), tCache.merge.get(), 1.0, weight.B.get(), 1.0);
            ALFloatMatrix::productT(u_h_t_1.get(), h_t_1.get(), weight.U.get());
            //Use w_x_t to save total result
            ALFloatMatrix::linear(tCache.merge.get(), tCache.merge.get(), 1.0, u_h_t_1.get(), 1.0);
            
            //Now merge means: Wx+Uh+b, compute others

            ALSp<ALFloatMatrix> i_t = tCache.i;
            ALSp<ALFloatMatrix> c_t = tCache.c;
            ALSp<ALFloatMatrix> f_t = tCache.f;
            ALSp<ALFloatMatrix> o_t = tCache.o;
            
            
            ALFloatMatrix::runLineFunction(i_t.get(), i_t.get(), gSigmod);
            ALFloatMatrix::runLineFunction(f_t.get(), f_t.get(), gSigmod);
            ALFloatMatrix::runLineFunction(o_t.get(), o_t.get(), gSigmod);

            ALFloatMatrix::runLineFunction(c_t.get(), c_t.get(), gTanh);
            ALFloatMatrix::productDot(c_t.get(), c_t.get(), i_t.get());
            ALFloatMatrix::productDot(c_t_1.get(), c_t_1.get(), f_t.get());
            ALFloatMatrix::linear(c_t.get(), c_t.get(), 1.0, c_t_1.get(), 1.0);
            ALFloatMatrix::runLineFunction(h_t.get(), c_t.get(), gTanh);
            
            ALFloatMatrix::productDot(h_t.get(), h_t.get(), o_t.get());
            
            //Copy c_t as c_t_1, h_t as h_t_1
            ALFloatMatrix::copy(c_t_1.get(), c_t.get());
            ALFloatMatrix::copy(h_t_1.get(), h_t.get());
        }
    }
    void LSTMLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        WeightMatrix weight(parameters, mInputSize, mOutputSize);
        ALFloatMatrix::zero(parameters_diff);
        WeightMatrix weightDiff(parameters_diff, mInputSize, mOutputSize);
        Cache cacheMatrix(cache, mOutputSize, mTime);
        auto batchSize = before_diff->height();
        ALASSERT(NULL!=before_diff);
        for (int t=(int)mTime-1; t>=0; --t)
        {
            auto tCache = cacheMatrix.get(t);
            ALSp<ALFloatMatrix> x_diff_t = ALFloatMatrix::createCropVirtualMatrix(before_diff, t*mInputSize, 0, (t+1)*mInputSize-1, batchSize-1);
            ALSp<ALFloatMatrix> h_t = ALFloatMatrix::createCropVirtualMatrix(after, t*mOutputSize, 0, (t+1)*mOutputSize-1, batchSize-1);

        }
    }
    
    //iw and ow is expanded by time
    LSTMLayer::LSTMLayer(size_t iw, size_t ow, size_t t):ILayer(iw, ow, _computeParamterSize(iw/t, ow/t), 1, ow*4, 1)
    {
        ALASSERT(iw%t==0);
        ALASSERT(ow%t==0);
        mTime = t;
        mInputSize = iw/t;
        mOutputSize = ow/t;
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new LSTMLayer(p.uInputSize, p.uOutputSize, p.get("time"));
    };

    static LayerFactoryRegister __reg(gCreateFunction, "LSTM");

};
