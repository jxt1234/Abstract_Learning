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
#include <fstream>
#define DUMP(x) {std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/dump/."#x); ALFloatMatrix::print(x.get(), output);}

namespace ALCNN {
    static auto gAddFunction = [](ALFLOAT* dst, ALFLOAT* src, size_t w) {
        for (size_t i=0; i<w; ++i)
        {
            dst[i] += src[i];
        }
    };
    
    //TODO move these function to math folder
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
    static auto gSigmodDetMulti = [](ALFLOAT* dst, ALFLOAT* src, size_t w){
        for (size_t i=0; i<w; ++i)
        {
            auto y = src[i];
            dst[i] *= y*(1-y);
        }
    };
    static auto gTanhDetMulti = [](ALFLOAT* dst, ALFLOAT* src, size_t w){
        for (size_t i=0; i<w; ++i)
        {
            auto y = src[i];
            dst[i] *= (1+y*y);
        }
    };
    
    static auto gSec2 = [](ALFLOAT* dst, ALFLOAT* src, size_t w){
        for (size_t i=0; i<w; ++i)
        {
            auto y = ::cos(src[i]);
            dst[i] = 1.0/y/y;
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
        
        size_t mIw;
        size_t mOw;
        
        WeightMatrix(const ALFloatMatrix* parameters, size_t iw, size_t ow)
        {
            auto p = parameters->vGetAddr();
            W = ALFloatMatrix::createRefMatrix(p, iw, ow*4);
            p+= iw*ow*4;
            U = ALFloatMatrix::createRefMatrix(p, ow*4, ow);
            p+= ow*4*ow;
            B = ALFloatMatrix::createRefMatrix(p, 4*ow, 1);
            mIw = iw;
            mOw = ow;
        }
        ~ WeightMatrix() {}
        
        void addBias(size_t index, const ALFloatMatrix* x_t, const ALFloatMatrix* h_t_1, const ALFloatMatrix* h_diff)
        {
            ALASSERT(0<=index && index<4);
            ALASSERT(x_t!=NULL);
            ALASSERT(h_diff!=NULL);
            ALSp<ALFloatMatrix> w = ALFloatMatrix::createCropVirtualMatrix(W.get(), 0, mOw*index, mIw-1, mOw*(index+1)-1);
            ALSp<ALFloatMatrix> b = ALFloatMatrix::createCropVirtualMatrix(B.get(), mOw*index, 0, mOw*(index+1)-1, 0);
            ALSp<ALFloatMatrix> u = ALFloatMatrix::createCropVirtualMatrix(U.get(), mOw*index, 0, mOw*(index+1)-1, mOw-1);
            ALSp<ALFloatMatrix> w_diff = ALFloatMatrix::create(mIw, mOw);
            ALFloatMatrix::productTA(w_diff.get(), h_diff, x_t);
            ALFloatMatrix::linear(w.get(), w.get(), 1.0, w_diff.get(), 1.0);
            
            ALFloatMatrix::runReduceFunction(b.get(), h_diff, gAddFunction);
            
            if (NULL != h_t_1)
            {
                ALSp<ALFloatMatrix> u_diff = ALFloatMatrix::create(mOw, mOw);
                ALFloatMatrix::productTA(u_diff.get(), h_diff, h_t_1);
                ALFloatMatrix::linear(u.get(), u.get(), 1.0, u_diff.get(), 1.0);
            }
        }
    };
    
    struct Cache
    {
#define WEIGHTSIZE 4
#define CACHESIZE 5
        struct tCache
        {
            ALSp<ALFloatMatrix> i;
            ALSp<ALFloatMatrix> c;
            ALSp<ALFloatMatrix> f;
            ALSp<ALFloatMatrix> o;
            ALSp<ALFloatMatrix> c_bar;
            ALSp<ALFloatMatrix> merge;
            void dump()
            {
                std::ofstream output1("/Users/jiangxiaotang/Documents/Abstract_Learning/.it");
                ALFloatMatrix::print(i.get(), output1);
                std::ofstream output2("/Users/jiangxiaotang/Documents/Abstract_Learning/.ct");
                ALFloatMatrix::print(c.get(), output2);
                std::ofstream output3("/Users/jiangxiaotang/Documents/Abstract_Learning/.ft");
                ALFloatMatrix::print(f.get(), output3);
                std::ofstream output4("/Users/jiangxiaotang/Documents/Abstract_Learning/.ot");
                ALFloatMatrix::print(o.get(), output4);
                std::ofstream output5("/Users/jiangxiaotang/Documents/Abstract_Learning/.c_bar_t");
                ALFloatMatrix::print(c_bar.get(), output5);
            }
        };
        ALSp<ALFloatMatrix> total;
        Cache(const ALFloatMatrix* cache, size_t ow, size_t t):mOw(ow), mT(t), mBatchSize(cache->height())
        {
            ALASSERT(cache->width() == ow*CACHESIZE*t);
            ALASSERT(cache->continues());
            //Reshape
            auto origin = cache->vGetAddr();
            auto w = ow;
            auto h = cache->height()*t;
            auto size = w*h;
            ALASSERT(size*CACHESIZE == cache->width()*cache->height());
            total = ALFloatMatrix::createRefMatrix(origin, w*CACHESIZE, h);
        }
        ~ Cache(){}
        tCache get(size_t t) const
        {
            ALASSERT(t<mT);
            ALASSERT(t>=0);
            tCache cache;
            cache.merge = ALFloatMatrix::createCropVirtualMatrix(total.get(), 0, t*mBatchSize, WEIGHTSIZE*mOw-1, (t+1)*mBatchSize-1);
            cache.i = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), 0, 0, mOw-1, mBatchSize-1);
            cache.c = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), mOw, 0, mOw*2-1, mBatchSize-1);
            cache.f = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), mOw*2, 0, mOw*3-1, mBatchSize-1);
            cache.o = ALFloatMatrix::createCropVirtualMatrix(cache.merge.get(), mOw*3, 0, mOw*4-1, mBatchSize-1);
            cache.c_bar = ALFloatMatrix::createCropVirtualMatrix(total.get(), WEIGHTSIZE*mOw, t*mBatchSize, mOw*(WEIGHTSIZE+1)-1, (t+1)*mBatchSize-1);
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
        ALASSERT(before->width() == mInputSize*mTime);
        ALASSERT(after->width() == mOutputSize*mTime);
        ALASSERT(before->height() == after->height());
        WeightMatrix weight(parameters, mInputSize, mOutputSize);
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
            ALFloatMatrix::product(u_h_t_1.get(), h_t_1.get(), weight.U.get());
            //Use w_x_t to save total result
            ALFloatMatrix::linear(tCache.merge.get(), tCache.merge.get(), 1.0, u_h_t_1.get(), 1.0);
            
            //Now merge means: Wx+Uh+b, compute others
            
            ALSp<ALFloatMatrix> i_t = tCache.i;
            ALSp<ALFloatMatrix> c_t = tCache.c;
            ALSp<ALFloatMatrix> f_t = tCache.f;
            ALSp<ALFloatMatrix> o_t = tCache.o;
            ALSp<ALFloatMatrix> c_t_bar = tCache.c_bar;
            
            
            
            ALFloatMatrix::runLineFunction(i_t.get(), i_t.get(), gSigmod);
            ALFloatMatrix::runLineFunction(f_t.get(), f_t.get(), gSigmod);
            ALFloatMatrix::runLineFunction(o_t.get(), o_t.get(), gSigmod);
            
            ALFloatMatrix::runLineFunction(c_t_bar.get(), c_t.get(), gTanh);
            ALFloatMatrix::productDot(c_t.get(), c_t_bar.get(), i_t.get());
            ALFloatMatrix::productDot(c_t_1.get(), c_t_1.get(), f_t.get());
            ALFloatMatrix::linear(c_t.get(), c_t.get(), 1.0, c_t_1.get(), 1.0);
            ALFloatMatrix::runLineFunction(h_t.get(), c_t.get(), gTanh);
            
            ALFloatMatrix::productDot(h_t.get(), h_t.get(), o_t.get());
            
            
            //Copy c_t as c_t_1, h_t as h_t_1
            ALFloatMatrix::copy(c_t_1.get(), c_t.get());
            ALFloatMatrix::copy(h_t_1.get(), h_t.get());
            
        }
        
        before = NULL;
        
    }
    void LSTMLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff_, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        WeightMatrix weight(parameters, mInputSize, mOutputSize);
        ALFloatMatrix::zero(parameters_diff);
        WeightMatrix weightDiff(parameters_diff, mInputSize, mOutputSize);
        Cache cacheMatrix(cache, mOutputSize, mTime);
        auto batchSize = before->height();
        size_t index;
        index = 0;
        ALSp<ALFloatMatrix> w_i = ALFloatMatrix::createCropVirtualMatrix(weight.W.get(), 0, mOutputSize*index, mInputSize-1, mOutputSize*(index+1)-1);
        index = 1;
        ALSp<ALFloatMatrix> w_c = ALFloatMatrix::createCropVirtualMatrix(weight.W.get(), 0, mOutputSize*index, mInputSize-1, mOutputSize*(index+1)-1);
        index = 2;
        ALSp<ALFloatMatrix> w_f = ALFloatMatrix::createCropVirtualMatrix(weight.W.get(), 0, mOutputSize*index, mInputSize-1, mOutputSize*(index+1)-1);
        index = 3;
        ALSp<ALFloatMatrix> w_o = ALFloatMatrix::createCropVirtualMatrix(weight.W.get(), 0, mOutputSize*index, mInputSize-1, mOutputSize*(index+1)-1);
        index = 0;
        ALSp<ALFloatMatrix> u_i = ALFloatMatrix::createCropVirtualMatrix(weight.U.get(), mOutputSize*index, 0, mOutputSize*(index+1)-1, mOutputSize-1);
        index = 1;
        ALSp<ALFloatMatrix> u_c = ALFloatMatrix::createCropVirtualMatrix(weight.U.get(), mOutputSize*index, 0, mOutputSize*(index+1)-1, mOutputSize-1);
        index = 2;
        ALSp<ALFloatMatrix> u_f = ALFloatMatrix::createCropVirtualMatrix(weight.U.get(), mOutputSize*index, 0, mOutputSize*(index+1)-1, mOutputSize-1);
        index = 3;
        ALSp<ALFloatMatrix> u_o = ALFloatMatrix::createCropVirtualMatrix(weight.U.get(), mOutputSize*index, 0, mOutputSize*(index+1)-1, mOutputSize-1);
        
        //Init Cache
        ALSp<ALFloatMatrix> ot_sec_2_c_t = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> tanh_c_t_o_t_det = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> w_u_b_i = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> w_u_b_c = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> w_u_b_f = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> w_u_b_o = ALFloatMatrix::create(mOutputSize, batchSize);
        
        ALSp<ALFloatMatrix> x_cache = ALFloatMatrix::create(mInputSize, batchSize);
        ALSp<ALFloatMatrix> h_cache = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> h_diff_t = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(h_diff_t.get());
        
        for (int t=(int)mTime-1; t>=0; --t)
        {
            auto tCache = cacheMatrix.get(t);
            ALSp<ALFloatMatrix> x_t = ALFloatMatrix::createCropVirtualMatrix(before, t*mInputSize, 0, (t+1)*mInputSize-1, batchSize-1);
            ALSp<ALFloatMatrix> h_t = ALFloatMatrix::createCropVirtualMatrix(after, t*mOutputSize, 0, (t+1)*mOutputSize-1, batchSize-1);
            ALSp<ALFloatMatrix> h_t_1;
            if (t>0)
            {
                h_t_1 = ALFloatMatrix::createCropVirtualMatrix(after, (t-1)*mOutputSize, 0, (t)*mOutputSize-1, batchSize-1);
            }
            ALSp<ALFloatMatrix> h_diff_t_outside = ALFloatMatrix::createCropVirtualMatrix(after_diff, t*mOutputSize, 0, (t+1)*mOutputSize-1, batchSize-1);
            ALFloatMatrix::linear(h_diff_t.get(), h_diff_t.get(), 1.0, h_diff_t_outside.get(), 1.0);
            
            //Compute ot_sec_2_c_t * h_diff_t
            ALFloatMatrix::runLineFunction(ot_sec_2_c_t.get(), tCache.c.get(), gSec2);
            ALFloatMatrix::productDot(ot_sec_2_c_t.get(), ot_sec_2_c_t.get(), tCache.o.get());
            ALFloatMatrix::productDot(ot_sec_2_c_t.get(), ot_sec_2_c_t.get(), h_diff_t.get());
            
            //Compute tanh_c_t_o_t_det * h_diff_t
            ALFloatMatrix::runLineFunction(tanh_c_t_o_t_det.get(), tCache.c.get(), gTanh);
            ALFloatMatrix::productDot(tanh_c_t_o_t_det.get(), tCache.o.get(), tanh_c_t_o_t_det.get());
            ALFloatMatrix::productDot(tanh_c_t_o_t_det.get(), tanh_c_t_o_t_det.get(), h_diff_t.get());
            
            //Compute w_u_b_o: tanh_c_t_o_t_det * o_t * (1-o_t)
            ALFloatMatrix::copy(w_u_b_o.get(), tanh_c_t_o_t_det.get());
            ALFloatMatrix::runLineFunction(w_u_b_o.get(), tCache.o.get(), gSigmodDetMulti);
            
            //Compute w_u_b_i: ot_sec_2_c_t * c_bar_t * i_t* (1-i_t)
            ALFloatMatrix::productDot(w_u_b_i.get(), ot_sec_2_c_t.get(), tCache.c_bar.get());
            ALFloatMatrix::runLineFunction(w_u_b_i.get(), tCache.i.get(), gSigmodDetMulti);
            
            //Compute w_u_b_c: ot_sec_2_c_t * i_t * (1+c_bar_t*c_bar_t)
            ALFloatMatrix::productDot(w_u_b_c.get(), ot_sec_2_c_t.get(), tCache.i.get());
            ALFloatMatrix::runLineFunction(w_u_b_c.get(), tCache.c_bar.get(), gTanhDetMulti);
            
            
            //Compute w_u_b_f: ot_sec_2_c_t * c_t_1 * f_t * (1-f_t)
            if (0 == t)
            {
                ALFloatMatrix::zero(w_u_b_f.get());
            }
            else
            {
                auto tCache_1 = cacheMatrix.get(t-1);
                ALSp<ALFloatMatrix> c_t_1 = tCache_1.c;
                ALFloatMatrix::productDot(w_u_b_f.get(), ot_sec_2_c_t.get(), c_t_1.get());
                ALFloatMatrix::runLineFunction(w_u_b_f.get(), tCache.f.get(), gSigmodDetMulti);
            }
            
            
            if (NULL!=before_diff_)
            {
                //Compute x_diff_t
                ALSp<ALFloatMatrix> x_diff_t = ALFloatMatrix::createCropVirtualMatrix(before_diff_, t*mInputSize, 0, (t+1)*mInputSize-1, batchSize-1);
                ALFloatMatrix::product(x_diff_t.get(), w_u_b_o.get(), w_o.get());
                ALFloatMatrix::product(x_cache.get(), w_u_b_i.get(), w_i.get());
                ALFloatMatrix::linear(x_diff_t.get(), x_diff_t.get(), 1.0, x_cache.get(), 1.0);
                ALFloatMatrix::product(x_cache.get(), w_u_b_c.get(), w_c.get());
                ALFloatMatrix::linear(x_diff_t.get(), x_diff_t.get(), 1.0, x_cache.get(), 1.0);
                ALFloatMatrix::product(x_cache.get(), w_u_b_f.get(), w_f.get());
                ALFloatMatrix::linear(x_diff_t.get(), x_diff_t.get(), 1.0, x_cache.get(), 1.0);
            }
            //Compute w_u_b_o_diff
            weightDiff.addBias(3, x_t.get(), h_t_1.get(), w_u_b_o.get());
            
            //Compute part of w_u_b_i_diff, w_u_b_c_diff, w_u_b_f_diff
            weightDiff.addBias(0, x_t.get(), h_t_1.get(), w_u_b_i.get());
            weightDiff.addBias(1, x_t.get(), h_t_1.get(), w_u_b_c.get());
            weightDiff.addBias(2, x_t.get(), h_t_1.get(), w_u_b_f.get());
            
            if (false)
            {
                DUMP(tCache.i);
                DUMP(tCache.c);
                DUMP(tCache.f);
                DUMP(tCache.o);
                DUMP(tCache.c_bar);
                DUMP(h_t);
                DUMP(h_diff_t);
                DUMP(x_t);
                DUMP(weight.W);
                DUMP(ot_sec_2_c_t);
                DUMP(tanh_c_t_o_t_det);
            }
            
            //Compute h_diff_t
            if (t > 0)
            {
                ALFloatMatrix::product(h_diff_t.get(), w_u_b_o.get(), u_o.get());
                ALFloatMatrix::product(h_cache.get(), w_u_b_i.get(), u_i.get());
                ALFloatMatrix::linear(h_diff_t.get(), h_diff_t.get(), 1.0, h_cache.get(), 1.0);
                ALFloatMatrix::product(h_cache.get(), w_u_b_c.get(), u_c.get());
                ALFloatMatrix::linear(h_diff_t.get(), h_diff_t.get(), 1.0, h_cache.get(), 1.0);
                ALFloatMatrix::product(h_cache.get(), w_u_b_f.get(), u_f.get());
                ALFloatMatrix::linear(h_diff_t.get(), h_diff_t.get(), 1.0, h_cache.get(), 1.0);
            }
        }
    }
    
    //iw and ow is expanded by time
    LSTMLayer::LSTMLayer(size_t iw, size_t ow, size_t t):ILayer(iw, ow, _computeParamterSize(iw/t, ow/t), 1, ow*CACHESIZE, 1)
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
