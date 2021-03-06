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
static int gNumber = 0;
#define DUMP(x) {std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/dump/."#x); ALFloatMatrix::print(x.get(), output);}
#define DUMP2(x) {std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/dump/."#x); ALFloatMatrix::print(x, output);}


namespace ALCNN {
    static auto gAddFunction = [](ALFLOAT* dst, ALFLOAT* src, size_t w) {
        for (size_t i=0; i<w; ++i)
        {
            dst[i] += src[i];
        }
    };
    
    static auto gMaskFunction = [](ALFLOAT* dst, ALFLOAT* src, ALFLOAT* mask, size_t w) {
        for (size_t i=0; i<w; ++i)
        {
            dst[i] = dst[i]*mask[i] + (1.0-mask[i])*src[i];
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
        
        void addBias(const ALFloatMatrix* x_t, const ALFloatMatrix* h_t_1, const ALFloatMatrix* w_u_b)
        {
            //return;
            ALASSERT(x_t!=NULL);
            ALASSERT(w_u_b!=NULL);
            ALSp<ALFloatMatrix> w_diff = ALFloatMatrix::create(mIw, mOw*4);
            ALFloatMatrix::productTA(w_diff.get(), w_u_b, x_t);
            ALFloatMatrix::linear(W.get(), W.get(), 1.0, w_diff.get(), 1.0);
            
            ALFloatMatrix::runReduceFunction(B.get(), w_u_b, gAddFunction);
            
            if (NULL != h_t_1)
            {
                ALSp<ALFloatMatrix> u_diff = ALFloatMatrix::create(mOw*4, mOw);
                for (int i=0; i<4; ++i)
                {
                    ALSp<ALFloatMatrix> sub_u_diff = ALFloatMatrix::createCropVirtualMatrix(u_diff.get(), i*mOw, 0, mOw*(i+1)-1, mOw-1);
                    ALSp<ALFloatMatrix> sub_w_u_b = ALFloatMatrix::createCropVirtualMatrix(w_u_b, i*mOw, 0, mOw*(i+1)-1, w_u_b->height()-1);
                    ALFloatMatrix::productTA(sub_u_diff.get(), sub_w_u_b.get(), h_t_1);
                }
                ALFloatMatrix::linear(U.get(), U.get(), 1.0, u_diff.get(), 1.0);
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
    
    void LSTMLayer::vForward(const ALFloatMatrix* _before, ALFloatMatrix* _after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        gNumber++;
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=parameters);
        ALASSERT(NULL!=cache);
        ALASSERT(cache->width() == getInfo().cw*getInfo().ch);
        ALASSERT(parameters->height() == 1);
        ALASSERT(_before->width() == mInputSize*mTime+1);
        ALASSERT(_after->width() == mOutputSize*mTime+1);
        ALASSERT(_before->height() == _after->height());
        WeightMatrix weight(parameters, mInputSize, mOutputSize);
        auto batchSize = _before->height();
        Cache cacheMatrix(cache, mOutputSize, mTime);
        
        ALASSERT(batchSize == _after->height());
        
        ALSp<ALFloatMatrix> _beforeWrap = ALFloatMatrix::createCropVirtualMatrix(_before, 1, 0, _before->width()-1, _before->height()-1);
        ALSp<ALFloatMatrix> _afterWrap = ALFloatMatrix::createCropVirtualMatrix(_after, 1, 0, _after->width()-1, _after->height()-1);
        auto before = _beforeWrap.get();
        auto after = _afterWrap.get();
        
        /*Reshape Before matrix*/
        ALSp<ALFloatMatrix> h_t_1 = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(h_t_1.get());
        ALSp<ALFloatMatrix> c_t_1 = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(c_t_1.get());
        
        ALSp<ALFloatMatrix> u_h_t_1 = ALFloatMatrix::create(mOutputSize*4, batchSize);
        
        ALSp<ALFloatMatrix> time = ALFloatMatrix::createCropVirtualMatrix(_before, 0, 0, 0, _before->height()-1);
        ALSp<ALFloatMatrix> targetTime = ALFloatMatrix::createCropVirtualMatrix(_after, 0, 0, 0, _after->height()-1);
        ALFloatMatrix::copy(targetTime.get(), time.get());
        
        time = ALFloatMatrix::transpose(time.get());
        ALSp<ALFloatMatrix> mask = ALFloatMatrix::create(mOutputSize, batchSize);
        
        for (size_t t=0; t<mTime; ++t)
        {
            _setMaskMatrix(mask.get(), time.get(), t);
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
            ALFloatMatrix::productDot(c_t.get(), c_t_bar.get(), i_t.get(), false);
            ALFloatMatrix::productDot(c_t.get(), c_t_1.get(), f_t.get(), true);
            ALFloatMatrix::runLineFunctionBi(c_t.get(), c_t_1.get(), mask.get(), gMaskFunction);
            
            ALFloatMatrix::runLineFunction(h_t.get(), c_t.get(), gTanh);
            
            ALFloatMatrix::productDot(h_t.get(), h_t.get(), o_t.get());
            ALFloatMatrix::runLineFunctionBi(h_t.get(), h_t_1.get(), mask.get(), gMaskFunction);
            
            
            //Copy c_t as c_t_1, h_t as h_t_1
            ALFloatMatrix::copy(c_t_1.get(), c_t.get());
            ALFloatMatrix::copy(h_t_1.get(), h_t.get());
            if (false)
            {
                if (gNumber % 100 == 99)
                {
                    DUMP(tCache.i);
                    DUMP(tCache.c);
                    DUMP(tCache.f);
                    DUMP(tCache.o);
                    DUMP(tCache.c_bar);
                    DUMP(h_t);
                    DUMP(x_t);
                    DUMP(weight.W);
                }
            }

        }

        before = NULL;
        
    }
    void LSTMLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        WeightMatrix weight(parameters, mInputSize, mOutputSize);
        ALFloatMatrix::zero(parameters_diff);
        WeightMatrix weightDiff(parameters_diff, mInputSize, mOutputSize);
        Cache cacheMatrix(cache, mOutputSize, mTime);
        auto batchSize = before->height();
        size_t index;
        //Init Cache
        ALSp<ALFloatMatrix> ot_sec_2_c_t = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> tanh_c_t_o_t_det = ALFloatMatrix::create(mOutputSize, batchSize);
        ALSp<ALFloatMatrix> w_u_b = ALFloatMatrix::create(mOutputSize*4, batchSize);
        index = 0;
        ALSp<ALFloatMatrix> w_u_b_i = ALFloatMatrix::createCropVirtualMatrix(w_u_b.get(), index*mOutputSize, 0, mOutputSize*(index+1)-1, batchSize-1);
        index = 1;
        ALSp<ALFloatMatrix> w_u_b_c = ALFloatMatrix::createCropVirtualMatrix(w_u_b.get(), index*mOutputSize, 0, mOutputSize*(index+1)-1, batchSize-1);
        index = 2;
        ALSp<ALFloatMatrix> w_u_b_f = ALFloatMatrix::createCropVirtualMatrix(w_u_b.get(), index*mOutputSize, 0, mOutputSize*(index+1)-1, batchSize-1);
        index = 3;
        ALSp<ALFloatMatrix> w_u_b_o = ALFloatMatrix::createCropVirtualMatrix(w_u_b.get(), index*mOutputSize, 0, mOutputSize*(index+1)-1, batchSize-1);
        
        ALSp<ALFloatMatrix> h_diff_t = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(h_diff_t.get());
        ALSp<ALFloatMatrix> c_diff_t = ALFloatMatrix::create(mOutputSize, batchSize);
        ALFloatMatrix::zero(c_diff_t.get());
        
        ALSp<ALFloatMatrix> time = ALFloatMatrix::createCropVirtualMatrix(before, 0, 0, 0, before->height()-1);
        time = ALFloatMatrix::transpose(time.get());
        ALSp<ALFloatMatrix> mask = ALFloatMatrix::create(mOutputSize, batchSize);

        for (int t=(int)mTime-1; t>=0; --t)
        {
            _setMaskMatrix(mask.get(), time.get(), t);
            auto tCache = cacheMatrix.get(t);
            ALSp<ALFloatMatrix> x_t = ALFloatMatrix::createCropVirtualMatrix(before, t*mInputSize+1, 0, (t+1)*mInputSize, batchSize-1);
            ALSp<ALFloatMatrix> h_t = ALFloatMatrix::createCropVirtualMatrix(after, t*mOutputSize+1, 0, (t+1)*mOutputSize, batchSize-1);
            ALSp<ALFloatMatrix> h_t_1;
            if (t>0)
            {
                h_t_1 = ALFloatMatrix::createCropVirtualMatrix(after, (t-1)*mOutputSize+1, 0, (t)*mOutputSize, batchSize-1);
            }
            ALSp<ALFloatMatrix> h_diff_t_outside = ALFloatMatrix::createCropVirtualMatrix(after_diff, t*mOutputSize+1, 0, (t+1)*mOutputSize, batchSize-1);
            ALFloatMatrix::linear(h_diff_t.get(), h_diff_t.get(), 1.0, h_diff_t_outside.get(), 1.0);
            
            //Compute ot_sec_2_c_t * h_diff_t
            ALFloatMatrix::runLineFunction(ot_sec_2_c_t.get(), tCache.c.get(), gSec2);
            ALFloatMatrix::productDot(ot_sec_2_c_t.get(), ot_sec_2_c_t.get(), tCache.o.get());
            ALFloatMatrix::productDot(ot_sec_2_c_t.get(), ot_sec_2_c_t.get(), h_diff_t.get());
            
            ALFloatMatrix::linear(c_diff_t.get(), c_diff_t.get(), 1.0f, ot_sec_2_c_t.get(), 1.0);
            
            //Compute tanh_c_t_o_t_det * h_diff_t
            ALFloatMatrix::runLineFunction(tanh_c_t_o_t_det.get(), tCache.c.get(), gTanh);
            ALFloatMatrix::productDot(tanh_c_t_o_t_det.get(), tCache.o.get(), tanh_c_t_o_t_det.get());
            ALFloatMatrix::productDot(tanh_c_t_o_t_det.get(), tanh_c_t_o_t_det.get(), h_diff_t.get());
            
            //Compute w_u_b_o: tanh_c_t_o_t_det * o_t * (1-o_t)
            ALFloatMatrix::copy(w_u_b_o.get(), tanh_c_t_o_t_det.get());
            ALFloatMatrix::runLineFunction(w_u_b_o.get(), tCache.o.get(), gSigmodDetMulti);
            
            //Compute w_u_b_i: ot_sec_2_c_t * c_bar_t * i_t* (1-i_t)
            ALFloatMatrix::productDot(w_u_b_i.get(), c_diff_t.get(), tCache.c_bar.get());
            ALFloatMatrix::runLineFunction(w_u_b_i.get(), tCache.i.get(), gSigmodDetMulti);
            
            //Compute w_u_b_c: ot_sec_2_c_t * i_t * (1+c_bar_t*c_bar_t) * mask
            ALFloatMatrix::productDot(w_u_b_c.get(), c_diff_t.get(), tCache.i.get());
            ALFloatMatrix::runLineFunction(w_u_b_c.get(), tCache.c_bar.get(), gTanhDetMulti);
            ALFloatMatrix::productDot(w_u_b_c.get(), w_u_b_c.get(), mask.get());
            
            //Compute w_u_b_f: ot_sec_2_c_t * c_t_1 * f_t * (1-f_t)
            if (0 == t)
            {
                ALFloatMatrix::zero(w_u_b_f.get());
            }
            else
            {
                auto tCache_1 = cacheMatrix.get(t-1);
                ALSp<ALFloatMatrix> c_t_1 = tCache_1.c;
                ALFloatMatrix::productDot(w_u_b_f.get(), c_diff_t.get(), c_t_1.get());
                ALFloatMatrix::runLineFunction(w_u_b_f.get(), tCache.f.get(), gSigmodDetMulti);
            }
            
            if (NULL!=before_diff)
            {
                //Compute x_diff_t
                ALSp<ALFloatMatrix> x_diff_t = ALFloatMatrix::createCropVirtualMatrix(before_diff, t*mInputSize+1, 0, (t+1)*mInputSize, batchSize-1);
                ALFloatMatrix::product(x_diff_t.get(), w_u_b.get(), weight.W.get());
            }
            //Compute weight bias
            weightDiff.addBias(x_t.get(), h_t_1.get(), w_u_b.get());

            //Compute h_diff_t and c_diff_t_1
            if (t > 0)
            {
                ALFloatMatrix::productT(h_diff_t.get(), w_u_b.get(), weight.U.get());
                
                ALFloatMatrix::productDot(h_diff_t.get(), h_diff_t.get(), mask.get());
                
                ALFloatMatrix::productDot(c_diff_t.get(), c_diff_t.get(), tCache.f.get());
                ALFloatMatrix::productDot(c_diff_t.get(), c_diff_t.get(), mask.get());
            }
        }
    }
    
    //iw and ow is expanded by time
    LSTMLayer::LSTMLayer(size_t iw, size_t ow, size_t t):ILayer(iw+1, ow+1, _computeParamterSize(iw/t, ow/t), 1, ow*CACHESIZE, 1)
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
