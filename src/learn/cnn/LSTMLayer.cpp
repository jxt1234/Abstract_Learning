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
    
    void LSTMLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        ALASSERT(NULL!=parameters);
        ALASSERT(parameters->height() == 1);
        WeightMatrix weight(parameters, before->width(), after->width());
        auto batchSize = before->height();
        ALASSERT(batchSize == after->height());
        /*Reshape Before matrix*/
        ALSp<ALFloatMatrix> h_t_1 = ALFloatMatrix::create(getInfo().ow, batchSize);
        ALFloatMatrix::zero(h_t_1.get());
        ALSp<ALFloatMatrix> c_t_1 = ALFloatMatrix::create(getInfo().ow, batchSize);
        ALFloatMatrix::zero(c_t_1.get());

        ALSp<ALFloatMatrix> u_h_t_1 = ALFloatMatrix::create(getInfo().ow*4, batchSize);
        ALSp<ALFloatMatrix> w_x_t = ALFloatMatrix::create(getInfo().ow*4, batchSize);
        for (size_t t=0; t<mTime; ++t)
        {
            //TODO Assert the before Matrix is continues
            ALSp<ALFloatMatrix> x_t = ALFloatMatrix::createCropVirtualMatrix(before, t*getInfo().iw, 0, (t+1)*getInfo().iw-1, batchSize-1);
            ALFloatMatrix::productT(w_x_t.get(), x_t.get(), weight.W.get());
            ALFloatMatrix::linearVector(w_x_t.get(), w_x_t.get(), 1.0, weight.B.get(), 1.0);
            ALFloatMatrix::productT(u_h_t_1.get(), h_t_1.get(), weight.U.get());
            //Use w_x_t to save total result
            ALFloatMatrix::linear(w_x_t.get(), w_x_t.get(), 1.0, u_h_t_1.get(), 1.0);
            
            //Now w_x_t means: Wx+Uh+b, compute others
            
            ALSp<ALFloatMatrix> i_t = ALFloatMatrix::createCropVirtualMatrix(w_x_t.get(), 0, 0, getInfo().ow-1, batchSize-1);
            ALFloatMatrix::runLineFunction(i_t.get(), i_t.get(), gSigmod);
            
            ALSp<ALFloatMatrix> c_t = ALFloatMatrix::createCropVirtualMatrix(w_x_t.get(), getInfo().ow, 0, getInfo().ow*2-1, batchSize-1);
            ALFloatMatrix::runLineFunction(c_t.get(), c_t.get(), gTanh);
            
            ALSp<ALFloatMatrix> f_t = ALFloatMatrix::createCropVirtualMatrix(w_x_t.get(), 0, getInfo().ow*2, getInfo().ow*3-1, batchSize-1);
            ALFloatMatrix::runLineFunction(f_t.get(), f_t.get(), gSigmod);
            
            ALFloatMatrix::productDot(c_t.get(), c_t.get(), i_t.get());
            ALFloatMatrix::productDot(c_t_1.get(), c_t_1.get(), f_t.get());
            
            ALFloatMatrix::linear(c_t.get(), c_t.get(), 1.0, c_t_1.get(), 1.0);
            
            ALSp<ALFloatMatrix> h_t = ALFloatMatrix::createCropVirtualMatrix(after, t*getInfo().ow, 0, (t+1)*getInfo().ow-1, batchSize-1);
            ALFloatMatrix::runLineFunction(h_t.get(), c_t.get(), gTanh);
            ALSp<ALFloatMatrix> o_t = ALFloatMatrix::createCropVirtualMatrix(w_x_t.get(), getInfo().ow*3, 0, getInfo().ow*4-1, batchSize-1);
            ALFloatMatrix::productDot(h_t.get(), h_t.get(), o_t.get());
            
            //Copy c_t as c_t_1, h_t as h_t_1
            ALFloatMatrix::copy(c_t_1.get(), c_t.get());
            ALFloatMatrix::copy(h_t_1.get(), h_t.get());
        }
    }
    void LSTMLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        WeightMatrix weight(parameters, getInfo().iw, getInfo().ow);
        WeightMatrix weightDiff(parameters_diff, getInfo().iw, getInfo().ow);
        if (NULL!=before_diff)
        {
        }
    }
    LSTMLayer::LSTMLayer(size_t iw, size_t ow, size_t t):ILayer(iw, ow, _computeParamterSize(iw, ow), 1, 0, 0)
    {
        mTime = t;
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new LSTMLayer(p.uInputSize, p.uOutputSize, p.get("time"));
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "LSTM");
  
};
