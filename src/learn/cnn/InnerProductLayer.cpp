//
//  InnerProductLayer.cpp
//  abs
//
//  Created by jiangxiaotang on 15/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "InnerProductLayer.hpp"
#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    //TODO Add basis
    ALFloatMatrix* InnerProductLayer::vInitParameters() const
    {
        return ALFloatMatrix::create(mInputSize, mOutputSize);
    }
    
    ALFloatMatrix* InnerProductLayer::vInitOutput(int batchSize) const
    {
        ALASSERT(batchSize>=1);
        return ALFloatMatrix::create(mOutputSize, batchSize);
    }
    bool InnerProductLayer::vCheckInput(const ALFloatMatrix* input) const
    {
        ALASSERT(NULL!=input);
        return input->width() == mInputSize;
    }
    void InnerProductLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const
    {
        ALLEARNAUTOTIME;
        ALFloatMatrix::productT(after, before, parameters);
    }
    void InnerProductLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const
    {
        ALLEARNAUTOTIME;
        ALSp<ALFloatMatrix> after_diff_T = ALFloatMatrix::transpose(after_diff);
        ALFloatMatrix::product(parameters_diff, after_diff_T.get(), before);
        
        if (NULL!=before_diff)
        {
            ALFloatMatrix::product(before_diff, after_diff, parameters);
        }
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new InnerProductLayer(p.uInputSize, p.uOutputSize);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "InnerProduct");
}