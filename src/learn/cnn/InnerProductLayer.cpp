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
    void InnerProductLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        ALFloatMatrix::product(after, before, parameters);
    }
    void InnerProductLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        //ALSp<ALFloatMatrix> after_diff_T = ALFloatMatrix::transpose(after_diff);
        ALFloatMatrix::productTA(parameters_diff, before, after_diff);
        
        if (NULL!=before_diff)
        {
            ALFloatMatrix::productT(before_diff, after_diff, parameters);
        }
    }
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new InnerProductLayer(p.uInputSize, p.uOutputSize);
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "InnerProduct");
}
