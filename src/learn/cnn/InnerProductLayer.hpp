//
//  InnerProductLayer.hpp
//  abs
//
//  Created by jiangxiaotang on 15/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#ifndef InnerProductLayer_hpp
#define InnerProductLayer_hpp

#include <stdio.h>
#include "ILayer.h"
namespace ALCNN {
    class InnerProductLayer : public ILayer
    {
    public:
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const override;
        
        virtual ~ InnerProductLayer(){}
        //TODO Add basis
        InnerProductLayer(size_t iw, size_t ow):ILayer(iw, ow, ow, iw, 0, 0){}
    };
    
}

#endif /* InnerProductLayer_hpp */
