//
//  ReluLayer.hpp
//  abs
//
//  Created by jiangxiaotang on 14/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#ifndef ReluLayer_hpp
#define ReluLayer_hpp

#include <stdio.h>
#include "ILayer.h"

namespace ALCNN {
    class ReluLayer : public ILayer
    {
    public:
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const override;
        
        virtual ~ ReluLayer(){}
        ReluLayer(int iw):ILayer(iw, iw, 0, 0, 0, 0){}
    private:
        ALFLOAT mSlope = 0.0f;
    };
}

#endif /* ReluLayer_hpp */
