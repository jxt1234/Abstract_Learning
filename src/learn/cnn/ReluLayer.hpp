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
        virtual ALFloatMatrix* vInitParameters() const override;
        virtual ALFloatMatrix* vInitOutput(int batchSize) const override;
        virtual bool vCheckInput(const ALFloatMatrix* input) const override;
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const override;
        
        virtual ~ ReluLayer(){}
        ReluLayer(int inputWidth)
        {
            mInputWidth = inputWidth;
        }
    private:
        ALFLOAT mSlope = 0.0f;
        size_t mInputWidth;
    };
}

#endif /* ReluLayer_hpp */
