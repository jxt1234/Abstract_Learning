//
//  MaxPoolLayer.hpp
//  abs
//
//  Created by jiangxiaotang on 15/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#ifndef MaxPoolLayer_hpp
#define MaxPoolLayer_hpp

#include <stdio.h>
#include "ILayer.h"
#include "math/ALIMatrix4DOp.h"
namespace ALCNN {
    class MaxPoolLayer : public ILayer
    {
    public:
        MaxPoolLayer(int stride, int width, int height, int depth);
        virtual ~ MaxPoolLayer();
        
        virtual ALFloatMatrix* vInitParameters() const override {return NULL;}
        virtual ALFloatMatrix* vInitOutput(int batchSize) const override;
        virtual bool vCheckInput(const ALFloatMatrix* input) const override;
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const override;
    private:
        int mStride;
        ALIMatrix4DOp::Matrix4D mInput;
        ALIMatrix4DOp::Matrix4D mOutput;
    };
}

#endif /* MaxPoolLayer_hpp */
