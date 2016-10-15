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
        virtual ALFloatMatrix* vInitParameters() const override;
        virtual ALFloatMatrix* vInitOutput(int batchSize) const override;
        virtual bool vCheckInput(const ALFloatMatrix* input) const override;
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const override;
        
        virtual ~ InnerProductLayer(){}
        InnerProductLayer(size_t inputSize, size_t outputSize):mInputSize(inputSize), mOutputSize(outputSize){}
    private:
        size_t mInputSize;
        size_t mOutputSize;
    };
    
}

#endif /* InnerProductLayer_hpp */
