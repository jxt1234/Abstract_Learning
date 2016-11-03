//
//  LSTMLayer.hpp
//  abs
//
//  Created by jiangxiaotang on 02/11/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#ifndef LSTMLayer_hpp
#define LSTMLayer_hpp

#include <stdio.h>
#include "ILayer.h"
namespace ALCNN {
    class LSTMLayer : public ILayer
    {
    public:
        virtual ALFloatMatrix* vInitParameters() const override;
        virtual ALFloatMatrix* vInitOutput(int batchSize) const override;
        virtual bool vCheckInput(const ALFloatMatrix* input) const override;
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const override;
        
        LSTMLayer(size_t iw, size_t ow, size_t t):mInputWidth(iw),mOutputWidth(ow),mTime(t){}
        virtual ~ LSTMLayer(){}
        
    private:
        size_t _computeParamterSize() const;
        
        size_t mInputWidth;
        size_t mOutputWidth;
        size_t mTime;
    };
};
#endif /* LSTMLayer_hpp */
