//
//  EmbeddingLayer.hpp
//  abs
//
//  Created by jiangxiaotang on 10/11/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#ifndef EmbeddingLayer_hpp
#define EmbeddingLayer_hpp

#include <stdio.h>
#include "ILayer.h"
namespace ALCNN {
    class EmbeddingLayer : public ILayer
    {
    public:
        EmbeddingLayer(size_t iw, size_t ow, size_t time, size_t numbers);
        virtual ~ EmbeddingLayer();
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const override;
    private:
        size_t mNumber;
    };
};

#endif /* EmbeddingLayer_hpp */
