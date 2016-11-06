#ifndef LEARN_CNN_SOFTMAXLAYER_H
#define LEARN_CNN_SOFTMAXLAYER_H
#include "ILayer.h"
namespace ALCNN {
    class SoftMaxLayer : public ILayer
    {
    public:
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const override;
        
        SoftMaxLayer(int inputSize, int outputSize);
        virtual ~ SoftMaxLayer();

    };
}
#endif
