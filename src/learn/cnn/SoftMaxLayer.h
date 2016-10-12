#ifndef LEARN_CNN_SOFTMAXLAYER_H
#define LEARN_CNN_SOFTMAXLAYER_H
#include "ILayer.h"
namespace ALCNN {
    class SoftMaxLayer : public ILayer
    {
    public:
        virtual ALFloatMatrix* vInitParameters() const override;
        virtual ALFloatMatrix* vInitOutput(int batchSize) const override;
        virtual bool vCheckInput(const ALFloatMatrix* input) const override;
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const override;
        
        SoftMaxLayer(int inputSize, int outputSize);
        virtual ~ SoftMaxLayer();

    private:
        int mOutputSize;
        int mInputSize;

    };
}
#endif
