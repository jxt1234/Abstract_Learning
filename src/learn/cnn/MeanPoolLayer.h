#ifndef LEARN_CNN_MeanPoolLayer_H
#define LEARN_CNN_MeanPoolLayer_H
#include "ILayer.h"
#include "math/ALIMatrix4DOp.h"
namespace ALCNN {
    class MeanPoolLayer : public ILayer
    {
    public:
        MeanPoolLayer(int stride, int width, int height, int depth);
        virtual ~ MeanPoolLayer();
        
        virtual ALFloatMatrix* vInitParameters() const override {return NULL;}
        virtual ALFloatMatrix* vInitOutput(int batchSize) const override;
        virtual bool vCheckInput(const ALFloatMatrix* input) const override;
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* before, const ALFloatMatrix* parameters, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const override;
    private:
        int mStride;
        ALIMatrix4DOp::Matrix4D mInput;
        ALIMatrix4DOp::Matrix4D mOutput;
    };
}
#endif
