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
        
        virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const override;
        virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const override;
    private:
        int mStride;
        ALIMatrix4DOp::Matrix4D mInput;
        ALIMatrix4DOp::Matrix4D mOutput;
    };
}
#endif
