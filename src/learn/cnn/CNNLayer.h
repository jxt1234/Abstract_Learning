#ifndef SRC_LEARN_CNN_CNNLAYER_H
#define SRC_LEARN_CNN_CNNLAYER_H
#include "ILayer.h"
#include "math/ALIMatrix4DOp.h"

#include <vector>
namespace ALCNN {


class CNNLayer:public ILayer
{
public:
    CNNLayer(int inputSize, int inputChannel, int kernelSize, int kernelNumber, int stride);
    virtual ~ CNNLayer();
    
    virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const override;
    virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const override;
private:
    int mStride = 1;
    int mFilterNumber;
    
    ALIMatrix4DOp::Matrix4D mInputInfo;
    ALIMatrix4DOp::Matrix4D mOutputInfo;
    ALIMatrix4DOp::Matrix4D mKernelInfo;
    
    ALSp<ALIMatrix4DOp> mMatrixOp;
};

}

#endif
