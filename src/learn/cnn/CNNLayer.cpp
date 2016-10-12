#include "CNNLayer.h"
namespace ALCNN {
    CNNLayer::CNNLayer(int inputSize, int inputChannel, int kernelSize, int kernelNumber, int stride)
    {
        ALASSERT(inputSize>=kernelSize);
        ALASSERT(kernelSize>=1);
        ALASSERT(kernelNumber>=1);
        ALASSERT(stride>=1);
        ALASSERT(inputChannel>=1);

        mInputInfo.iDepth = inputChannel;
        mInputInfo.iWidth = inputSize;
        mInputInfo.iHeight = inputSize;
        mInputInfo.iExpand = 0;
        
        mOutputInfo.iDepth = kernelNumber;
        mOutputInfo.iWidth = (inputSize-kernelSize)/stride+1;
        mOutputInfo.iHeight = (inputSize-kernelSize)/stride+1;
        mOutputInfo.iExpand = 0;

        mKernelInfo.iDepth = inputChannel;
        mKernelInfo.iHeight = kernelSize;
        mKernelInfo.iWidth = kernelSize;
        mKernelInfo.iExpand = 1;
        mFilterNumber = kernelNumber;
        
        mStride = stride;
        
        mMatrixOp = ALIMatrix4DOp::create();
    }
    CNNLayer::~ CNNLayer()
    {
        
    }
    
    ALFloatMatrix* CNNLayer::vInitParameters() const
    {
        auto w = mKernelInfo.iWidth*mKernelInfo.iHeight*mKernelInfo.iDepth+mKernelInfo.iExpand;
        auto h = mFilterNumber;
        return ALFloatMatrix::create(w, h);
    }
    
    ALFloatMatrix* CNNLayer::vInitOutput(int batchSize) const
    {
        ALASSERT(batchSize>=1);
        return ALFloatMatrix::create(mOutputInfo.getTotalWidth(), batchSize);
    }
    bool CNNLayer::vCheckInput(const ALFloatMatrix* input) const
    {
        ALASSERT(NULL!=input);
        return input->width() == mInputInfo.getTotalWidth();
    }
    void CNNLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const
    {
        ALIMatrix4DOp::Matrix4D output = mOutputInfo;
        output.pOrigin = after;
        
        ALIMatrix4DOp::Matrix4D input = mInputInfo;
        input.pOrigin = before;
        
        ALIMatrix4DOp::Matrix4D kernel = mKernelInfo;
        kernel.pOrigin = parameters;
        
        mMatrixOp->vFilter(output, input, kernel, mStride);
    }
    void CNNLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const
    {
        ALIMatrix4DOp::Matrix4D output_diff = mOutputInfo;
        output_diff.pOrigin = after_diff;
        
        ALIMatrix4DOp::Matrix4D output = mOutputInfo;
        output.pOrigin = after;
        
        ALIMatrix4DOp::Matrix4D kernel = mKernelInfo;
        kernel.pOrigin = parameters;

        ALIMatrix4DOp::Matrix4D input_diff= mInputInfo;
        input_diff.pOrigin = before_diff;

        ALIMatrix4DOp::Matrix4D input= mInputInfo;
        input.pOrigin = before;

        ALIMatrix4DOp::Matrix4D kernel_diff = mKernelInfo;
        kernel_diff.pOrigin = parameters_diff;

        mMatrixOp->vDeterFilter(output_diff, output, input, input_diff, kernel, kernel_diff, mStride);
    }
}
