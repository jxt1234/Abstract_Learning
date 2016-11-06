#include "CNNLayer.h"
#include "LayerFactoryRegistor.hpp"
namespace ALCNN {
    static size_t _computeSize(int size, int channel)
    {
        return size*size*channel;
    }

    static size_t _computeOutput(int size, int kernelSize, int kernelNumber, int stride)
    {
        int outputSize = (size - kernelSize)/stride + 1;
        return outputSize*outputSize*kernelNumber;
    }

    CNNLayer::CNNLayer(int inputSize, int inputChannel, int kernelSize, int kernelNumber, int stride):ILayer(_computeSize(inputSize, inputChannel), _computeOutput(inputSize, kernelSize, kernelNumber, stride), inputChannel*kernelSize*kernelSize+1, kernelNumber, 0, 0)
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
        //mMatrixOp = ALIMatrix4DOp::create(ALIMatrix4DOp::BASIC);
    }
    CNNLayer::~ CNNLayer()
    {
        
    }
    void CNNLayer::vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
        ALIMatrix4DOp::Matrix4D output = mOutputInfo;
        output.pOrigin = after;
        
        ALIMatrix4DOp::Matrix4D input = mInputInfo;
        input.pOrigin = before;
        
        ALIMatrix4DOp::Matrix4D kernel = mKernelInfo;
        kernel.pOrigin = parameters;
        
        mMatrixOp->vFilter(output, input, kernel, mStride);
    }
    void CNNLayer::vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const
    {
        ALLEARNAUTOTIME;
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
    
    static auto gCreateFunction = [](const LayerParameters& p) {
        return new CNNLayer(p.mMatrixInfo.iWidth, p.mMatrixInfo.iDepth, p.get("kernelSize"), p.get("kernelNumber"), p.get("stride"));
    };
    
    static LayerFactoryRegister __reg(gCreateFunction, "Convolution");

}
