#ifndef LEARN_CNN_LAYERWRAP_H
#define LEARN_CNN_LAYERWRAP_H
#include "ILayer.h"
namespace ALCNN {
    
    class LayerWrap : public ALRefCount
    {
    public:
        LayerWrap(ALSp<ILayer> layer);
        virtual ~ LayerWrap();
        int getParameterSize() const;
        void setParameters(const ALFloatMatrix* p, int offset);
        void readParametersDiff(const ALFloatMatrix* p, int offset);
        
        void resetBatchSize(int batchSize);
        
        void connectInput(LayerWrap* input);
        void connectOutput(ALSp<LayerWrap> output);
        
        ALSp<ALFloatMatrix> getOutput() const;
        
        void forward(ALSp<ALFloatMatrix> input);
        
        void backward(ALSp<ALFloatMatrix> error);
        
    private:
        ALSp<ILayer> mLayer;
        
        LayerWrap* mBefore;
        ALSp<LayerWrap> mNext;
        
        
        ALSp<ALFloatMatrix> mOutput;
        ALSp<ALFloatMatrix> mOutputDiff;
        
        ALSp<ALFloatMatrix> mParameters;
        ALSp<ALFloatMatrix> mParameterDiff;
        
        ALSp<ALFloatMatrix> mInput;
        ALSp<ALFloatMatrix> mInputError;
    };
}
#endif
