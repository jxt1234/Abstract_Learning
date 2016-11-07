#ifndef LEARN_CNN_LAYERWRAP_H
#define LEARN_CNN_LAYERWRAP_H
#include "ILayer.h"
#include <ostream>
namespace ALCNN {
    
    class LayerWrap : public ALRefCount
    {
    public:
        LayerWrap(ALSp<ILayer> layer);
        virtual ~ LayerWrap();
        size_t getParameterSize() const;
        void mapParameters(const ALFloatMatrix* p, size_t offset);
        void mapParametersDiff(const ALFloatMatrix* p, size_t offset);
        
        
        void connectInput(LayerWrap* input);
        void connectOutput(ALSp<LayerWrap> output);
        
        ALSp<ALFloatMatrix> forward(ALSp<ALFloatMatrix> input);
        
        void backward(ALSp<ALFloatMatrix> error);
        
        void setForwardDebug(std::ostream* output) {mForwardDump = output;}
        
    private:
        ALSp<ILayer> mLayer;
        
        LayerWrap* mBefore;
        ALSp<LayerWrap> mNext;
        
        
        ALSp<ALFloatMatrix> mParameters;
        ALSp<ALFloatMatrix> mParameterDiff;
        ALSp<ALFloatMatrix> mOutput;
        
        ALSp<ALFloatMatrix> mInput;
        ALSp<ALFloatMatrix> mCache;
        size_t mCacheSize;
        
        std::ostream* mForwardDump = NULL;
    };
}
#endif
