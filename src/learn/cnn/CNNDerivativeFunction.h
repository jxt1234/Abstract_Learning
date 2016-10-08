#ifndef LEARN_CNN_CNNDERIVATIVEFUNCTION_H
#define LEARN_CNN_CNNDERIVATIVEFUNCTION_H
#include "math/ALIGradientDecent.h"
#include "LayerWrap.h"
namespace ALCNN {
    class CNNDerivativeFunction : public ALIGradientDecent::DerivativeFunction
    {
    public:
        virtual ALFloatMatrix* vCompute(ALFloatMatrix* coefficient, const ALFloatMatrix* Merge) const override;
        
        CNNDerivativeFunction(ALSp<LayerWrap> net, int outputSize);
        virtual ~ CNNDerivativeFunction();
    private:
        ALSp<LayerWrap> mNet;
        int mOutputSize;
    };
}
#endif
