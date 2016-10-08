#ifndef LEARN_CNN_ILAYER_H
#define LEARN_CNN_ILAYER_H
#include "math/ALFloatMatrix.h"
namespace ALCNN {
class ILayer : public ALRefCount
{
public:
    virtual ALFloatMatrix* vInitParameters() const = 0;
    virtual ALFloatMatrix* vInitOutput(int batchSize) const = 0;
    virtual bool vCheckInput(const ALFloatMatrix* input) const = 0;
    virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters) const = 0;
    virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* before, const ALFloatMatrix* parameters, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff) const = 0;

    virtual ~ ILayer(){}
    
protected:
    ILayer(){}
};
}
#endif
