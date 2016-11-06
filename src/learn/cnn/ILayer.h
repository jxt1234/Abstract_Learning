#ifndef LEARN_CNN_ILAYER_H
#define LEARN_CNN_ILAYER_H
#include "math/ALFloatMatrix.h"
namespace ALCNN {
class ILayer : public ALRefCount
{
public:

    virtual void vForward(const ALFloatMatrix* before, ALFloatMatrix* after, const ALFloatMatrix* parameters, ALFloatMatrix* cache) const = 0;
    virtual void vBackward(const ALFloatMatrix* after_diff, const ALFloatMatrix* after, const ALFloatMatrix* parameters, const ALFloatMatrix* before, ALFloatMatrix* before_diff, ALFloatMatrix* parameters_diff, ALFloatMatrix* cache) const = 0;

    virtual ~ ILayer(){}

    struct Info
    {
        size_t iw;//Input width
        size_t ow;//Output width
        size_t pw;//Parameter width
        size_t ph;//Parameters height
        size_t cw;//Cache width
        size_t ch;//Cache height
    };
    
    const Info& getInfo() const {return mInfo;}
protected:
    ILayer(size_t iw, size_t ow, size_t pw, size_t ph, size_t cw, size_t ch);
private:
    Info mInfo;
};
}
#endif
