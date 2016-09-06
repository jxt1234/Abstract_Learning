#ifndef INCLUDE_MATH_ALIMATRIXTRANSFORMER_H
#define INCLUDE_MATH_ALIMATRIXTRANSFORMER_H
#include "ALFloatMatrix.h"
class ALIMatrixTransformer:public ALRefCount
{
public:
    virtual ALFloatMatrix* vTransform(const ALFloatMatrix* origin) const = 0;
    virtual void vPrint(std::ostream& output) const{}
    ALIMatrixTransformer(){}
    virtual ~ALIMatrixTransformer(){}
};


#endif
