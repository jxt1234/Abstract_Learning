#ifndef SRC_MATH_ALINDEXVIRTUALMATRIX_H
#define SRC_MATH_ALINDEXVIRTUALMATRIX_H
#include "math/ALFloatMatrix.h"
class ALIndexVirtualMatrix:public ALFloatMatrix
{
public:
    ALIndexVirtualMatrix(ALFLOAT** indexes, size_t w, size_t h);
    virtual ~ALIndexVirtualMatrix();
    
    virtual ALFLOAT* vGetAddr(size_t y) const;
private:
    ALFLOAT** mIndexes;
};
#endif
