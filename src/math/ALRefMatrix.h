#ifndef MATH_ALREFMATRIX_H
#define MATH_ALREFMATRIX_H
#include "math/ALFloatMatrix.h"
class ALRefMatrix : public ALFloatMatrix
{
public:
    ALRefMatrix(ALFLOAT* base, size_t w, size_t h);
    virtual ~ALRefMatrix();
    virtual ALFLOAT* vGetAddr(size_t y=0) const;
private:
    ALFLOAT* mBase;
};
#endif
