#ifndef MATH_AL3DBASICMATRIXOP_H
#define MATH_AL3DBASICMATRIXOP_H
#include "math/ALIMatrix4DOp.h"
class ALBasicMatrix4DOp : public ALIMatrix4DOp
{
public:
    virtual void vFilter(Matrix4D& dst, const Matrix4D& src, const Matrix4D& kernelData, int stride) const override;
    
    virtual void vDeterFilter(const Matrix4D& dstDiff, const Matrix4D& dst, const Matrix4D& src,  Matrix4D& srcDiff, const Matrix4D& kernelData, Matrix4D& kernelDataDiff, int stride) const override;

    ALBasicMatrix4DOp(){}
    virtual ~ ALBasicMatrix4DOp(){}
};
#endif
