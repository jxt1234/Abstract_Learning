#ifndef MATH_ALI3DMATRIXOP_H
#define MATH_ALI3DMATRIXOP_H
#include "ALFloatMatrix.h"
class ALIMatrix4DOp : public ALRefCount
{
public:
    struct Matrix4D
    {
        //Each line is a 3D Matrix with iExpand extra elements
        const ALFloatMatrix* pOrigin = NULL;
        int iWidth;
        int iHeight;
        int iDepth;
        int iExpand = 0;
        bool valid() const;
        int getTotalWidth() const;
        
        ALFloatMatrix* getMutable() {return (ALFloatMatrix*)pOrigin;}
    };
    
    
    virtual void vFilter(Matrix4D& dst, const Matrix4D& src, const Matrix4D& kernelData, int stride) const = 0;
    virtual void vDeterFilter(const Matrix4D& dstDiff, const Matrix4D& src, Matrix4D& srcDiff/*Output*/, const Matrix4D& kernelData, Matrix4D& kernelDataDiff/*Output*/, int stride) const = 0;

    
    ALIMatrix4DOp(){}
    virtual ~ ALIMatrix4DOp(){}
    
    typedef enum {
        BASIC,
        OPENCL,
        SSE
    } TYPE;
    
    static ALIMatrix4DOp* create(TYPE t = BASIC);
};
#endif
