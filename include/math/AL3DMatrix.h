#ifndef INCLUDE_MATH_AL3DMATRIX_H
#define INCLUDE_MATH_AL3DMATRIX_H
#include "ALFloatMatrix.h"
class AL3DMatrix:public ALRefCount
{
public:
    static AL3DMatrix* create(ALSp<ALFloatMatrix> origin, int w, int h);
    virtual ~ AL3DMatrix();
    
    ALFLOAT* getAddr(int y, int z);
    
    inline int width() const {return mWidth;}
    inline int height() const {return mHeight;}
    inline int depth() const  {return mOrigin->height();}
    
private:
    AL3DMatrix(ALSp<ALFloatMatrix> origin, int w, int h);
    
    ALSp<ALFloatMatrix> mOrigin;
    int mWidth;
    int mHeight;
    
};
#endif
