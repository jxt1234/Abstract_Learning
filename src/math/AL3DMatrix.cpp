#include "math/AL3DMatrix.h"

AL3DMatrix::AL3DMatrix(ALSp<ALFloatMatrix> origin, int w, int h)
{
    ALASSERT(NULL!=origin.get());
    ALASSERT(w>0&&h>0);
    ALASSERT(w*h == origin->width());
    mOrigin = origin;
    mWidth = w;
    mHeight = h;
}
AL3DMatrix::~ AL3DMatrix()
{
}

ALFLOAT* AL3DMatrix::getAddr(int y, int z) const
{
    ALFLOAT* basic = mOrigin->vGetAddr(z);
    return basic + mWidth*y;
}
AL3DMatrix* AL3DMatrix::create(ALSp<ALFloatMatrix> origin, int w, int h)
{
    return new AL3DMatrix(origin, w, h);
}

ALSp<ALFloatMatrix> AL3DMatrix::getReferenceMatrix(int z) const
{
    ALFLOAT* basic = mOrigin->vGetAddr(z);
    return ALFloatMatrix::createRefMatrix(basic, mWidth, mHeight);
}
