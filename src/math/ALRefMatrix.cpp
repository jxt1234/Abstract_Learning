#include "ALRefMatrix.h"

ALRefMatrix::ALRefMatrix(ALFLOAT* base, size_t w, size_t h):ALFloatMatrix(w, h)
{
    mBase = base;
}
ALRefMatrix::~ALRefMatrix()
{
}
ALFLOAT* ALRefMatrix::vGetAddr(size_t y) const
{
    return mBase + y*width();
}
