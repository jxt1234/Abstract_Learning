#include "ALBaseFloatMatrix.h"
int ALBaseFloatMatrix::gNum = 0;
ALBaseFloatMatrix::ALBaseFloatMatrix(size_t w, size_t h):ALFloatMatrix(w, h)
{
    ALASSERT(w>0 && h>0);
    size_t size = w*h;
    mBase = new ALFLOAT[size];

    gNum++;//FIXME DEBUG
}

ALBaseFloatMatrix::~ALBaseFloatMatrix()
{
    ALASSERT(NULL!=mBase);
    delete [] mBase;

    gNum--;//FIXME DEBUG
}

ALFLOAT* ALBaseFloatMatrix::vGetAddr(size_t y) const
{
    ALASSERT(y < height());
    return mBase+y*width();
}
