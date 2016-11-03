#include "ALIndexVirtualMatrix.h"
#include <string.h>
ALIndexVirtualMatrix::ALIndexVirtualMatrix(ALFLOAT** indexes, size_t w, size_t h, bool copy):ALFloatMatrix(w,h, 0)
{
    ALASSERT(h>0);
    ALASSERT(w>0);
    ALASSERT(NULL!=indexes);
    if (copy)
    {
        mOwn = true;
        mIndexes = new ALFLOAT*[h];
        ::memcpy(mIndexes, indexes, h*sizeof(ALFLOAT*));
    }
    else
    {
        mIndexes = indexes;
        mOwn = false;
    }
}

ALIndexVirtualMatrix::~ALIndexVirtualMatrix()
{
    if (mOwn)
    {
        delete [] mIndexes;
    }
}

ALFLOAT* ALIndexVirtualMatrix::vGetAddr(size_t y) const
{
    return mIndexes[y];
}
