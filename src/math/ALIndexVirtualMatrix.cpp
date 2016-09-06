#include "ALIndexVirtualMatrix.h"
#include <string.h>
ALIndexVirtualMatrix::ALIndexVirtualMatrix(ALFLOAT** indexes, size_t w, size_t h):ALFloatMatrix(w,h)
{
    ALASSERT(h>0);
    ALASSERT(w>0);
    mIndexes = indexes;
//    mIndexes = new ALFLOAT*[h];
//    ::memcpy(mIndexes, indexes, h*sizeof(ALFLOAT*));
}

ALIndexVirtualMatrix::~ALIndexVirtualMatrix()
{
//    delete [] mIndexes;
}

ALFLOAT* ALIndexVirtualMatrix::vGetAddr(size_t y) const
{
    return mIndexes[y];
}
