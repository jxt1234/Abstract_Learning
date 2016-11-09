#include "data/ALVaryArrayMatrix.h"
#include <string.h>

ALVaryArrayMatrix::ALVaryArrayMatrix(const ALVaryArray* array, size_t time, size_t number):ALFloatMatrix(time*number, array->size(), 0)
{
    ALASSERT(number>0);
    ALASSERT(time>0);
    ALASSERT(array!=NULL);
    mTime = time;
    mNumber = number;
    mCache = new ALFLOAT[time*number];
    mCur = 0;
    mArray = array;

    _refreshCache();
}
ALVaryArrayMatrix::~ ALVaryArrayMatrix()
{
    delete [] mCache;
}
ALFLOAT* ALVaryArrayMatrix::vGetAddr(size_t y) const
{
    ALASSERT(y < height());
    if (y == mCur)
    {
        return mCache;
    }
    mCur = y;
    _refreshCache();
    return mCache;
}

void ALVaryArrayMatrix::_refreshCache() const
{
    auto a = mArray->getArray(mCur);
    ::memset(mCache, 0, sizeof(ALFLOAT)*mTime*mNumber);
    size_t t_sta = 0;
    int sta = a.length-mTime;
    if (sta < 0)
    {
        t_sta = mTime-a.length;
    }

    for (size_t t=t_sta; t<mTime; ++t)
    {
        auto ct = mCache + t*mNumber;
        ct[a.c[sta+t]] = 1.0f;
    }
}
