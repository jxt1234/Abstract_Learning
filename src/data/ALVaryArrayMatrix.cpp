#include "data/ALVaryArrayMatrix.h"
#include <string.h>
static size_t _getWidth(const ALFloatMatrix* l)
{
    if (NULL == l)
    {
        return 0;
    }
    return l->width();
}


ALVaryArrayMatrix::ALVaryArrayMatrix(const ALVaryArray* array, size_t time, const ALFloatMatrix* target):ALFloatMatrix(time+_getWidth(target)+1, array->size(), 0)
{
    ALASSERT(time>0);
    ALASSERT(array!=NULL);
    mTime = time;
    mCache = new ALFLOAT[time+_getWidth(target)+1];
    mCur = 0;
    mArray = array;
    mLabel = target;

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
    ::memset(mCache, 0, sizeof(ALFLOAT)*seriesSize());
    size_t t_fin = mTime;
    if (a.length < mTime)
    {
        t_fin = a.length;
        mCache[0] = a.length;
    }
    else
    {
        mCache[0] = mTime;
    }

    for (size_t t=1; t<=t_fin; ++t)
    {
        mCache[t] = a.c[t-1];
    }
    if (NULL!=mLabel)
    {
        auto ct = mCache + seriesSize();
        auto y = mLabel->vGetAddr(mCur);
        ::memcpy(ct, y, mLabel->width()*sizeof(ALFLOAT));
    }
}
