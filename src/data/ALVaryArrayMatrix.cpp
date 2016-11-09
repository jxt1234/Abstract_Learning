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


ALVaryArrayMatrix::ALVaryArrayMatrix(const ALVaryArray* array, size_t time, size_t number, const ALFloatMatrix* target):ALFloatMatrix(time*number+_getWidth(target), array->size(), 0)
{
    ALASSERT(number>0);
    ALASSERT(time>0);
    ALASSERT(array!=NULL);
    mTime = time;
    mNumber = number;
    mCache = new ALFLOAT[time*number+_getWidth(target)];
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
    ::memset(mCache, 0, sizeof(ALFLOAT)*mTime*mNumber);
    size_t t_fin = mTime;
    if (a.length < mTime)
    {
        t_fin = a.length;
    }

    for (size_t t=0; t<t_fin; ++t)
    {
        auto ct = mCache + t*mNumber;
        ct[a.c[t]] = 1.0f;
    }
    if (NULL!=mLabel)
    {
        auto ct = mCache + mTime*mNumber;
        auto y = mLabel->vGetAddr(mCur);
        ::memcpy(ct, y, mLabel->width()*sizeof(ALFLOAT));
    }
}
