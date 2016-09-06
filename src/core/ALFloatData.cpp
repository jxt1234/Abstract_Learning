#include "core/ALFloatData.h"
#include <string.h>
#include "utils/ALUtils.h"
ALFloatData::ALFloatData(size_t num)
{
    ALASSERT(0<num);
    mNum = num;
    mData = new ALFLOAT[num];
    mFront = NULL;
}

ALFloatData::~ALFloatData()
{
    delete mData;
}

void ALFloatData::load(std::istream& input)
{
    for (int i=0; i<mNum; ++i)
    {
        input >> mData[i];
    }
}
void ALFloatData::copy(void* dst) const
{
    ::memcpy(dst, mData, sizeof(ALFLOAT)*mNum);
}

ALFLOAT ALFloatData::value(int n) const
{
    ALASSERT( 0<=n && n<mNum);
    return mData[n];
}
void ALFloatData::addNext(ALFloatData* next)
{
    ALASSERT(NULL!=next);
    next->mFront = this;
}

bool ALFloatData::canBack(int l) const
{
    bool result = true;
    const ALFloatData* res = this;
    for (int i=0; i<l; ++i)
    {
        res = res->front();
        if (NULL == res)
        {
            result = false;
            break;
        }
    }
    return result;
}
void ALLabeldData::collect(std::vector<const ALFloatData*>& result) const
{
    for (auto p : mData)
    {
        result.push_back(p.second);
    }
}
