#include "core/ALFloatDataChain.h"
#include <iostream>

using namespace std;

ALFloatDataChain::~ALFloatDataChain()
{
    auto iter = mSeries.begin();
    for (; iter!=mSeries.end(); ++iter)
    {
        (*iter)->decRef();
    }
}

void ALFloatDataChain::add(ALFloatData* d)
{
    ALASSERT(NULL!=d);
    ALASSERT(mWidth == d->num());
    mSeries.push_back(d);
    d->addRef();
}
void ALFloatDataChain::expand(void* dst, int stride) const
{
    ALASSERT(NULL!=dst);
    ALASSERT(stride > 0);
    unsigned char* d = (unsigned char*)dst;
    for (int i=0; i<mSeries.size(); ++i)
    {
        auto data = mSeries[i]->get();
        ::memcpy(d+stride*i, data, mSeries[i]->num()*sizeof(ALFLOAT));
    }
}
