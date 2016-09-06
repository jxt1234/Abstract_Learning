#include "core/ALNormalizer.h"
#include "ALHead.h"
using namespace std;

ALNormalizer::ALNormalizer(ALFLOAT* k, ALFLOAT* b, ALIExpander* base)
{
    ALASSERT(NULL!=k);
    ALASSERT(NULL!=b);
    ALASSERT(NULL!=base);
    mBase = base;
    base->addRef();
    mL = base->vLength();

    mK = new ALFLOAT[mL];
    mB = new ALFLOAT[mL];

    memcpy(mK, k, mL*sizeof(ALFLOAT));
    memcpy(mB, b, mL*sizeof(ALFLOAT));
}
ALNormalizer::ALNormalizer(const std::vector<const ALFloatData*>& data, ALIExpander* base)
{
    ALASSERT(NULL!=base);
    mBase = base;
    mBase->addRef();

    int l = mBase->vLength();
    mL = l;
    //TODO Support the case that l <=0 or ALFloatDataChain's size <1
    ALASSERT(l>0);
    ALASSERT(data.size()>0);
    mK = new ALFLOAT[l];
    mB = new ALFLOAT[l];

    ALAutoStorage<ALFLOAT> max_v(l);
    ALAutoStorage<ALFLOAT> min_v(l);
    ALAutoStorage<ALFLOAT> value(l);
    ALFLOAT* M = max_v.get();
    ALFLOAT* m = min_v.get();
    ALFLOAT* v = value.get();

    auto iter = data.begin();
    /*Find the first data*/
    bool res = false;
    for (;iter!=data.end();iter++)
    {
        res = mBase->vExpand((*iter), v);
        if (res)
        {
            memcpy(M, v, l*sizeof(ALFLOAT));
            memcpy(m, v, l*sizeof(ALFLOAT));
            break;
        }
    }
    ALASSERT(true == res);//FIXME
    /*Compute max and min*/
    for (;iter!=data.end();iter++)
    {
        res = mBase->vExpand((*iter), v);
        if (res)
        {
            for (int i=0; i<mL; ++i)
            {
                M[i] = max(M[i], v[i]);
                m[i] = min(m[i], v[i]);
            }
        }
    }
    /*Turn to K, B*/
    for (int i=0; i<mL; ++i)
    {
        mK[i] = (ALFLOAT)1/(M[i]-m[i]);
        mB[i] = -m[i]/(M[i]-m[i]);
    }
}

ALNormalizer::~ALNormalizer()
{
    delete [] mK;
    delete [] mB;
    mBase->decRef();
}

bool ALNormalizer::vExpand(const ALFloatData* d, ALFLOAT* dst) const
{
    bool res = mBase->vExpand(d, dst);
    if (!res) return res;
    for (int i=0; i<mL; ++i)
    {
        dst[i] = mK[i]*dst[i] + mB[i];
    }
    return res;
}

void ALNormalizer::reverse(ALFLOAT* dst)
{
    for (int i=0; i<mL; ++i)
    {
        dst[i] = (dst[i] - mB[i])/mK[i];
    }
}

