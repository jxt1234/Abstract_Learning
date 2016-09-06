#include <math.h>
#include <algorithm>
#include <iostream>
#include "SMO.h"
namespace ALSMO
{
#define MVALUE(m, i, j) ((m)->vGetAddr(j)[(i)])
#define ISSUPPORT(i) (0<a[i]&&a[i]<c[i])
SMO::SMO(int n)
{
}

SMO::~SMO()
{
}

inline static void assertForSMO(ALFloatMatrix* alpha/*output*/, ALFLOAT& b/*output*/, const ALFloatMatrix* CT, const ALFloatMatrix* KX, const ALFloatMatrix* YT)
{
    ALASSERT(NULL!=alpha);
    ALASSERT(NULL!=KX);
    ALASSERT(NULL!=YT);
    ALASSERT(1==CT->height());
    ALASSERT(1==YT->height());
    auto l = CT->width();
    ALASSERT(l>=1);
    ALASSERT(l == YT->width());
    ALASSERT(l == KX->height());
    ALASSERT(l == KX->width());
    ALASSERT(l == alpha->width());
    ALASSERT(1 == alpha->height());
}

static int maxOrNeighbour(ALFLOAT* a, ALFLOAT* c, ALFLOAT* e, int mid, size_t n)
{
    ALASSERT(0<=mid && mid<n);
    int pos = -1;
    ALFLOAT maxE = -1.0;
    ALFLOAT emid = e[mid];
#define SELECT \
        if (ISSUPPORT(i))\
        {\
            ALFLOAT d = e[i]-emid;\
            d = d*d;\
            if (d > maxE)\
            {\
                pos = i;\
                maxE = d;\
            }\
        }
    for (int i=0; i<mid; ++i)
    {
        SELECT;
    }
    for (int i=mid+1; i<n; ++i)
    {
        SELECT;
    }
#undef SELECT
    if (pos == -1)
    {
        pos = mid+1;
        if (pos >= n)
        {
            pos = 0;
        }
    }
    return pos;
}

static void refreshE(const ALFLOAT* y, const ALFLOAT* a, ALFLOAT b, ALFLOAT* e, const ALFloatMatrix* KX, size_t l)
{
    /*Compute E*/
    for (int i=0; i<l; ++i)
    {
        e[i] = (b-y[i]);
        ALFLOAT* k = KX->vGetAddr(i);
        for (int j=0; j<l; ++j)
        {
            e[i]+=a[j]*k[j]*y[j];
        }
    }
}
    
void SMO::sovle(ALFloatMatrix* alpha/*output*/, ALFLOAT& b/*output*/, const ALFloatMatrix* CT, const ALFloatMatrix* KX, const ALFloatMatrix* YT, int maxIters)
{
    ALAUTOTIME;
    //ALRandom::init();
    assertForSMO(alpha, b, CT, KX, YT);
    ALASSERT(maxIters>1);
    auto l = YT->width();
    ALFLOAT* a = alpha->vGetAddr();
    ::memset(a, 0, sizeof(ALFLOAT)*l);
    ALSp<ALFloatMatrix> E = ALFloatMatrix::create(l, 1);
    ALFLOAT* e = E->vGetAddr();
    ALFLOAT* y = YT->vGetAddr();
    ALFLOAT* c = CT->vGetAddr();
    b = 0.0;
    refreshE(y, a, b, e, KX, l);
    for(; maxIters > 0; maxIters--)
    {
        for (int i=0; i<l; ++i)
        {
            if ((y[i]*e[i] < 0 && a[i] < c[i])
                    || (y[i]*e[i] > 0 && a[i] > 0))
            {
                int j = maxOrNeighbour(a, c, e, i, l);
                ALFLOAT Lj;
                ALFLOAT Hj;
                if (ZERO(y[i]-y[j]))
                {
                    Lj = std::max((ALFLOAT)0.0, a[i]+a[j]-c[j]);
                    Hj = std::min(c[j], a[i]+a[j]);
                }
                else
                {
                    Lj = std::max((ALFLOAT)0.0, a[j]-a[i]);
                    Hj = std::min(c[j], c[j]-a[i]+a[j]);
                }
                ALFLOAT eta = 2*MVALUE(KX,i,j) - MVALUE(KX,i,i) - MVALUE(KX,j,j);
                if (eta>=0)
                {
                    continue;
                }
                ALFLOAT aj = a[j]-y[j]*(e[i]-e[j])/eta;
                aj = std::max(Lj, aj);
                aj = std::min(Hj, aj);//limit to [Lj, Hj]
                ALFLOAT ai = a[i]+y[i]*y[j]*(a[j]-aj);
                /*Update b*/
                ALFLOAT b1 = b-e[i]-y[i]*(ai-a[i])*MVALUE(KX, i, i)-y[j]*(aj-a[j])*MVALUE(KX, i, j);
                ALFLOAT b2 = b-e[j]-y[i]*(ai-a[i])*MVALUE(KX, i, j)-y[j]*(aj-a[j])*MVALUE(KX, j, j);
                auto oldb = b;
                if (0 < ai && ai < c[i])
                {
                    b = b1;
                }
                else if(0 < aj && aj < c[j])
                {
                    b = b2;
                }
                else
                {
                    b = (b1+b2)/2.0;
                }
                for (int ii=0; ii<l; ++ii)
                {
                    ALFLOAT* k = KX->vGetAddr(ii);
                    auto diff = b - oldb + (aj-a[j])*k[j]*y[j] + (ai-a[i])*k[i]*y[i];
                    e[ii] = e[ii] + diff;
                }
                /*Refresh alpha at last*/
                a[j] = aj;
                a[i] = ai;
            }
        }
    }
}

};
