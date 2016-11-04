#include "core/ALIExpander.h"
#include "core/ALFloatDataChain.h"
#include "math/ALFloatMatrix.h"
void ALIExpander::expandXY(const ALIExpander* Xe, const ALLabeldData* data, ALSp<ALFloatMatrix> &X, ALSp<ALFloatMatrix> &YT)
{
    ALASSERT(NULL!=data);
    ALASSERT(NULL!=Xe);
    ALASSERT(Xe->vLength() > 0);
    auto datalist = data->get();
    size_t sum = 0;
    auto l = Xe->vLength();
    /*measure, compute valid datas*/
    {
        ALAUTOSTORAGE(_x, ALFLOAT, l);
        for (auto iter : datalist)
        {
            auto dp = iter.second;
            if (Xe->vExpand(dp, _x))
            {
                sum++;
            }
        }
    }
    if (sum == 0)
    {
        return;
    }
    /*Expand*/
    X = ALFloatMatrix::create(l, sum);
    YT = ALFloatMatrix::create(sum, 1);
    int cur = 0;
    ALFLOAT* _y = YT->vGetAddr();
    for (auto iter:datalist)
    {
        auto v = iter.first;
        auto dp = iter.second;
        ALFLOAT* _x = X->vGetAddr(cur);
        if (Xe->vExpand(dp, _x))
        {
            _y[cur] = v;
            cur++;
        }
    }
}
