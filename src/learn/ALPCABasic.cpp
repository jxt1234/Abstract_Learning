#include "learn/ALPCABasic.h"
#include "math/ALStatistics.h"
#include <iostream>
ALPCABasic::ALPCABasic(const ALFloatMatrix* XT, ALFLOAT rate)
{
    ALASSERT(NULL!=XT);
    ALASSERT(rate>0 && rate<=1);
    ALSp<ALFloatMatrix> coe = ALStatistics::covariance(XT);
    ALSp<ALFloatMatrix> root;
    ALSp<ALFloatMatrix> vector;
    ALStatistics::characteristic_compute(coe.get(), root, vector);
    auto n = root->width();
    ALFLOAT* r = root->vGetAddr();
    ALFLOAT sum = 0;
    for (int i=0; i<n; ++i)
    {
        sum += r[i];
    }
    ALFLOAT target = sum*rate;
    int select = 0;
    sum = 0;
    for (;select<n; ++select)
    {
        sum += r[select];
        if (sum >= target)
        {
            break;
        }
    }
    select = select + 1;
    /*Transpose the vector inorder to improve speed*/
    mTransform = ALFloatMatrix::transpose(vector.get());
    for (int i=0; i<n; ++i)
    {
        for (int j=0; j<select; ++j)
        {
            mTransform->vGetAddr(j)[i] = vector->vGetAddr(i)[j];
        }
    }
}
ALPCABasic::~ALPCABasic()
{
}
ALFloatMatrix* ALPCABasic::vTransform(const ALFloatMatrix* origin) const
{
    return ALFloatMatrix::productT(origin, mTransform.get());
}
