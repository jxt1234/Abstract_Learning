#include <iostream>
#include "math/ALPolynomial.h"
#include "ALHead.h"
#include "math/ALFloatMatrix.h"
#define VALID(M)\
    ALASSERT(NULL!=(M));ALASSERT((M)->height()==1);
ALFloatMatrix* ALPolynomial::det(const ALFloatMatrix* P)
{
    VALID(P);
    int w = P->width()-1;
    int resW = w>0?w:1;
    ALFloatMatrix* result = ALFloatMatrix::create(resW, 1);
    ALFLOAT* r = result->vGetAddr(0);
    ALFLOAT* p = P->vGetAddr(0);
    r[0] = 0.0;
    for (int i=0; i<w; ++i)
    {
        r[i] = (i+1)*p[i+1];
    }
    return result;
}
ALFloatMatrix* ALPolynomial::divide(const ALFloatMatrix* Y, const ALFloatMatrix* X)
{
    VALID(Y);
    VALID(X);
    int yw = Y->width();
    int xw = X->width();
    int diff = yw-xw+1;
    int w = diff>0?diff:1;
    ALFloatMatrix* res = ALFloatMatrix::create(w, 1);
    ALFLOAT* r = res->vGetAddr();
    r[0] = 0.0;
    if (diff<1)
    {
        return res;
    }
    ALFLOAT* y = Y->vGetAddr();
    for (int i=0; i<diff; ++i)
    {
        r[diff-i-1] = y[yw-i-1];
    }
    ALFLOAT* x = X->vGetAddr();
    for (int i=0; i<diff; ++i)
    {
        ALFLOAT p = r[diff-i-1]/x[xw-1];
        int rlen = diff-i-1;
        int len = rlen>xw?xw:rlen;
        for (int j=1;j<=len;++j)
        {
            r[diff-i-1-j]-=p*x[xw-1-j];
        }
        r[diff-i-1] = p;
    }
    return res;
}
ALFloatMatrix* ALPolynomial::multi(const ALFloatMatrix* X1, const ALFloatMatrix* X2)
{
    VALID(X1);
    VALID(X2);
    int w1 = X1->width();
    int w2 = X2->width();
    int w = (w1-1)+(w2-1)+1;
    ALFloatMatrix* res = ALFloatMatrix::create(w, 1);
    ALFLOAT* x1 = X1->vGetAddr();
    ALFLOAT* x2 = X2->vGetAddr();
    ALFLOAT* r = res->vGetAddr();
    ::memset(r, 0, sizeof(ALFLOAT)*w);
    for (int i=0; i<w1; ++i)
    {
        for (int j=0; j<w2; ++j)
        {
            r[i+j] += (x1[i]*x2[j]);
        }
    }
    return res;
}

ALFloatMatrix* ALPolynomial::solve(const ALFloatMatrix* P)
{
    VALID(P);
    int rootnumber = P->width()-1;
    ALASSERT(rootnumber>0);
    ALSp<ALFloatMatrix> current = construct(P->vGetAddr(), P->width());
    ALFloatMatrix* result = ALFloatMatrix::create(rootnumber, 1);
    ALFLOAT* r = result->vGetAddr();
    ALSp<ALFloatMatrix> rootMatrix = ALFloatMatrix::create(2, 1);
    ALFLOAT* root = rootMatrix->vGetAddr();
    root[1] = 1;
    for (int i=0; i<rootnumber; ++i)
    {
        r[i] = NewTonSolve(current.get());
        root[0] = -r[i];
        current = divide(current.get(), rootMatrix.get());
    }
    return result;
}

ALFloatMatrix* ALPolynomial::construct(const ALFLOAT* p, int n)
{
    ALASSERT(n>0);
    ALFloatMatrix* res = ALFloatMatrix::create(n, 1);
    ALFLOAT* r = res->vGetAddr();
    ::memcpy(r, p, n*sizeof(ALFLOAT));
    return res;
}

ALFLOAT ALPolynomial::NewTonSolve(const ALFloatMatrix* X)
{
    VALID(X);
    ALASSERT(X->width()>1);
    if (X->width()==2)
    {
        ALFLOAT* x = X->vGetAddr();
        return -x[0]/x[1];
    }
    ALSp<ALFloatMatrix> detX = det(X);
    ALFLOAT x0 = 0.0;
    /*Move x0 until X'|x=x0 != 0*/
    const int maxTestTime = 100;
    for (int i=0; i<maxTestTime; ++i)
    {
        if (!ZERO(compute(detX.get(), x0)))
        {
            x0+=0.123;
        }
    }
    const int maxIterTime = 100;
    for (int i=0; i<maxIterTime; ++i)
    {
        ALFLOAT x1 = x0 - compute(X, x0)/compute(detX.get(), x0);
        if (ZERO(x1-x0))
        {
            break;
        }
        x0 = x1;
    }
    return x0;
}

ALFLOAT ALPolynomial::compute(const ALFloatMatrix* P, ALFLOAT x)
{
    VALID(P);
    ALFLOAT* p = P->vGetAddr();
    int w = P->width();
    ALFLOAT res = p[w-1];
    for (int i=w-2; i>=0; --i)
    {
        res = res*x + p[i];
    }
    return res;
}
