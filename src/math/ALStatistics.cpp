#include <iostream>
#include <math.h>
#include <list>
#include "math/ALStatistics.h"
#include "math/ALPolynomial.h"
#include "math/ALFloatMatrix.h"
#include "ALHead.h"
static const int maxIter = 100;

ALFloatMatrix* ALStatistics::statistics(const ALFloatMatrix* X)
{
    ALAUTOTIME;
    ALASSERT(NULL!=X);
    ALASSERT(1 <= X->height());
    auto h = X->height();
    auto w = X->width();
    ALFloatMatrix* result = ALFloatMatrix::create(X->width(), 3);
    auto mean = result->vGetAddr(0);
    auto min = result->vGetAddr(1);
    auto max = result->vGetAddr(2);
    {
        auto x = X->vGetAddr(0);
        for (int i=0; i<w; ++i)
        {
            mean[i] = x[i];
            min[i] = x[i];
            max[i] = x[i];
        }
    }
    for (int i=1; i<h; ++i)
    {
        auto x = X->vGetAddr(i);
        for (int j=0; j<w; ++j)
        {
            mean[j] += x[j];
            min[j] = min[j] > x[j] ? x[j] : min[j];
            max[j] = max[j] > x[j] ? max[j] : x[j];
        }
    }
    for (int j=0; j<w; ++j)
    {
        mean[j]/= h;
    }
    return result;
}


ALFloatMatrix* ALStatistics::covariance(const ALFloatMatrix* XT)
{
    ALASSERT(NULL!=XT);
    ALASSERT(XT->width()>0 && XT->height()>0);
    auto l = XT->width();
    auto n = XT->height();
    ALFloatMatrix* C = ALFloatMatrix::create(n, n);
    ALAutoStorage<ALFLOAT> meansStorage(n);
    ALFLOAT* mean = meansStorage.get();
    for (size_t i=0; i<n; ++i)
    {
        ALFLOAT sum = 0;
        ALFLOAT* x = XT->vGetAddr(i);
        for (size_t j=0; j<l; ++j)
        {
            sum+=x[j];
        }
        mean[i] = sum / (ALFLOAT)l;
    }
    for (size_t i=0; i<n; ++i)
    {
        for (size_t j=i; j<n; ++j)
        {
            ALFLOAT sum = 0;
            ALFLOAT* x1 = XT->vGetAddr(i);
            ALFLOAT* x2 = XT->vGetAddr(j);
            ALFLOAT m1 = mean[i];
            ALFLOAT m2 = mean[j];
            for (int k=0; k<l; ++k)
            {
                sum += ((x1[k]-m1)*(x2[k]-m2));
            }
            sum = sum / (ALFLOAT)l;
            C->vGetAddr(j)[i] = sum;
            C->vGetAddr(i)[j] = sum;
        }
    }
    return C;
}

ALFloatMatrix* constructHouseholder(const ALFloatMatrix* X)
{
    ALASSERT(NULL!=X);
    ALASSERT(X->height()==1);
    auto n = X->width();
    ALFLOAT norm = ALFloatMatrix::norm(X);
    ALSp<ALFloatMatrix> W = ALFloatMatrix::create(n, 1);
    ALFLOAT* w = W->vGetAddr();
    ALFLOAT* x = X->vGetAddr();
    ::memcpy(w, x, n*sizeof(ALFLOAT));
    w[0] = (x[0]-norm);
    ALFLOAT wnorm = ALFloatMatrix::norm(W.get());
    for (size_t i=0; i<n; ++i)
    {
        w[i] = w[i]/wnorm;
    }
    ALFloatMatrix* R = ALFloatMatrix::create(n, n);
    for (size_t i=0; i<n; ++i)
    {
        auto r = R->vGetAddr(i);
        for (size_t j=0; j<n; ++j)
        {
            r[j] = - 2*w[i]*w[j];
        }
        r[i] = r[i] + 1.0;
    }
    return R;
}

static ALFloatMatrix* TurnHessenberg(const ALFloatMatrix* A, ALSp<ALFloatMatrix> &HStore)
{
    ALAUTOTIME;
    ALASSERT(NULL!=A);
    ALASSERT(A->width() == A->height());
    ALASSERT(A->width()>3);
    auto n = A->width();
    ALSp<ALFloatMatrix> HAH;
    const ALFloatMatrix* res = A;
    for (int i=1; i<n-1; ++i)
    {
        ALSp<ALFloatMatrix> X = ALFloatMatrix::create(n-i, 1);
        ALFLOAT* x = X->vGetAddr();
        ALFLOAT* a = res->vGetAddr(i-1)+i;
        for (int j=0; j<n-i; ++j)
        {
            x[j] = a[j];
        }
        ALSp<ALFloatMatrix> H = constructHouseholder(X.get());
        HAH = ALFloatMatrix::HAH(res, H.get());
        res = HAH.get();
        if (NULL!=HStore.get())
        {
            H = ALFloatMatrix::enlarge(n, H.get());
            HStore = ALFloatMatrix::product(HStore.get(), H.get());
        }
    }
    HAH->addRef();
    return HAH.get();
}

static ALSp<ALFloatMatrix> QROneStep(ALSp<ALFloatMatrix> X, ALSp<ALFloatMatrix>& VS)
{
    ALAUTOTIME;
    ALASSERT(NULL!=X.get());
    ALASSERT(X->width() == X->height());
    int n = X->width();
    VS = ALFloatMatrix::createIdentity(n);
    for (int i=0; i<n-1; ++i)
    {
        ALFLOAT x1 = *(X->vGetAddr(i)+i);
        ALFLOAT x2 = *(X->vGetAddr(i+1)+i);
        ALFLOAT r = sqrt(x1*x1+x2*x2);
        /*equal to this:
          x(i) = x(i) * x1/r + x(i+1)*x2/r
          x(i+1) = r/x1*x(i+1) - x2/x1*x(i)
         */
        auto xi = X->vGetAddr(i);
        auto xj = X->vGetAddr(i+1);
        for (auto k=i; k<n; ++k)
        {
            xi[k] = xi[k]*x1/r + xj[k]*x2/r;
            xj[k] = xj[k]*r/x1 - x2/x1*xi[k];
        }
        /*The Same for VS, but it's column transformation*/
        for (auto k=0; k<n; ++k)
        {
            auto v1 = VS->vGetAddr(k)+(i);
            auto v2 = VS->vGetAddr(k)+(i+1);
            v1[0] = v1[0]*x1/r + v2[0]*x2/r;
            v2[0] = v2[0]*r/x1 - x2/x1*v1[0];
        }
    }
    X = ALFloatMatrix::product(X.get(), VS.get());
    return X;
}
/*It's assume that X is Hesenberg one */
static bool isUpTrianger(const ALFloatMatrix* X)
{
    ALASSERT(NULL!=X);
    ALASSERT(X->width()==X->height());
    auto n = X->width();
    auto f = [](ALFLOAT x){return x<-0.0001 ||x>0.0001;};
    for (int i=0; i<n-1; ++i)
    {
        ALFLOAT x = *(X->vGetAddr(i+1)+i);
        if (f(x))
        {
            return false;
        }
    }
    return true;
}
ALFloatMatrix* ALStatistics::characteristic_root(const ALFloatMatrix* X)
{
    ALASSERT(NULL!=X);
    ALASSERT(X->width()==X->height());
    int n = X->width();
    if (X->width()<=3)
    {
        /*TODO*/
        return NULL;
    }
    ALSp<ALFloatMatrix> VS;
    ALSp<ALFloatMatrix> H = TurnHessenberg(X, VS);
    //ALFloatMatrix::print(H.get(), std::cout);
    ALSp<ALFloatMatrix> QR = QROneStep(H, VS);
    for (int i=0; i<maxIter; ++i)
    {
        if (isUpTrianger(QR.get()))
        {
            break;
        }
        QR = QROneStep(QR, VS);
    }
    ALFloatMatrix* Root = ALFloatMatrix::create(n, 1);
    ALFLOAT* r = Root->vGetAddr();
    for (int i=0; i<n; ++i)
    {
        r[i] = *(QR->vGetAddr(i)+i);
    }
    return Root;
}
void ALStatistics::characteristic_compute(const ALFloatMatrix* X, ALSp<ALFloatMatrix> &Root, ALSp<ALFloatMatrix> &Vector)
{
    ALASSERT(NULL!=X);
    ALASSERT(X->width()==X->height());
    auto n = X->width();
    ALSp<ALFloatMatrix> HS = ALFloatMatrix::createIdentity(n);
    ALSp<ALFloatMatrix> VS;
    ALSp<ALFloatMatrix> H = TurnHessenberg(X, HS);
    ALSp<ALFloatMatrix> QR = QROneStep(H, VS);
    HS = ALFloatMatrix::product(HS.get(), VS.get());
    for (int i=0; i<maxIter; ++i)
    {
        if (isUpTrianger(QR.get()))
        {
            break;
        }
        QR = QROneStep(QR, VS);
        HS = ALFloatMatrix::product(HS.get(), VS.get());
    }
    Root = ALFloatMatrix::create(n, 1);
    ALFLOAT* r = Root->vGetAddr();
    for (int i=0; i<n; ++i)
    {
        r[i] = *(QR->vGetAddr(i)+i);
    }
    Vector = HS;
}
ALFloatMatrix* ALStatistics::normalize(const ALFloatMatrix* X)
{
    ALASSERT(NULL!=X);
    auto w = X->width();
    auto h = X->height();
    /*Compute min and max*/
    ALAutoStorage<ALFLOAT> _min(w);
    ALAutoStorage<ALFLOAT> _max(w);
    ALFLOAT* mins = _min.get();
    ALFLOAT* maxs = _max.get();
    ::memcpy(mins, X->vGetAddr(), w*sizeof(ALFLOAT));
    ::memcpy(maxs, X->vGetAddr(), w*sizeof(ALFLOAT));
    for (auto i=1; i<h; ++i)
    {
        ALFLOAT* x = X->vGetAddr(i);
        for (auto j=0; j<w; ++j)
        {
            if (mins[j] > x[j])
            {
                mins[j] = x[j];
            }
            if (maxs[j] < x[j])
            {
                maxs[j] = x[j];
            }
        }
    }
    ALFloatMatrix* N = ALFloatMatrix::create(w, h);
    /*Pretreat max = max - min*/
    for (auto j=0; j<w; ++j)
    {
        maxs[j] -= mins[j];
    }
    for (auto i=0; i<h; ++i)
    {
        ALFLOAT* n = N->vGetAddr(i);
        ALFLOAT* x = X->vGetAddr(i);
        for (auto j=0; j<w; ++j)
        {
            n[j] = (x[j]-mins[j])/maxs[j];
        }
    }
    return N;
}
