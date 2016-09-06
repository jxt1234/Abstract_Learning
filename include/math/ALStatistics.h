#ifndef INCLUDE_MATH_ALSTATISTICS_H
#define INCLUDE_MATH_ALSTATISTICS_H
#include "ALFloatMatrix.h"
class ALStatistics
{
public:
    template<typename T>
    static size_t max(T* t, size_t n)
    {
        ALASSERT(NULL!=t && n>0);
        T maxT = t[0];
        size_t p = 0;
        for (size_t i=1; i<n; ++i)
        {
            if (t[i] > maxT)
            {
                maxT = t[i];
                p = i;
            }
        }
        return p;
    }
    template<typename T>
    static size_t min(T* t, size_t n)
    {
        ALASSERT(NULL!=t && n>0);
        T minT = t[0];
        size_t p = 0;
        for (size_t i=1; i<n; ++i)
        {
            if (t[i] < minT)
            {
                minT = t[i];
                p = i;
            }
        }
        return p;
    }
    /*First line is mean, Second is min, Third is max*/
    static ALFloatMatrix* statistics(const ALFloatMatrix* X);

    /*XT is l*n matrix, l is the length of whole data, n is the number of component, usally, l is much larger than n*/
    static ALFloatMatrix* covariance(const ALFloatMatrix* XT);
    /*X is l*l matrix, Return l characteristic_root, the roots may be repeated*/
    static ALFloatMatrix* characteristic_root(const ALFloatMatrix* X);
    /*Solve characteristic vector at the same time*/
    static void characteristic_compute(const ALFloatMatrix* X, ALSp<ALFloatMatrix> &Root, ALSp<ALFloatMatrix> &Vector);
    /*normalize x by x = (x-min)/(max-min)*/
    static ALFloatMatrix* normalize(const ALFloatMatrix* X);
};
#endif
