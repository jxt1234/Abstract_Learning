#include "math/ALGradientDown.h"
ALFLOAT ALGradientDown(std::function<ALFLOAT(ALFLOAT)> f, std::function<ALFLOAT(ALFLOAT)> detf, ALFLOAT start, ALFLOAT alpha, size_t maxIter)
{
    ALFLOAT after = start;
    for (size_t i = 0; i <maxIter; ++i)
    {
        after = start - alpha*f(start)/detf(start);
        if (ZERO((after-start)))
        {
            break;
        }
    }
    return after;
}
