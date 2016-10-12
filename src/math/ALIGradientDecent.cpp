#include "math/ALIGradientDecent.h"
#include "gradientDecent/GradientDecent.h"
#include "gradientDecent/StochasticGradientDecent.h"

ALIGradientDecent* ALIGradientDecent::create(TYPE t, int batchSize)
{
    switch (t) {
        case FULL:
            return new GradientDecent;
        case SGD:
            return new StochasticGradientDecent(batchSize);
        default:
            ALASSERT(false);
            break;
    }
    ALASSERT(false);
    return NULL;
}
