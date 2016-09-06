#ifndef INCLUDE_MATH_ALGRADIENTDOWN_H
#define INCLUDE_MATH_ALGRADIENTDOWN_H
#include "ALHead.h"
#include <functional>
ALFLOAT ALGradientDown(std::function<ALFLOAT(ALFLOAT)> f, std::function<ALFLOAT(ALFLOAT)> detf, ALFLOAT start, ALFLOAT alpha=1.0, size_t maxIter=1000);
#endif
