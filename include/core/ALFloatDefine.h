#ifndef INCLUDE_CORE_ALFLOATDEFINE_H
#define INCLUDE_CORE_ALFLOATDEFINE_H
#define ZERO(x) (0.0001>(x) && -0.0001 < (x))
#define NOTNAN(x) ((x)>0 || (x)<=0)
#include "ALHead.h"
#ifdef ALSPEEDFIRST
typedef float ALFLOAT;
#else
typedef double ALFLOAT;
#endif
#endif
