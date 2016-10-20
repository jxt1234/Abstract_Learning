#ifndef INCLUDE_ALHEAD_H
#define INCLUDE_ALHEAD_H
#include <assert.h>
#include <string.h>
#include "utils/ALDebug.h"
#include "utils/ALAutoStorage.h"
#include "utils/ALRefCount.h"
#include "utils/ALSp.h"
#include "utils/ALClock.h"
#include "utils/ALDefer.h"
#include "utils/ALRandom.h"
#include "utils/ALAutoFile.h"
#include "utils/ALStream.h"

#define ZERO(x) (0.0001>(x) && -0.0001 < (x))
#define NOTNAN(x) ((x)>0 || (x)<=0)
#ifdef ALSPEEDFIRST
typedef float ALFLOAT;
#else
typedef double ALFLOAT;
#endif

#endif
