#ifndef INCLUDE_UTILS_ALCLOCK_H
#define INCLUDE_UTILS_ALCLOCK_H
#include <time.h>
#include <string>
#include <sys/time.h>
#include "ALDebug.h"
class ALClock
{
public:
    ALClock(const char* name, int line);
    ~ALClock();
private:
    timeval mSta;
    timeval mFin;
    std::string mName;
    int mLine;
};
//#define ALAUTOTIME ALClock __alclock(__func__, __LINE__)
#define ALAUTOTIME

#define ALLEARNAUTOTIME
//#define ALLEARNAUTOTIME ALClock __alclock(__FILE__, __LINE__)
#define ALFORCEAUTOTIME ALClock __alclock(__func__, __LINE__)
#endif
