#include "utils/ALClock.h"

ALClock::ALClock(const char* name, int line)
{
    gettimeofday(&mSta, NULL);
    mName = name;
    mLine = line;
}

ALClock::~ ALClock()
{
    gettimeofday(&mFin, NULL);
    float inter_ms = (mFin.tv_usec-mSta.tv_usec)/1000.0 + (mFin.tv_sec-mSta.tv_sec)*1000.0;
    printf("For %s, %d Time is %.3fms\n", mName.c_str(), mLine, inter_ms);
}