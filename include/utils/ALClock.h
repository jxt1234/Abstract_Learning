#ifndef INCLUDE_UTILS_ALCLOCK_H
#define INCLUDE_UTILS_ALCLOCK_H
#include <time.h>
#include <string>
class ALClock
{
    public:
        ALClock(const char* name, int line)
        {
            mName = name;
            mLine = line;
            mSta = clock();
        }
        ~ALClock()
        {
            clock_t fin = clock();
            long interval = fin - mSta;
            printf("For %s, %d Time is %lums+%luus\n", mName.c_str(), mLine, interval/1000, interval%1000);
        }
    private:
        std::string mName;
        int mLine;
        clock_t mSta;
};
//#define ALAUTOTIME ALClock __alclock(__func__, __LINE__)
#define ALAUTOTIME

#define ALLEARNAUTOTIME
//#define ALLEARNAUTOTIME ALClock __alclock(__FILE__, __LINE__)
#define ALFORCEAUTOTIME ALClock __alclock(__func__, __LINE__)
#endif
