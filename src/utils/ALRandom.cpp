#include "utils/ALRandom.h"
#include <time.h>
#include <random>
namespace ALRandom
{
    static const int inc = 100000;
    static std::random_device rd;
    float rate()
    {
        int r = rd()%inc;
        float p = (float)r/(float)inc;
        return p;
    }

    bool init()
    {
        return true;
    }
    int mid(int min_, int max_)
    {
        if (min_ >= max_)
        {
            return min_;
        }
        int r = rd()%(max_-min_);
        return r+min_;
    }

    bool reset()
    {
        return true;
    }
};
