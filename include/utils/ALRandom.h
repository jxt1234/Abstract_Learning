#ifndef INCLUDE_UTILS_ALRANDOM_H
#define INCLUDE_UTILS_ALRANDOM_H
namespace ALRandom
{
    //Return rand float number [0.0~1.0)
    float rate();
    //Return rand int [min_, max_)
    int mid(int min_, int max_);
    //init random seed
    bool init();
    bool reset();
};
#endif
