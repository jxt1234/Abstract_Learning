#ifndef CORE_ALARSTRUCTURE_H
#define CORE_ALARSTRUCTURE_H
#include "ALHead.h"
class ALARStructure:public ALRefCount
{
    public:
        int l;//The length of ar for predict
        int w;//Selece x0, x1, ..., x(w-1) for predict
        int c;//whether has constant value
        int d;//Delayed
        ALARStructure():l(1),w(1),c(1),d(0){}
        ALARStructure(const ALARStructure& ar) = default;
        ALARStructure& operator=(const ALARStructure& ar) = default;
        ~ALARStructure(){}
};


#endif
