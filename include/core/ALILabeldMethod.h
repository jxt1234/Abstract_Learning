#ifndef INCLUDE_CORE_ALILABELDMETHOD_H
#define INCLUDE_CORE_ALILABELDMETHOD_H
#include "ALHead.h"
#include "ALFloatData.h"
class ALILabeldMethod:public ALRefCount
{
    public:
        virtual ALFLOAT vLabel(const ALFloatData* data, bool &success) const = 0;
        ALILabeldMethod(){}
        virtual ~ALILabeldMethod(){}
};
#endif
