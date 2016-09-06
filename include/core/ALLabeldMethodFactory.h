#ifndef INCLUDE_CORE_ALLABELDMETHODFACTORY_H
#define INCLUDE_CORE_ALLABELDMETHODFACTORY_H
#include "ALILabeldMethod.h"
class ALLabeldMethodFactory
{
    public:
        static ALILabeldMethod* createBasic();
        static ALLabeldData* delayLabel(const std::vector<const ALFloatData*>& data, const ALILabeldMethod* basic, int delay);
};
#endif
