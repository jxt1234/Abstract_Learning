#ifndef INCLUDE_CORE_ALEXPANDERFACTORY_H
#define INCLUDE_CORE_ALEXPANDERFACTORY_H
#include "ALIExpander.h"
#include "ALFloatDataChain.h"
#include "ALARStructure.h"
class ALExpanderFactory
{
    public:
        static ALIExpander* createAR(const ALARStructure& ar);
        static ALIExpander* createY();
        static ALIExpander* normalize(ALIExpander* base, const ALFloatDataChain* c);
};
#endif
