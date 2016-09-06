#include "core/ALExpanderFactory.h"
#include "core/ALBasicExpander.h"
#include "core/ALNormalizer.h"

ALIExpander* ALExpanderFactory::createAR(const ALARStructure& ar)
{
    return new ALARExpander(ar);
}

ALIExpander* ALExpanderFactory::createY()
{
    return new ALYExpander;
}

ALIExpander* ALExpanderFactory::normalize(ALIExpander* base, const ALFloatDataChain* c)
{
    return new ALNormalizer(c->get(), base);
}
