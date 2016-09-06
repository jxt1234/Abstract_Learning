#ifndef INCLUDE_LEARN_ALSVMHANDLEFACTORY_H
#define INCLUDE_LEARN_ALSVMHANDLEFACTORY_H
#include "ALSVM.h"
class ALSVMHandleFactory
{
    public:
        static ALSp<ALSVM::Reportor> create(const std::map<std::string, std::string>& heads);
};
#endif
