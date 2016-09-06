#ifndef INCLUDE_LEARN_ALSVMKERNELFACTORY_H
#define INCLUDE_LEARN_ALSVMKERNELFACTORY_H
#include "ALSVM.h"
class ALSVMKernelFactory
{
    public:
        static ALSp<ALSVM::Kernel> create(const std::map<std::string, std::string>& heads);
        static ALSp<ALSVM::Kernel> createRBF(ALFLOAT gamma);
};
#endif
