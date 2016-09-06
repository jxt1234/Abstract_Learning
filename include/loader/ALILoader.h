#ifndef LOADER_ALILOADER_H
#define LOADER_ALILOADER_H
#include "ALHead.h"
#include "core/ALFloatMatrix.h"

class ALILoader:public ALRefCount
{
    public:
        ALFloatMatrix* vLoad(int time) const = 0;
        ALILoader(){}
        virtual ~ALILoader(){}
};


#endif
