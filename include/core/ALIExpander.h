#ifndef INCLUDE_CORE_ALIEXPANDER_H
#define INCLUDE_CORE_ALIEXPANDER_H
#include "ALFloatData.h"
class ALFloatMatrix;
class ALFloatDataChain;
//TODO Support other type of data
class ALIExpander:public ALRefCount
{
public:
    ALIExpander(){}
    virtual ~ALIExpander(){}
    //Interface
    virtual int vLength() const = 0;
    virtual bool vExpand(const ALFloatData* d, ALFLOAT* dst) const = 0;
    virtual void vPrint(std::ostream& os) const {}
    static void expandXY(const ALIExpander* Xe, const ALLabeldData* data, ALSp<ALFloatMatrix> &X, ALSp<ALFloatMatrix> &YT);
};
#endif
