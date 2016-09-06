#ifndef INCLUDE_CORE_ALBASICEXPANDER_H
#define INCLUDE_CORE_ALBASICEXPANDER_H
#include "ALIExpander.h"
#include "ALARStructure.h"
#include "ALHead.h"
class ALYExpander:public ALIExpander
{
    public:
        ALYExpander(){}
        ~ALYExpander(){}
        virtual int vLength() const {return 1;}
        virtual bool vExpand(const ALFloatData* d, ALFLOAT* dst) const
        {
            *dst = d->value(0);
            return true;
        }
};
class ALARExpander:public ALIExpander
{
    public:
        ALARExpander(const ALARStructure& ar);
        ~ALARExpander();
        virtual int vLength() const {return mL;}
        virtual bool vExpand(const ALFloatData* d, ALFLOAT* dst) const;
        virtual void vPrint(std::ostream& os) const;
    private:
        int mL;
        ALARStructure mAR;
};
#endif
