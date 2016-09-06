#ifndef INCLUDE_CORE_ALNORMALIZER_H
#define INCLUDE_CORE_ALNORMALIZER_H
#include "ALFloatDataChain.h"
#include "ALIExpander.h"
/*TODO Support other type of data*/
class ALNormalizer:public ALIExpander
{
    public:
        ALNormalizer(const std::vector<const ALFloatData*>& c, ALIExpander* base);
        ALNormalizer(ALFLOAT* k, ALFLOAT* b, ALIExpander* base);
        ~ALNormalizer();
        virtual int vLength() const{return mL;}
        virtual bool vExpand(const ALFloatData* d, ALFLOAT* dst) const;
        /*Reverse the dst from normalized value to origin value*/
        void reverse(ALFLOAT* dst);
    private:
        ALFLOAT* mK;
        ALFLOAT* mB;
        int mL;
        ALIExpander* mBase;
};
#endif
