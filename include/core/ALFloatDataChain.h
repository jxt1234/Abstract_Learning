#ifndef CORE_ALFLOATDATACHAIN_H
#define CORE_ALFLOATDATACHAIN_H

#include <vector>
#include "ALFloatData.h"

/*This class has no actual data and didn't free any memory for database*/
class ALFloatDataChain:public ALRefCount
{
    public:
        ALFloatDataChain(size_t num):mWidth(num){}
        virtual ~ALFloatDataChain();
        void add(ALFloatData* d);
        inline size_t size() const{return mSeries.size();}
        inline size_t width() const{return mWidth;}
        void expand(void* dst, int stride) const;
        inline const std::vector<const ALFloatData*>& get() const{return mSeries;}
    private:
        std::vector<const ALFloatData*> mSeries;
        /*Max size of each point*/
        size_t mWidth;
};

#endif
