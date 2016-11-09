#ifndef DATA_ALVARYARRAY_H
#define DATA_ALVARYARRAY_H
#include "ALHead.h"
#include "utils/ALStream.h"
#include <vector>
class ALVaryArray : public ALRefCount
{
    public:
        virtual ~ALVaryArray();
        static ALVaryArray* create(ALStream* read);
        size_t size() const {return mArray.size();}
        struct Array
        {
            const ALINT* c;
            size_t length;
        };
        const Array& getArray(size_t index) const;
    private:
        ALVaryArray();
        void addArray(ALINT* v, size_t length);
        std::vector<Array> mArray;
        std::vector<ALINT*> mContent;
};
#endif
