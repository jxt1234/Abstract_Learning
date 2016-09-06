#ifndef INCLUDE_CORE_ALBASEFLOATMATRIX_H
#define INCLUDE_CORE_ALBASEFLOATMATRIX_H
#include "math/ALFloatMatrix.h"
class ALBaseFloatMatrix:public ALFloatMatrix
{
    public:
        ALBaseFloatMatrix(size_t w, size_t h);
        virtual ~ALBaseFloatMatrix();
        virtual ALFLOAT* vGetAddr(size_t y=0) const;
        static int number() {return gNum;}
    private:
        ALFLOAT* mBase;
        static int gNum;
};
#endif
