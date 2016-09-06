#ifndef INCLUDE_LEARN_ALPCABASIC_H
#define INCLUDE_LEARN_ALPCABASIC_H
#include "math/ALIMatrixTransformer.h"
class ALPCABasic:public ALIMatrixTransformer
{
    public:
        virtual ALFloatMatrix* vTransform(const ALFloatMatrix* origin) const;
        /*X is n*l data and rate is 0~1.0*/
        ALPCABasic(const ALFloatMatrix* XT, ALFLOAT rate=0.965);
        virtual ~ALPCABasic();
    private:
        ALSp<ALFloatMatrix> mTransform;
};
#endif
