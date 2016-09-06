#ifndef INCLUDE_LEARN_ALMATRIXNORMALIZER_H
#define INCLUDE_LEARN_ALMATRIXNORMALIZER_H

#include "math/ALIMatrixTransformer.h"
#include "ALMatrixSelector.h"

class ALMatrixNormalizer:public ALIMatrixTransformer
{
public:
    /*Turn x to 0,1 by (x-min)/(max-min)*/
    ALMatrixNormalizer(const ALFloatMatrix* train, ALFLOAT rate=0.0f/*Enlarge value, not used now*/);
    virtual ~ALMatrixNormalizer();
    virtual ALFloatMatrix* vTransform(const ALFloatMatrix* origin) const;
private:
    ALSp<ALFloatMatrix> mK;
    ALSp<ALFloatMatrix> mB;
    ALSp<ALIMatrixTransformer> mSelect;
};

#endif
