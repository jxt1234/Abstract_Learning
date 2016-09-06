#ifndef INCLUDE_LEARN_ALMATRIXSELECTOR_H
#define INCLUDE_LEARN_ALMATRIXSELECTOR_H
#include "math/ALIMatrixTransformer.h"
#include <vector>
class ALMatrixSelector:public ALIMatrixTransformer
{
public:
    ALMatrixSelector(const std::vector<int> positions);
    virtual ~ALMatrixSelector();
    virtual ALFloatMatrix* vTransform(const ALFloatMatrix* origin) const;
    virtual void vPrint(std::ostream& output) const;
private:
    int mMaxPos;
    int* mPos;
    size_t mN;
};
#endif
