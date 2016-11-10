#ifndef DATA_ALVARYARRAYMATRIX_H
#define DATA_ALVARYARRAYMATRIX_H
#include "ALVaryArray.h"
#include "math/ALFloatMatrix.h"
class ALVaryArrayMatrix : public ALFloatMatrix
{
    public:
        ALVaryArrayMatrix(const ALVaryArray* array, size_t time, const ALFloatMatrix* label);
        virtual ~ ALVaryArrayMatrix();
        virtual ALFLOAT* vGetAddr(size_t y) const override;
    private:
        void _refreshCache() const;
        mutable size_t mCur = 0;
        mutable ALFLOAT* mCache = NULL;
        size_t mTime = 0;
        const ALVaryArray* mArray;
        const ALFloatMatrix* mLabel;
};
#endif
