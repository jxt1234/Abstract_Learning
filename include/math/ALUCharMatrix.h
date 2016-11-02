#ifndef MATH_ALUCHARMATRIX_H
#define MATH_ALUCHARMATRIX_H
#include "ALFloatMatrix.h"
class ALUCharMatrix : public ALRefCount
{
public:
    virtual ~ ALUCharMatrix();
    
    static ALUCharMatrix* create(const ALFloatMatrix* origin);
    
    
    size_t width() const {return mWidth;}
    size_t height() const {return mHeight;}
    
    unsigned char* getLine(size_t s) const;
    unsigned char getMaxNumber(size_t w) const;
    ALFLOAT* getTypes(size_t w) const;
    
    void print(std::ostream& output) const;
    
private:
    ALUCharMatrix(size_t w, size_t h);
    
    
    unsigned char* mContents;
    unsigned char* mMaxNumbers;
    size_t mWidth;
    size_t mHeight;
    
    ALSp<ALFloatMatrix> mRealValues;
};
#endif
