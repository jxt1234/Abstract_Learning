#ifndef MATH_ALIGRADIENTDECENT_H
#define MATH_ALIGRADIENTDECENT_H
#include "ALHead.h"
#include "ALFloatMatrix.h"
class ALIGradientDecent : public ALRefCount
{
public:
    class DerivativeFunction : public ALRefCount
    {
    public:
        /*Input:
         coefficient: Initialized coefficient Matrix
         X: the matrix to update coefficient
         
         Output:
         detCoefficient, a matrix that has same width and height with coefficient
         The output should be deleted after used
         */
        virtual ALFloatMatrix* vCompute(ALFloatMatrix* coefficient, const ALFloatMatrix* X) const = 0;
        
        virtual ~ DerivativeFunction() {}
    protected:
        DerivativeFunction(){}
    };
    virtual void vOptimize(ALFloatMatrix* coefficient, const ALFloatMatrix* X, const DerivativeFunction* delta, double alpha, int iteration) const = 0;
    
    virtual ~ ALIGradientDecent(){}
    
    typedef enum
    {
        FULL,
        SGD
    } TYPE;
    
    static ALIGradientDecent* create(TYPE t = FULL, int batchSize = 50);
protected:
    ALIGradientDecent(){}
};
#endif
