#ifndef INCLUDE_MATH_ALPOLYNOMIAL_H
#define INCLUDE_MATH_ALPOLYNOMIAL_H
#include "ALFloatMatrix.h"
/*Y = [1,x,x^2,x^3,...,x^n]*P'*/
/*Only handle 1*n matrix*/
class ALPolynomial
{
    public:
        static ALFLOAT compute(const ALFloatMatrix* P, ALFLOAT x);
        static ALFloatMatrix* construct(const ALFLOAT* p, int n);
        static ALFloatMatrix* det(const ALFloatMatrix* P);//return dP/dx
        static ALFloatMatrix* divide(const ALFloatMatrix* Y, const ALFloatMatrix* X);//return Y/X
        static ALFloatMatrix* multi(const ALFloatMatrix* X1, const ALFloatMatrix* X2);//return X1*X2
        //Solve XP' = 0, return all root which may has repeat value
        static ALFloatMatrix* solve(const ALFloatMatrix* P);
        /*Return one root of x, can't solve a+bi root*/
        static ALFLOAT NewTonSolve(const ALFloatMatrix* x);
};
#endif
