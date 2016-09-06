#ifndef SRC_MATH_ALLARGEMATRIX_H
#define SRC_MATH_ALLARGEMATRIX_H
//
//  ALLargeMatrix.h
//  abs
//
//  Created by jiangxiaotang on 15/7/13.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//
#ifndef abs_ALLargeMatrix_h
#define abs_ALLargeMatrix_h
#include "math/ALFloatMatrix.h"
#include <vector>
class ALLargeMatrix:public ALFloatMatrix
{
public:
    ALLargeMatrix(ALSp<ALFloatMatrix> matrix);
    virtual ~ALLargeMatrix();
    virtual ALFLOAT* vGetAddr(size_t y) const;
    void addMatrix(ALSp<ALFloatMatrix> matrix);
private:
    /*Offset : Matrix*/
    std::vector<size_t> mOffsets;
    std::vector<ALSp<ALFloatMatrix> > mMatrixs;
    
    /*Cache for Search*/
    mutable size_t mCurrentSta;
    mutable size_t mCurrentFin;
    mutable ALFloatMatrix* mCurrentMatrix;
};
#endif
#endif
