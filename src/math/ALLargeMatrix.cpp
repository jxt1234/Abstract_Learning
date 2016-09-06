//
//  ALLargeMatrix.cpp
//  abs
//
//  Created by jiangxiaotang on 15/7/13.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//

#include <stdio.h>
#include "ALLargeMatrix.h"

ALLargeMatrix::ALLargeMatrix(ALSp<ALFloatMatrix> m):ALFloatMatrix(m->width(), m->height())
{
    mMatrixs.push_back(m);
    mOffsets.push_back(m->height());
    mCurrentMatrix = m.get();
    mCurrentSta = 0;
    mCurrentFin = m->height();
}

ALLargeMatrix::~ALLargeMatrix()
{
    mMatrixs.clear();
    mOffsets.clear();
}

ALFLOAT* ALLargeMatrix::vGetAddr(size_t y) const
{
    if (mCurrentFin > y && mCurrentSta <= y)
    {
        return mCurrentMatrix->vGetAddr(y-mCurrentSta);
    }
    size_t cur = 0;
    for (size_t i=0; i<mOffsets.size(); ++i)
    {
        if (mOffsets[i] > y)
        {
            cur = i;
            break;
        }
    }
    mCurrentFin = mOffsets[cur];
    mCurrentMatrix = mMatrixs[cur].get();
    mCurrentSta = mCurrentFin-mCurrentMatrix->height();
    return mCurrentMatrix->vGetAddr(y-mCurrentSta);
}
void ALLargeMatrix::addMatrix(ALSp<ALFloatMatrix> matrix)
{
    ALASSERT(NULL!=matrix.get());
    ALASSERT(matrix->width() == width());
    ALASSERT(mOffsets.size() == mMatrixs.size());
    mHeight += matrix->height();
    auto lastoffset = mOffsets[mOffsets.size()-1];
    auto lasth = matrix->height();
    mOffsets.push_back(lastoffset + lasth);
    mMatrixs.push_back(matrix);
}
