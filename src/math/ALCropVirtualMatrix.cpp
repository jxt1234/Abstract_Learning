//
//  ALCropVirtualMatrix.cpp
//  abs
//
//  Created by jiangxiaotang on 15/9/27.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//

#include "ALCropVirtualMatrix.h"
ALCropVirtualMatrix::ALCropVirtualMatrix(const ALFloatMatrix* basic, size_t l, size_t t, size_t r, size_t b):ALFloatMatrix(r-l+1, b-t+1, basic->stride())
{
    mL = l;
    mT = t;
    mBasic = basic;
}
ALCropVirtualMatrix::~ALCropVirtualMatrix()
{
}
ALFLOAT* ALCropVirtualMatrix::vGetAddr(size_t y) const
{
    return mBasic->vGetAddr(y+mT)+mL;
}
