//
//  ALCropVirtualMatrix.h
//  abs
//
//  Created by jiangxiaotang on 15/9/27.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//

#ifndef __abs__ALCropVirtualMatrix__H
#define __abs__ALCropVirtualMatrix__H
#include "math/ALFloatMatrix.h"
class ALCropVirtualMatrix:public ALFloatMatrix
{
public:
    /*The Life time of ALCropVirtualMatrix is little than basic!
     * It will be invalid after basic freed!
     */
    ALCropVirtualMatrix(const ALFloatMatrix* basic, size_t l, size_t t, size_t r, size_t b);
    virtual ~ALCropVirtualMatrix();
    virtual ALFLOAT* vGetAddr(size_t y=0) const;
private:
    const ALFloatMatrix* mBasic;
    size_t mL;
    size_t mT;
};

#endif /* defined(__abs__ALCropVirtualMatrix__) */
