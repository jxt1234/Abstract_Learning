//
//  ALOpenCLMatrix4DOp.hpp
//  abs
//
//  Created by jiangxiaotang on 19/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#ifndef ALOpenCLMatrix4DOp_hpp
#define ALOpenCLMatrix4DOp_hpp

#include <stdio.h>
#include "ALHead.h"
#include "ALBasicMatrix4DOp.h"

/*If Don't Support OpenCL, this op operate just like ALBasicMatrix4DOp*/
class ALOpenCLMatrix4DOp : public ALBasicMatrix4DOp
{
public:
    ALOpenCLMatrix4DOp(){}
    virtual ~ ALOpenCLMatrix4DOp(){}
#ifdef ALOPENCL_MAC
    virtual void vFilter(Matrix4D& dst, const Matrix4D& src, const Matrix4D& kernelData, int stride) const override;
    
    virtual void vDeterFilter(const Matrix4D& dstDiff, const Matrix4D& dst, const Matrix4D& src,  Matrix4D& srcDiff, const Matrix4D& kernelData, Matrix4D& kernelDataDiff, int stride) const override;
#endif
};

#endif /* ALOpenCLMatrix4DOp_hpp */
