//
//  ALAutoFile.cpp
//  abs
//
//  Created by jiangxiaotang on 15/7/13.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//

#include <stdio.h>
#include "ALHead.h"

ALAutoFile::ALAutoFile(const char* file, const char* mode)
{
    ALASSERT(NULL!=file);
    ALASSERT(NULL!=mode);
    mF = fopen(file, mode);
    ALASSERT(NULL!=mF);
}
ALAutoFile::~ALAutoFile()
{
    fclose(mF);
}
