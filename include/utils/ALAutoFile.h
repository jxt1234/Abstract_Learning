#ifndef INCLUDE_UTILS_ALAUTOFILE_H
#define INCLUDE_UTILS_ALAUTOFILE_H
//
//  ALAutoFile.h
//  abs
//
//  Created by jiangxiaotang on 15/7/13.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//
#include <stdio.h>
class ALAutoFile
{
public:
    ALAutoFile(const char* file, const char* mode);
    ~ALAutoFile();
    FILE* get() const {return mF;}
private:
    FILE* mF;
    
};
#define ALAUTOFILE(file, name, mode) ALAutoFile __##file(name, mode); FILE* file = __##file.get();
#endif
