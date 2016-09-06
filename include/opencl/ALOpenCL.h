#ifndef INCLUDE_OPENCL_ALOPENCL_H
#define INCLUDE_OPENCL_ALOPENCL_H
//
//  ALOpenCL.h
//  opencl
//
//  Created by jiangxiaotang on 15/3/7.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//
#include <stdio.h>
#ifdef __APPLE__  
    #include <OpenCL/cl.h>  
#elif defined(__linux__)  
    #include <CL/cl.h>  
#endif
#include <set>
#include "ALHead.h"
#define KERNEL(...)#__VA_ARGS__
class ALOpenCL
{
public:
    struct PrepareWork
    {
        std::function<bool(cl_context, cl_device_id)> prepare;
        std::function<bool(cl_context)> release;
    };
    /*For user to use in PREPARE function*/
    static cl_kernel compileAndBuild(const char* sourcecode, const char* kernalname, cl_context context, cl_device_id device);
    
    static ALOpenCL& getInstance();
    static void destory();
    bool prepare(PrepareWork* work);
    /*The run function will be called immediately, but for opencl, if don't wait for the result, the sync should be done by user*/
    bool queueWork(std::function<bool(cl_context, cl_command_queue)> run);
    void flush(bool finish=false);
private:
    ALOpenCL();
    ~ALOpenCL();
    std::set<PrepareWork*> mRecordedTable;
    cl_context mContext;
    cl_command_queue mQueue;
    cl_device_id mDeviceId;
    static ALOpenCL* gInstance;
};
#endif
