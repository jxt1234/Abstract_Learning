//
//  ALOpenCL.cpp
//  opencl
//
//  Created by jiangxiaotang on 15/3/7.
//  Copyright (c) 2015年 jiangxiaotang. All rights reserved.
//

#include <assert.h>
#include <thread>
#include <string.h>
#include "ALHead.h"
#ifdef ALOPENCL_MAC
#include "opencl/ALOpenCL.h"
ALOpenCL* ALOpenCL::gInstance = NULL;
static std::mutex gInstanceMutex;
/*TODO Add Lock*/
ALOpenCL::ALOpenCL()
{
    // create a GPU context
    int type = CL_DEVICE_TYPE_GPU;
    mContext = clCreateContextFromType(NULL, type, NULL, NULL, NULL);
    ALASSERT(NULL!=mContext);
    if(mContext == 0) {
        printf("Can't create context of %d\n", type);
        return;
    }
    size_t deviceListSize;
    clGetContextInfo(mContext,CL_CONTEXT_DEVICES, 0,NULL,&deviceListSize);
    cl_device_id* devices = new cl_device_id[deviceListSize];
    ALDefer([=](){delete [] devices;});
    clGetContextInfo(mContext, CL_CONTEXT_DEVICES, deviceListSize, devices, NULL);
    mDeviceId = devices[0];
    mQueue = clCreateCommandQueue(mContext, mDeviceId, 0, NULL);
    ALASSERT(NULL!=mQueue);
}

ALOpenCL::~ALOpenCL()
{
    for (auto iter : mRecordedTable)
    {
        iter->release(mContext);
    }
    clReleaseCommandQueue(mQueue);
    clReleaseDevice(mDeviceId);
    clReleaseContext(mContext);
}

void ALOpenCL::destory()
{
    ALStartEnd(gInstanceMutex.lock(), [&](){gInstanceMutex.unlock();});
    if (NULL != gInstance)
    {
        delete gInstance;
        gInstance = NULL;
    }
}
cl_kernel ALOpenCL::compileAndBuild(const char* sourcecode, const char* kernalname, cl_context context, cl_device_id device)
{
    ALASSERT(NULL!=sourcecode);
    ALASSERT(NULL!=kernalname);
    size_t sourcesize[] = {strlen(sourcecode)};
    cl_int errercode = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(context, 1, &sourcecode, sourcesize, &errercode);
    ALASSERT(errercode == CL_SUCCESS);
    errercode = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (CL_SUCCESS != errercode)
    {
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* buffer = new char[len+1];
        ALDefer([=]{delete [] buffer;});
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer, &len);
        printf("Build log is %s\n", buffer);//TODO use FUNC_PRINT_ALL
        ALASSERT(CL_SUCCESS == errercode);
    }
    cl_kernel kernel = clCreateKernel(program, kernalname, &errercode);
    ALASSERT(errercode == CL_SUCCESS);
    errercode = clReleaseProgram(program);
    ALASSERT(errercode == CL_SUCCESS);
    return kernel;
}
ALOpenCL& ALOpenCL::getInstance()
{
    /*TODO Lock*/
    if (NULL == gInstance)
    {
        ALStartEnd(gInstanceMutex.lock(), [&](){gInstanceMutex.unlock();});
        if (NULL == gInstance)
        {
            gInstance = new ALOpenCL;
        }
    }
    return *gInstance;
}
/*TODO New a thread to handle the work*/
bool ALOpenCL::queueWork(std::function<bool(cl_context, cl_command_queue)> run)
{
    ALStartEnd(gInstanceMutex.lock(), [&](){gInstanceMutex.unlock();});
    return run(mContext, mQueue);
}

void ALOpenCL::flush(bool finish)
{
    if (finish)
    {
        clFinish(mQueue);
    }
    else
    {
        clFlush(mQueue);
    }
}
/*TODO New a thread to handle the work*/
bool ALOpenCL::prepare(ALOpenCL::PrepareWork* w)
{
    ALASSERT(NULL!=w);
    ALStartEnd(gInstanceMutex.lock(), [&](){gInstanceMutex.unlock();});
    if (mRecordedTable.find(w)!=mRecordedTable.end())
    {
        return true;
    }
    bool res = w->prepare(mContext, mDeviceId);
    if (res)
    {
        mRecordedTable.insert(w);
    }
    return res;
}
#endif
