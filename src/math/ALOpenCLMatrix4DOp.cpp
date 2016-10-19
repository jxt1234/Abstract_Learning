//
//  ALOpenCLMatrix4DOp.cpp
//  abs
//
//  Created by jiangxiaotang on 19/10/2016.
//  Copyright © 2016 jiangxiaotang. All rights reserved.
//

#include "ALOpenCLMatrix4DOp.hpp"

#ifdef ALOPENCL_MAC
#include "opencl/ALOpenCL.h"

static cl_mem _uploadMatrix(cl_context context, cl_command_queue queue, const ALFloatMatrix* M)
{
    ALASSERT(NULL!=M);
    ALASSERT(sizeof(ALFLOAT) == sizeof(float));
    auto w = M->width();
    auto h = M->height();
    auto sizea = sizeof(float)*w*h;
    cl_int errorcode;
    auto flag = CL_MEM_READ_ONLY;
    cl_mem ma = clCreateBuffer(context, flag, sizea, NULL, &errorcode);
    ALASSERT(CL_SUCCESS == errorcode);
    errorcode = clEnqueueWriteBuffer(queue, ma, true, 0, sizea, M->vGetAddr()/*FIXME*/, 0, NULL, NULL);
    ALASSERT(CL_SUCCESS == errorcode);
    return ma;
}

static cl_mem _allocForWrite(cl_context context, cl_command_queue queue, size_t w, size_t h)
{
    ALASSERT(sizeof(ALFLOAT) == sizeof(float));
    auto sizea = sizeof(float)*w*h;
    cl_int errorcode;
    auto flag = CL_MEM_WRITE_ONLY;
    cl_mem ma = clCreateBuffer(context, flag, sizea, NULL, &errorcode);
    ALASSERT(CL_SUCCESS == errorcode);
    return ma;
}

static void _downloadMatrix(cl_command_queue queue, cl_mem mem, ALFloatMatrix* dst)
{
    auto sizec = dst->width()*dst->height()*sizeof(float);
    clEnqueueReadBuffer(queue, mem, true, 0, sizec, dst->vGetAddr()/*FIXME*/, 0, NULL, NULL);
}

static const char* gFilterSource = KERNEL(
                                          __kernel void filter(__global float *input, __global float* kernelM, __global float* output, size_t kw, size_t kh, size_t kd, size_t stride, size_t iw, size_t ih, size_t od)
                                          {
                                              int x = get_global_id(0);
                                              int y = get_global_id(1);
                                              int z = get_global_id(2);
                                              int ow = get_global_size(0);
                                              int oh = get_global_size(1);
                                              float sum = 0.0;
                                              int oi,i,j,k;
                                              int kbatchsize = kw*kh*kd+1;
                                              int inputBatchSize = iw*ih*kd;
                                              int outputBatchSize = ow*oh*od;
                                              __global float* input_base = input + inputBatchSize*z + x*stride + y*stride*iw;
                                              __global float* input_unit;
                                              __global float* kernel_base;
                                              __global float* output_base = output + outputBatchSize*z + x + y*ow;
                                              for (oi=0; oi<od; ++oi)
                                              {
                                                  kernel_base = kernelM+kbatchsize*oi;
                                                  sum = 0.0;
                                                  for (i=0;i<kd;++i)
                                                  {
                                                      for (j=0;j<kh; ++j)
                                                      {
                                                          for (k=0;k<kw;++k)
                                                          {
                                                              sum += input_base[k+j*iw+i*iw*ih]*kernel_base[k+j*kw+i*kw*kh];
                                                          }
                                                      }
                                                  }
                                                  sum += kernel_base[kbatchsize-1];
                                                  *(output_base + ow*oh*oi) = sum;
                                              }
                                          }
                                          );

void ALOpenCLMatrix4DOp::vFilter(Matrix4D& dst, const Matrix4D& src, const Matrix4D& kernelData, int stride) const
{
    static cl_kernel gKernel = NULL;
    static ALOpenCL::PrepareWork gPrepare = {
        [&](cl_context context, cl_device_id device) {
            gKernel = ALOpenCL::compileAndBuild(gFilterSource, "filter", context, device);
            return true;
        },
        [&](cl_context c){
            clReleaseKernel(gKernel);
            return true;
        }
    };
    auto run = [&](cl_context context, cl_command_queue queue)
    {
        auto input_gpu = _uploadMatrix(context, queue, src.pOrigin);
        auto kernel_gpu = _uploadMatrix(context, queue, kernelData.pOrigin);
        auto output_gpu = _allocForWrite(context, queue, dst.pOrigin->width(), dst.pOrigin->height());
        
        cl_int errorcode;
        //ALASSERT(CL_SUCCESS == errorcode);
        ALDefer([&](){clReleaseMemObject(input_gpu);clReleaseMemObject(kernel_gpu);clReleaseMemObject(output_gpu);});
        {
            size_t l;
            errorcode = clSetKernelArg(gKernel, 0, sizeof(cl_mem), &input_gpu);
            ALASSERT(errorcode == CL_SUCCESS);
            errorcode = clSetKernelArg(gKernel, 1, sizeof(cl_mem), &kernel_gpu);
            ALASSERT(errorcode == CL_SUCCESS);
            errorcode = clSetKernelArg(gKernel, 2, sizeof(cl_mem), &output_gpu);
            ALASSERT(errorcode == CL_SUCCESS);
            l = kernelData.iWidth;
            errorcode = clSetKernelArg(gKernel, 3, sizeof(size_t), &l);
            ALASSERT(errorcode == CL_SUCCESS);
            l = kernelData.iHeight;
            errorcode = clSetKernelArg(gKernel, 4, sizeof(size_t), &l);
            ALASSERT(errorcode == CL_SUCCESS);
            l = kernelData.iDepth;
            errorcode = clSetKernelArg(gKernel, 5, sizeof(size_t), &l);
            ALASSERT(errorcode == CL_SUCCESS);
            l = 1;//stride
            errorcode = clSetKernelArg(gKernel, 6, sizeof(size_t), &l);
            ALASSERT(errorcode == CL_SUCCESS);
            l = src.iWidth;
            errorcode = clSetKernelArg(gKernel, 7, sizeof(size_t), &l);
            ALASSERT(errorcode == CL_SUCCESS);
            l = src.iHeight;
            errorcode = clSetKernelArg(gKernel, 8, sizeof(size_t), &l);
            ALASSERT(errorcode == CL_SUCCESS);
            l = dst.iDepth;
            errorcode = clSetKernelArg(gKernel, 9, sizeof(size_t), &l);
            ALASSERT(errorcode == CL_SUCCESS);
            size_t size[] = {
                (size_t)dst.iWidth, (size_t)dst.iHeight, dst.pOrigin->height()
            };
            errorcode = clEnqueueNDRangeKernel(queue, gKernel, 3, NULL, size, NULL, 0, NULL, NULL);
            ALASSERT(errorcode == CL_SUCCESS);
        }
        _downloadMatrix(queue, output_gpu, dst.getMutable());
        return true;
    };
    ALOpenCL& cl = ALOpenCL::getInstance();
    cl.prepare(&gPrepare);
    cl.queueWork(run);
}

void ALOpenCLMatrix4DOp::vDeterFilter(const Matrix4D& dstDiff, const Matrix4D& dst, const Matrix4D& src,  Matrix4D& srcDiff, const Matrix4D& kernelData, Matrix4D& kernelDataDiff, int stride) const
{
    return ALBasicMatrix4DOp::vDeterFilter(dstDiff, dst, src, srcDiff, kernelData, kernelDataDiff, stride);
}

#endif