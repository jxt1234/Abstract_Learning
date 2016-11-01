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
                                          __kernel void filter(__global float *input, __global float* kernelM, __global float* output, size_t kw, size_t kh, size_t kd, size_t stride, size_t iw, size_t ih, size_t od, size_t ow, size_t oh, size_t n_batch)
                                          {
                                              int x = get_global_id(0);
                                              int y = get_global_id(1);
                                              int z = get_global_id(2);
                                              float sum = 0.0;
                                              int oi,i,j,k;
                                              int kbatchsize = kw*kh*kd+1;
                                              int inputBatchSize = iw*ih*kd;
                                              int outputBatchSize = ow*oh*od;
                                              __global float* input_base = input + inputBatchSize*z + x*stride + y*stride*iw;
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

static void _setKernel(cl_kernel kernel, cl_mem input_gpu, cl_mem kernel_gpu, cl_mem output_gpu, const ALIMatrix4DOp::Matrix4D& src, const ALIMatrix4DOp::Matrix4D& dst, const ALIMatrix4DOp::Matrix4D& kernelData)
{
    cl_int errorcode;
    size_t l;
    errorcode = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_gpu);
    ALASSERT(errorcode == CL_SUCCESS);
    errorcode = clSetKernelArg(kernel, 1, sizeof(cl_mem), &kernel_gpu);
    ALASSERT(errorcode == CL_SUCCESS);
    errorcode = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_gpu);
    ALASSERT(errorcode == CL_SUCCESS);
    l = kernelData.iWidth;
    errorcode = clSetKernelArg(kernel, 3, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = kernelData.iHeight;
    errorcode = clSetKernelArg(kernel, 4, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = kernelData.iDepth;
    errorcode = clSetKernelArg(kernel, 5, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = 1;//stride
    errorcode = clSetKernelArg(kernel, 6, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = src.iWidth;
    errorcode = clSetKernelArg(kernel, 7, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = src.iHeight;
    errorcode = clSetKernelArg(kernel, 8, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = dst.iDepth;
    errorcode = clSetKernelArg(kernel, 9, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = dst.iWidth;
    errorcode = clSetKernelArg(kernel, 10, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = dst.iHeight;
    errorcode = clSetKernelArg(kernel, 11, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    l = dst.pOrigin->height();
    errorcode = clSetKernelArg(kernel, 12, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
}

void ALOpenCLMatrix4DOp::vFilter(Matrix4D& dst, const Matrix4D& src, const Matrix4D& kernelData, int stride) const
{
    ALASSERT(stride == 1);
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
        
        ALDefer([&](){clReleaseMemObject(input_gpu);clReleaseMemObject(kernel_gpu);clReleaseMemObject(output_gpu);});
        _setKernel(gKernel, input_gpu, kernel_gpu, output_gpu, src, dst, kernelData);
        {
            size_t size[] = {
                (size_t)dst.iWidth, (size_t)dst.iHeight, dst.pOrigin->height()
            };
            cl_int errorcode = clEnqueueNDRangeKernel(queue, gKernel, 3, NULL, size, NULL, 0, NULL, NULL);
            ALASSERT(errorcode == CL_SUCCESS);
        }
        _downloadMatrix(queue, output_gpu, dst.getMutable());
        return true;
    };
    ALOpenCL& cl = ALOpenCL::getInstance();
    cl.prepare(&gPrepare);
    cl.queueWork(run);
}

static const char* gDeterKernelErrorSource = KERNEL(
                                                    __kernel void KernelError(__global float *input, __global float* kernelDataDiff, __global float* output_diff, size_t kw, size_t kh, size_t kd, size_t stride, size_t iw, size_t ih, size_t od, size_t ow, size_t oh, size_t n_batch)
                                                    {
                                                        int x = get_global_id(0);
                                                        int z = get_global_id(1);
                                                        int kbatchsize = kw*kh*kd+1;
                                                        int i, j, k;
                                                        int outputBatchSize = ow*oh*od;
                                                        int inputBatchSize = iw*ih*kd;
                                                        __global float* kernelDataDiff_target = kernelDataDiff + kbatchsize*z + x;
                                                        if (x == kbatchsize-1)//Constant
                                                        {
                                                            float sum = 0.0;
                                                            for (int k=0; k<n_batch; ++k)
                                                            {
                                                                __global float* output_diff_base = output_diff + z*ow*oh + k*outputBatchSize;
                                                                for (i=0; i<oh; ++i)
                                                                {
                                                                    for (j=0; j<ow; ++j)
                                                                    {
                                                                        sum += output_diff_base[i*ow+j];
                                                                    }
                                                                }
                                                            }
                                                            *kernelDataDiff_target = sum;
                                                            return;
                                                        }
                                                        {
                                                            int ki = x % kw;
                                                            int kj = ((x - ki) / kw) % kh;
                                                            int kk = (x - ki - kj*kw)/kh/kw;
                                                            float sum = 0.0;
                                                            for (int k=0; k<n_batch; ++k)
                                                            {
                                                                __global float* output_diff_base = output_diff + z*ow*oh + k*outputBatchSize;
                                                                __global float* input_base = input + k*inputBatchSize + kk*iw*ih;
                                                                for (j=0; j<oh; ++j)
                                                                {
                                                                    for (i=0; i<ow; ++i)
                                                                    {
                                                                        sum += output_diff_base[j*ow+i]*input_base[(j+kj)*iw+(i+ki)];
                                                                    }
                                                                }
                                                            }
                                                            *kernelDataDiff_target = sum;
                                                        }
                                                    });
static const char* gDeterInputErrorSource = KERNEL(
                                                   __kernel void InputError(__global float *input_diff, __global float* kernelData, __global float* output_diff, size_t kw, size_t kh, size_t kd, size_t stride, size_t iw, size_t ih, size_t od, size_t ow, size_t oh, size_t n_batch)
                                                   {
                                                       int x = get_global_id(0);
                                                       int y = get_global_id(1);
                                                       int z = get_global_id(2);
                                                       float sum = 0.0;
                                                       int ii,i,j,k;
                                                       int kbatchsize = kw*kh*kd+1;
                                                       int inputBatchSize = iw*ih*kd;
                                                       int outputBatchSize = ow*oh*od;
                                                       __global float* input_base_diff = input_diff + inputBatchSize*z + x + y*iw;
                                                       __global float* kernel_base;
                                                       __global float* output_base = output_diff + outputBatchSize*z + x + y*ow;
                                                       for (ii=0; ii<kd; ++ii)
                                                       {
                                                           sum = 0.0;
                                                           for (i=0;i<od;++i)
                                                           {
                                                               kernel_base = kernelData+kbatchsize*i+ii*kw*kh;
                                                               for (j=0;j<kh; ++j)
                                                               {
                                                                   if (y<j || (y-j)>=oh)
                                                                   {
                                                                       continue;
                                                                   }
                                                                   for (k=0;k<kw;++k)
                                                                   {
                                                                       if (x<k || (x-k)>=ow)
                                                                       {
                                                                           continue;
                                                                       }
                                                                       sum += output_base[-k-j*ow+i*ow*oh]*kernel_base[k+j*kw];
                                                                   }
                                                               }
                                                           }
                                                           *(input_base_diff + iw*ih*ii) = sum;
                                                       }
                                                   });


void ALOpenCLMatrix4DOp::vDeterFilter(const Matrix4D& dstDiff, const Matrix4D& dst, const Matrix4D& src,  Matrix4D& srcDiff, const Matrix4D& kernelData, Matrix4D& kernelDataDiff, int stride) const
{
    ALASSERT(stride == 1);
    static cl_kernel gKernelInputError = NULL;
    static cl_kernel gKernelFilterError = NULL;
    static ALOpenCL::PrepareWork gPrepare = {
        [&](cl_context context, cl_device_id device) {
            gKernelInputError = ALOpenCL::compileAndBuild(gDeterInputErrorSource, "InputError", context, device);
            gKernelFilterError = ALOpenCL::compileAndBuild(gDeterKernelErrorSource, "KernelError", context, device);
            return true;
        },
        [&](cl_context c){
            clReleaseKernel(gKernelInputError);
            clReleaseKernel(gKernelFilterError);
            return true;
        }
    };
    ALOpenCL& cl = ALOpenCL::getInstance();
    cl.prepare(&gPrepare);
    
    /*Kernel*/
    auto krun = [&](cl_context context, cl_command_queue queue)
    {
        auto input_gpu = _uploadMatrix(context, queue, src.pOrigin);
        auto output_gpu = _uploadMatrix(context, queue, dstDiff.pOrigin);
        auto kernel_diff_gpu = _allocForWrite(context, queue, kernelDataDiff.pOrigin->width(), kernelDataDiff.pOrigin->height());
        ALDefer([&](){clReleaseMemObject(input_gpu);clReleaseMemObject(kernel_diff_gpu);clReleaseMemObject(output_gpu);});
        _setKernel(gKernelFilterError, input_gpu, kernel_diff_gpu, output_gpu, src, dst, kernelData);
        {
            size_t size[] = {
                (size_t)kernelDataDiff.getTotalWidth(), kernelDataDiff.pOrigin->height()
            };
            auto errorcode = clEnqueueNDRangeKernel(queue, gKernelFilterError, 2, NULL, size, NULL, 0, NULL, NULL);
            ALASSERT(errorcode == CL_SUCCESS);
        }
        
        if (NULL != srcDiff.pOrigin)
        {
            auto input_diff_gpu = _allocForWrite(context, queue, srcDiff.pOrigin->width(), srcDiff.pOrigin->height());
            auto kernel_gpu = _uploadMatrix(context, queue, kernelData.pOrigin);
            ALDefer([&](){clReleaseMemObject(input_diff_gpu);clReleaseMemObject(kernel_gpu);});
            
            _setKernel(gKernelInputError, input_diff_gpu, kernel_gpu, output_gpu, srcDiff, dstDiff, kernelData);
            size_t size[] = {
                (size_t)srcDiff.iWidth, (size_t)srcDiff.iHeight, srcDiff.pOrigin->height()
            };
            auto errorcode = clEnqueueNDRangeKernel(queue, gKernelInputError, 3, NULL, size, NULL, 0, NULL, NULL);
            ALASSERT(errorcode == CL_SUCCESS);
            _downloadMatrix(queue, input_diff_gpu, srcDiff.getMutable());
        }
        _downloadMatrix(queue, kernel_diff_gpu, kernelDataDiff.getMutable());

        return true;
    };
    cl.queueWork(krun);
    
    //ALBasicMatrix4DOp::vDeterFilter(dstDiff, dst, src, srcDiff, kernelData, kernelDataDiff, stride);
}

#endif