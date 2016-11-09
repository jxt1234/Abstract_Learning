#include "utils/ALStreamReader.h"
#include <math.h>
#include "loader/ALStandardLoader.h"
#include "math/ALFloatMatrix.h"
#include "ALBaseFloatMatrix.h"
#include "ALRefMatrix.h"
#include "ALCropVirtualMatrix.h"
#include "ALIndexVirtualMatrix.h"
#include "ALLargeMatrix.h"
#include <stdio.h>
#include <sstream>
#include <algorithm>
#ifdef ALOPENCL_MAC
#include "opencl/ALOpenCL.h"
#endif
#ifdef ALBLAS
#include "cblas.h"
#endif

#ifdef ALOPENCL_MAC
static const char* gProductSource = KERNEL(
                                       __kernel void product(__global float *a, __global float* b, __global float* c, size_t w, size_t h, size_t l)
                                       {
                                           int x = get_global_id(0);
                                           int y = get_global_id(1);
                                           {
                                               float sum = 0.0;
                                               int i=0;
                                               __global float* _a = a + y*l;
                                               __global float* _b = b + x*l;
                                               for (i=0; i<l/4; i=i+1)
                                               {
                                                   float4 aa = vload4(i, _a);
                                                   float4 bb = vload4(i, _b);
                                                   sum += dot(aa, bb);
                                               }
                                               i = i*4;
                                               for (; i<l; ++i)
                                               {
                                                   sum+= _a[i] * _b[i];
                                               }
                                               *(c + y*w + x) = sum;
                                           }
                                       }
                                       );
static cl_kernel gKernel = NULL;
static ALOpenCL::PrepareWork gPrepare = {
    [](cl_context context, cl_device_id device) {
        gKernel = ALOpenCL::compileAndBuild(gProductSource, "product", context, device);
        return true;
    },
    [](cl_context c){
        clReleaseKernel(gKernel);
        return true;
    }
};

static void setup(cl_mem ma, cl_mem mb, cl_mem mc, size_t w, size_t h, size_t l, cl_command_queue queue)
{
    cl_int errorcode;
    errorcode = clSetKernelArg(gKernel, 0, sizeof(cl_mem), &ma);
    ALASSERT(errorcode == CL_SUCCESS);
    errorcode = clSetKernelArg(gKernel, 1, sizeof(cl_mem), &mb);
    ALASSERT(errorcode == CL_SUCCESS);
    errorcode = clSetKernelArg(gKernel, 2, sizeof(cl_mem), &mc);
    ALASSERT(errorcode == CL_SUCCESS);
    errorcode = clSetKernelArg(gKernel, 3, sizeof(size_t), &w);
    ALASSERT(errorcode == CL_SUCCESS);
    errorcode = clSetKernelArg(gKernel, 4, sizeof(size_t), &h);
    ALASSERT(errorcode == CL_SUCCESS);
    errorcode = clSetKernelArg(gKernel, 5, sizeof(size_t), &l);
    ALASSERT(errorcode == CL_SUCCESS);
    size_t size[] = {
        w, h
    };
    errorcode = clEnqueueNDRangeKernel(queue, gKernel, 2, NULL, size, NULL, 0, NULL, NULL);
    ALASSERT(errorcode == CL_SUCCESS);
}

#endif
#ifdef ALOPENCL_MAC
static const char* gSSTSource = KERNEL(
                                       __kernel void product(__global float *a, __global float* c, size_t w, size_t l)
                                       {
                                           int x = get_global_id(0);
                                           int y = get_global_id(1);
                                           {
                                               float sum = 0.0;
                                               int i;
                                               __global float* _a = a + y*l;
                                               __global float* _b = a + x*l;
                                               for (i=0; i<l; ++i)
                                               {
                                                   sum+= _a[i] * _b[i];
                                               }
                                               *(c + y*w + x) = sum;
                                           }
                                       }
                                       );
static cl_kernel gSSTKernel = NULL;
static ALOpenCL::PrepareWork gSSTPrepare = {
    [](cl_context context, cl_device_id device) {
        gSSTKernel = ALOpenCL::compileAndBuild(gSSTSource, "product", context, device);
        return true;
    },
    [](cl_context c){
        clReleaseKernel(gSSTKernel);
        return true;
    }
};
#endif


ALFloatMatrix* ALFloatMatrix::create(size_t w, size_t h)
{
    return new ALBaseFloatMatrix(w, h);
}

ALFloatMatrix* ALFloatMatrix::productT(const ALFloatMatrix* A, const ALFloatMatrix* BT)
{
    ALAUTOTIME;
    ALASSERT(NULL!=A && NULL!=BT);
    ALASSERT(A->width() == BT->width());
    auto w = BT->height();
    auto h = A->height();
    ALFloatMatrix* C = ALFloatMatrix::create(w, h);
    productT(C, A, BT);
    return C;
}

void ALFloatMatrix::productT(ALFloatMatrix* C, const ALFloatMatrix* A, const ALFloatMatrix* BT)
{
    ALAUTOTIME;
    ALASSERT(NULL!=A && NULL!=BT);
    ALASSERT(A->width() == BT->width());
    ALASSERT(C->width() == BT->height());
    ALASSERT(C->height() == A->height());
    auto w = BT->height();
    auto h = A->height();
    auto l = A->width();
    productBasicT(C->vGetAddr(), C->stride(), A->vGetAddr(), A->stride(), BT->vGetAddr(), BT->stride(), w, h, l);
}

void ALFloatMatrix::productTA(ALFloatMatrix* C, const ALFloatMatrix* AT, const ALFloatMatrix* B)
{
    ALASSERT(NULL!=AT && NULL!=B);
    ALASSERT(AT->height() == B->height());
    ALASSERT(C->width() == B->width());
    ALASSERT(C->height() == AT->width());
    auto w = C->width();
    auto h = C->height();
    auto l = AT->height();
    auto c = C->vGetAddr();
    auto a = AT->vGetAddr();
    auto b = B->vGetAddr();
#ifdef ALBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, h, w, l, 1.0, a, h, b, w, 0.0, c, w);
    return;
#endif
    for (auto i=0; i<h; ++i)
    {
        for (auto j=0; j<w; ++j)
        {
            auto _c = c + w*i + j;
            auto _a = a + i;
            auto _b = b + j;
            ALFLOAT sum = 0.0;
            for (auto k=0; k<l; ++k)
            {
                sum += (_b[k*w]*_a[k*h]);
            }
            *_c = sum;
        }
    }

}


ALFloatMatrix* ALFloatMatrix::productSS(const ALFloatMatrix* A, const ALFloatMatrix* B)
{
    ALAUTOTIME;
    ALASSERT(NULL!=A && NULL!=B);
    ALASSERT(A->width() == B->height());
    ALASSERT(A->width() == A->height());
    ALASSERT(B->width() == B->height());
    auto w = A->width();
    ALFloatMatrix* R = ALFloatMatrix::create(w, w);
    for (size_t i=0; i<w; ++i)
    {
        auto a = A->vGetAddr(i);
        for (size_t j=i; j<w; ++j)
        {
            auto b = B->vGetAddr(j);
            ALFLOAT sum = 0.0;
            for (size_t k=0; k<w; ++k)
            {
                sum += a[k]*b[k];
            }
            R->vGetAddr(j)[i] = sum;
            R->vGetAddr(i)[j] = sum;
        }
    }
    return R;
}
ALFloatMatrix* ALFloatMatrix::product(const ALFloatMatrix* A, const ALFloatMatrix* B)
{
    ALAUTOTIME;
    ALASSERT(NULL!=A && NULL!=B);
    ALASSERT(A->width() == B->height());
    auto w = B->width();
    auto h = A->height();

    ALFloatMatrix* C = new ALBaseFloatMatrix(w, h);
    product(C, A, B);
    return C;
}
void ALFloatMatrix::product(ALFloatMatrix* C, const ALFloatMatrix* A, const ALFloatMatrix* B)
{
    ALAUTOTIME;
    ALASSERT(NULL!=A && NULL!=B);
    ALASSERT(A->width() == B->height());
    ALASSERT(C->width() == B->width());
    ALASSERT(C->height() == A->height());
    auto rA = A->stride();
    auto rB = B->stride();
    auto rC = C->stride();
    ALFLOAT* a = A->vGetAddr();
    ALFLOAT* b = B->vGetAddr();
    ALFLOAT* c = C->vGetAddr();
    auto w = B->width();
    auto h = A->height();
    auto l = A->width();
    productBasic(c, rC, a, rA, b, rB, w, h, l);
}

ALFloatMatrix* ALFloatMatrix::HAH(const ALFloatMatrix* A, const ALFloatMatrix* H)
{
    ALASSERT(A->width() == A->height());
    ALASSERT(H->width() == H->height());
    ALASSERT(A->width()>=H->width());
    ALAUTOTIME;
    if (A->width() == H->width())
    {
        ALSp<ALFloatMatrix> HA = product(A, H);
        return product(HA.get(), H);
    }
    auto aw = A->width();
    auto hw = H->width();
    auto n = aw - hw;
    auto R = ALFloatMatrix::create(aw, aw);
    /*A11, copy*/
    //zero(R);//FIXME DEBUG
    for (auto i=0; i<n; ++i)
    {
        auto r = R->vGetAddr(i);
        auto a = A->vGetAddr(i);
        ::memcpy(r, a, n*sizeof(ALFLOAT));
    }
    /* A12*H */
    productBasic(R->vGetAddr(0)+n, aw, A->vGetAddr(0)+n, aw, H->vGetAddr(), hw, hw, n, hw);
    /* H*A21 */
    productBasic(R->vGetAddr(n), aw, H->vGetAddr(), hw, A->vGetAddr(n), aw, n, hw, hw);
    /* H*A22*H */
    ALSp<ALFloatMatrix> HA = ALFloatMatrix::create(hw, hw);
    productBasic(HA->vGetAddr(), hw, H->vGetAddr(), hw, A->vGetAddr(n)+n, aw, hw, hw, hw);
    productBasic(R->vGetAddr(n)+n, aw, HA->vGetAddr(), hw, H->vGetAddr(), hw, hw, hw, hw);
    return R;
}
void ALFloatMatrix::productBasicT(ALFLOAT* c, size_t c_stride, const ALFLOAT* a, size_t a_stride, const ALFLOAT* b, size_t b_stride, size_t w, size_t h, size_t l)
{
    ALASSERT(a_stride!=0);
    ALASSERT(b_stride!=0);
    ALASSERT(c_stride!=0);
#ifdef ALOPENCL_MAC
    if (w > 10 && h>10)
    {
        auto run = [=](cl_context context, cl_command_queue queue)
        {
            cl_int errorcode;
            auto sizea = sizeof(ALFLOAT)*h*l;
            auto sizeb = sizeof(ALFLOAT)*w*l;
            auto sizec = sizeof(ALFLOAT)*h*w;
            cl_mem ma = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizea, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem mb = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeb, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem mc = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizec, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            ALDefer([&](){clReleaseMemObject(ma);clReleaseMemObject(mb);clReleaseMemObject(mc);});
            {
                ALFLOAT* ca = (ALFLOAT*)clEnqueueMapBuffer(queue, ma, false, CL_MAP_WRITE, 0, sizea, 0, NULL, NULL, &errorcode);
                ALFLOAT* cb = (ALFLOAT*)clEnqueueMapBuffer(queue, mb, false, CL_MAP_WRITE, 0, sizeb, 0, NULL, NULL, &errorcode);
                ALDefer([&](){clEnqueueUnmapMemObject(queue, ma, ca, 0, 0, 0);clEnqueueUnmapMemObject(queue, mb, cb, 0, NULL, NULL);});
                /*Copy b*/
                for (int i=0; i<w; ++i)
                {
                    ::memcpy(cb+l*i, b+b_stride*i, l*sizeof(ALFLOAT));
                }
                /*Copy a*/
                for (int i=0; i<h; ++i)
                {
                    ::memcpy(ca+l*i, a+a_stride*i, l*sizeof(ALFLOAT));
                }
            }
            setup(ma, mb, mc, w, h, l, queue);
            {
                ALFLOAT* cc = (ALFLOAT*)clEnqueueMapBuffer(queue, mc, true, CL_MAP_READ, 0, sizec, 0, NULL, NULL, &errorcode);
                ALDefer([&](){clEnqueueUnmapMemObject(queue, mc, cc, 0, NULL, NULL);});
                for (int i=0; i<h; ++i)
                {
                    ::memcpy(c+c_stride*i, cc+i*w, sizeof(ALFLOAT)*w);
                }
            }
            return true;
        };
        ALOpenCL& cl = ALOpenCL::getInstance();
        cl.prepare(&gPrepare);
        cl.queueWork(run);
        return;
    }
#endif
#ifdef ALBLAS
    const CBLAS_ORDER Order=CblasRowMajor;
    const CBLAS_TRANSPOSE TransA=CblasNoTrans;
    const CBLAS_TRANSPOSE TransB=CblasTrans;
    cblas_sgemm(Order, TransA, TransB, h, w, l, 1, a, a_stride, b, b_stride, 0.0, c, c_stride);
    return;
#endif
    for (auto i=0; i<h; ++i)
    {
        for (auto j=0; j<w; ++j)
        {
            auto _c = c + c_stride*i + j;
            auto _a = a + a_stride*i;
            auto _b = b + b_stride*j;
            ALFLOAT sum = 0.0;
            for (auto k=0; k<l; ++k)
            {
                sum += (_b[k]*_a[k]);
            }
            *_c = sum;
        }
    }
}
void ALFloatMatrix::productBasic(ALFLOAT* c, size_t c_stride, const ALFLOAT* a, size_t a_stride, const ALFLOAT* b, size_t b_stride, size_t w, size_t h, size_t l)
{
    ALASSERT(c_stride!=0);
    ALASSERT(b_stride!=0);
    ALASSERT(a_stride!=0);
#ifdef ALBLAS
    const CBLAS_ORDER Order=CblasRowMajor;
    const CBLAS_TRANSPOSE TransA=CblasNoTrans;
    const CBLAS_TRANSPOSE TransB=CblasNoTrans;
    cblas_sgemm(Order, TransA, TransB, h, w, l, 1, a, a_stride, b, b_stride, 0.0, c, c_stride);
    return;
#endif
#ifdef ALOPENCL_MAC
    if (w > 10 && h>10)
    {
        auto run = [=](cl_context context, cl_command_queue queue)
        {
            cl_int errorcode;
            auto sizea = sizeof(ALFLOAT)*h*l;
            auto sizeb = sizeof(ALFLOAT)*w*l;
            auto sizec = sizeof(ALFLOAT)*h*w;
            cl_mem ma = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizea, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem mb = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeb, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem mc = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizec, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            ALDefer([&](){clReleaseMemObject(ma);clReleaseMemObject(mb);clReleaseMemObject(mc);});
            {
                ALFLOAT* ca = (ALFLOAT*)clEnqueueMapBuffer(queue, ma, false, CL_MAP_WRITE, 0, sizea, 0, NULL, NULL, &errorcode);
                ALFLOAT* cb = (ALFLOAT*)clEnqueueMapBuffer(queue, mb, false, CL_MAP_WRITE, 0, sizeb, 0, NULL, NULL, &errorcode);
                ALDefer([&](){clEnqueueUnmapMemObject(queue, ma, ca, 0, 0, 0);clEnqueueUnmapMemObject(queue, mb, cb, 0, NULL, NULL);});
                /*transpose b firstly*/
                for (int i=0; i<w; ++i)
                {
                    for (int j=0; j<l ;++j)
                    {
                        cb[l*i+j] = b[b_stride*j+i];
                    }
                }
                /*Copy a*/
                for (int i=0; i<h; ++i)
                {
                    ::memcpy(ca+l*i, a+a_stride*i, l*sizeof(ALFLOAT));
                }
            }
            setup(ma, mb, mc, w, h, l, queue);
            {
                ALFLOAT* cc = (ALFLOAT*)clEnqueueMapBuffer(queue, mc, true, CL_MAP_READ, 0, sizec, 0, NULL, NULL, &errorcode);
                ALDefer([&](){clEnqueueUnmapMemObject(queue, mc, cc, 0, NULL, NULL);});
                for (int i=0; i<h; ++i)
                {
                    ::memcpy(c+c_stride*i, cc+i*w, sizeof(ALFLOAT)*w);
                }
            }
            return true;
        };
        ALOpenCL& cl = ALOpenCL::getInstance();
        cl.prepare(&gPrepare);
        cl.queueWork(run);
        return;
    }
#endif
    for (auto i=0; i<h; ++i)
    {
        for (auto j=0; j<w; ++j)
        {
            auto _c = c + c_stride*i + j;
            auto _a = a + a_stride*i;
            auto _b = b + j;
            ALFLOAT sum = 0.0;
            for (auto k=0; k<l; ++k)
            {
                sum += (*_a) * (*_b);
                _a += 1;
                _b += b_stride;
            }
            *_c = sum;
        }
    }
}

ALFloatMatrix* ALFloatMatrix::inverse(const ALFloatMatrix* A)
{
    ALASSERT(NULL!=A);
    ALASSERT(A->width() == A->height());
    ALFloatMatrix* result = ALFloatMatrix::create(A->width(), A->height());
    inverse_basic(A, result);
    return result;
}

ALFLOAT ALFloatMatrix::inverse_basic(const ALFloatMatrix* A, ALFloatMatrix* dst)
{
    ALASSERT(NULL!=A);
    ALASSERT(NULL!=dst);
    ALASSERT(A->stride()!=0 && dst->stride() != 0);
    ALASSERT(A->width() > 0 && A->height() > 0);
    ALFLOAT det = 1.0;
    auto w = A->width();
    auto h = A->height();
    ALASSERT(w == h);
    ALASSERT(dst->width() == dst->height() && dst->height() == w);
    ALFloatMatrix* result = dst;
    ALAutoStorage<ALFLOAT> __a(w*w);
    ALFLOAT* a = __a.get();
    ALAutoStorage<ALFLOAT> _a_r(w);
    ALFLOAT* a_r = _a_r.get();
    ALAutoStorage<ALFLOAT> _c_r(w);
    ALFLOAT* c_r = _c_r.get();
    ALFLOAT* _a = (ALFLOAT*)A->vGetAddr();
    auto rA = A->stride();
    auto rC = result->stride();
    ALFLOAT* c = result->vGetAddr();
    for (int i=0; i<w; ++i)
    {
        for (int j=0; j<w; ++j)
        {
            *(a+i*w+j) = *(_a+i*rA+j);
            if (i==j)
            {
                *(c+rC*i+j) = 1;
            }
            else
            {
                *(c+rC*i+j) = 0;
            }
        }
    }
    auto n = w;
    for (int i=0; i<n; ++i)
    {
        /*Swap until all M(i, i) is not zero*/
        bool zero = false;
        ALFLOAT weight = *(a+i*rA+i);
        if (ZERO(weight))
        {
            zero = true;
            for (int j=i+1; j<n; ++j)
            {
                ALFLOAT temp;
                weight = *(a+j*rA+i);
                if (!ZERO(weight))
                {
#define SWAP(x, y) temp=(x);(x)=(y);(y)=temp;
                    zero = false;
                    for (int r=i; r<n; ++r)
                    {
                        SWAP(*(a+i*rA+r), *(a+j*rA+r));
                        SWAP(*(c+i*rC+r), *(c+j*rC+r))/*Do the same for c*/
                    }
#undef SWAP
                }
            }
        }
        det*=weight;
        /*permit failed for inverse*/
        if (zero)
        {
            for (int j=0; j<n; ++j)
            {
                *(c+i+j*rC) = 0;
            }
            continue;
        }
        //assert(!ZERO(weight));
        /*Do row transform to make the i row of A be "0 0 0 1 x1 x2...."*/
        //For A, the value before i column must be zero
        for (int j=i; j<n; ++j)
        {
            *(a+i*rA+j) /=weight;
            a_r[j] = *(a+i*rA+j);
        }
        //For C, it must do the same, but the value before i column may not be zero
        for (int j=0; j<n; ++j)
        {
            *(c+i*rC+j) /= weight;
            c_r[j] = *(c+i*rC+j);
        }
        /*Do row transform for other rows, make sure that all value in the column i zero*/
        for (int k=0; k<n; ++k)
        {
            if (k==i) continue;
            ALFLOAT p = *(a+k*rA+i);
            for (int j=i; j<n; ++j)
            {
                *(a+k*rA+j) -= (p*a_r[j]);
            }
            //The Same for c
            for (int j=0; j<n; ++j)
            {
                *(c+k*rC+j) -= (p*c_r[j]);
            }
        }
    }
    return det;
}

ALFloatMatrix* ALFloatMatrix::transpose(const ALFloatMatrix* A)
{
    ALAUTOTIME;
    ALASSERT(NULL!=A);
    ALASSERT(A->width() > 0 && A->height() > 0);
    auto w = A->width();
    auto h = A->height();
    auto result = new ALBaseFloatMatrix(h, w);
    for (size_t i=0; i<w; ++i)
    {
        auto dst = result->vGetAddr(i);
        for (size_t j=0; j<h; ++j)
        {
            auto src = A->vGetAddr(j);
            dst[j] = src[i];
        }
    }
    return result;
}

void ALFloatMatrix::transpose(const ALFloatMatrix* src, ALFloatMatrix* dst)
{
    ALASSERT(NULL!=src);
    ALASSERT(NULL!=dst);
    ALASSERT(src->width() == dst->height());
    ALASSERT(src->height() == dst->width());
    auto w = src->width();
    auto h = src->height();
    for (size_t i=0; i<w; ++i)
    {
        auto dst_ = dst->vGetAddr(i);
        for (size_t j=0; j<h; ++j)
        {
            auto src_ = src->vGetAddr(j);
            dst_[j] = src_[i];
        }
    }
}

/*TODO*/
ALFloatMatrix* ALFloatMatrix::sts(const ALFloatMatrix* A, bool transpose)
{
    ALASSERT(NULL!=A);
    ALASSERT(A->width() > 0 && A->height() > 0);
    auto w = A->width();
    auto h = A->height();
    auto resw = w;
    if (transpose)
    {
        resw = h;
    }
    ALFloatMatrix* result = new ALBaseFloatMatrix(resw, resw);
#ifdef ALOPENCL_MAC
    auto run = [=](cl_context context, cl_command_queue queue){
        ALFLOAT* a = A->vGetAddr();
        cl_int errorcode;
        auto sizea = sizeof(ALFLOAT)*h*w;
        auto sizec = sizeof(ALFLOAT)*resw*resw;
        cl_mem ma = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizea, NULL, &errorcode);
        ALASSERT(CL_SUCCESS == errorcode);
        cl_mem mc = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizec, NULL, &errorcode);
        ALASSERT(CL_SUCCESS == errorcode);
        auto ws = w;
        auto hs = h;
        ALDefer([&](){clReleaseMemObject(ma);clReleaseMemObject(mc);});
        {
            ALFLOAT* ca = (ALFLOAT*)clEnqueueMapBuffer(queue, ma, false, CL_MAP_WRITE, 0, sizea, 0, NULL, NULL, &errorcode);
            ALDefer([&](){clEnqueueUnmapMemObject(queue, ma, ca, 0, 0, 0);});
            if (!transpose)
            {
                /*transpose a firstly*/
                for (int i=0; i<w; ++i)
                {
                    for (int j=0; j<h ;++j)
                    {
                        ca[i*h+j] = a[j*w+i];
                    }
                }
            }
            else
            {
                /*Copy a*/
                ::memcpy(ca, a, sizea);
                ws = h;
                hs = w;
            }
        }
        {
            cl_int errorcode;
            errorcode = clSetKernelArg(gSSTKernel, 0, sizeof(cl_mem), &ma);
            ALASSERT(errorcode == CL_SUCCESS);
            errorcode = clSetKernelArg(gSSTKernel, 1, sizeof(cl_mem), &mc);
            ALASSERT(errorcode == CL_SUCCESS);
            errorcode = clSetKernelArg(gSSTKernel, 2, sizeof(size_t), &resw);
            ALASSERT(errorcode == CL_SUCCESS);
            errorcode = clSetKernelArg(gSSTKernel, 3, sizeof(size_t), &hs);
            ALASSERT(errorcode == CL_SUCCESS);
            size_t size[] = {resw, resw};
            errorcode = clEnqueueNDRangeKernel(queue, gSSTKernel, 2, NULL, size, NULL, 0, NULL, NULL);
            ALASSERT(errorcode == CL_SUCCESS);
        }
        {
            auto c = result->vGetAddr();
            ALFLOAT* cc = (ALFLOAT*)clEnqueueMapBuffer(queue, mc, true, CL_MAP_READ, 0, sizec, 0, NULL, NULL, &errorcode);
            ALDefer([&](){clEnqueueUnmapMemObject(queue, mc, cc, 0, NULL, NULL);});
            for (int i=0; i<resw; ++i)
            {
                ::memcpy(c+resw*i, cc+i*resw, sizeof(ALFLOAT)*resw);
            }
        }
        return true;
    };
    ALOpenCL& cl = ALOpenCL::getInstance();
    cl.prepare(&gSSTPrepare);
    cl.queueWork(run);
    return result;
#endif
    ALFLOAT* a = A->vGetAddr();
    ALFLOAT* b = a;
    auto rA = A->width();
    size_t ws, hs;
    if (!transpose)
    {
        ws = 1;
        hs = rA;
    }
    else
    {
        ws = rA;
        hs = 1;
        auto temp = w;
        w = h;
        h = temp;
    }
    auto rC = result->width();
    ALFLOAT* c = result->vGetAddr();

#ifdef ALBLAS
    auto a_stride = rA;
    auto c_stride = rC;
    if (transpose)
    {
        //SST
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rC, rC, A->width(), 1.0, a, a_stride, a, a_stride, 0.0, c, c_stride);
    }
    else
    {
        //STS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, rC, rC, A->height(), 1.0, a, a_stride, a, a_stride, 0.0, c, c_stride);
    }
    return result;
#endif
    /*Compute the Matrix, it's symmetrical*/
    for (int i=0; i<w; ++i)
    {
        for (int j=i; j<w; ++j)
        {
            ALFLOAT sum = 0; 
            ALFLOAT* sA = a + i*ws;
            ALFLOAT* sB = b + j*ws;
            for (int k=0; k<h; ++k)
            {
                sum += (*sA)*(*sB);
                sA += hs;
                sB += hs;
            }
            *(c+i*rC+j) = sum;
            *(c+j*rC+i) = sum;
        }
    }
    return result;
}

void ALFloatMatrix::print(const ALFloatMatrix* A, std::ostream& os)
{
    ALASSERT(NULL!=A);
    for (int i=0; i<A->height(); ++i)
    {
        auto a = A->vGetAddr(i);
        for (int j=0; j<A->width(); ++j)
        {
            os << *(a+j) <<"    ";
        }
        os << "\n";
    }
}


ALFloatMatrix* ALFloatMatrix::createIdentity(size_t n)
{
    ALASSERT(n>0);
    ALFloatMatrix* M = new ALBaseFloatMatrix(n,n);
    for (size_t i=0; i<n; ++i)
    {
        ALFLOAT* m = M->vGetAddr(i);
        ::memset(m, 0, n*sizeof(ALFLOAT));
        m[i] = 1;
    }
    return M;
}

ALFLOAT ALFloatMatrix::norm(const ALFloatMatrix* X)
{
    ALASSERT(NULL!=X);
    ALASSERT(X->height()==1);
    ALFLOAT sum = 0;
    ALFLOAT* x = X->vGetAddr();
    auto w = X->width();
    for (int i=0; i<w; ++i)
    {
        sum += (x[i]*x[i]);
    }
    return sqrt(sum);
}

ALFloatMatrix* ALFloatMatrix::enlarge(size_t n, const ALFloatMatrix* X)
{
    ALAUTOTIME;
    ALASSERT(NULL!=X);
    ALASSERT(X->width() == X->height());
    ALASSERT(X->width() < n);
    auto w = X->width();
    ALFloatMatrix* Large = new ALBaseFloatMatrix(n, n);
    auto diff = n-w;
    for (int i=0; i<diff; ++i)
    {
        ALFLOAT* l = Large->vGetAddr(i);
        ::memset(l, 0, sizeof(ALFLOAT)*n);
        l[i] = 1.0;
    }
    for (auto i=diff; i<n; ++i)
    {
        ALFLOAT* l = Large->vGetAddr(i);
        ALFLOAT* x = X->vGetAddr(i-diff);
        ::memset(l, 0, sizeof(ALFLOAT)*diff);
        ::memcpy(l+diff, x, w*sizeof(ALFLOAT));
    }
    return Large;
}
ALFloatMatrix* ALFloatMatrix::createDiag(const ALFloatMatrix* X)
{
    ALASSERT(NULL!=X);
    ALASSERT(X->height()==1);
    auto n = X->width();
    ALFloatMatrix* D = new ALBaseFloatMatrix(n, n);
    ALFLOAT* x = X->vGetAddr();
    for (size_t i=0; i<n; ++i)
    {
        ALFLOAT* d = D->vGetAddr(i);
        ::memset(d, 0, n*sizeof(ALFLOAT));
        d[i] = x[i];
    }
    return D;
}

void ALFloatMatrix::zero(ALFloatMatrix* X)
{
    ALASSERT(NULL!=X);
    for (auto i=0; i<X->height(); ++i)
    {
        ALFLOAT* x = X->vGetAddr(i);
        ::memset(x, 0, sizeof(ALFLOAT)*(X->width()));
    }
}
void ALFloatMatrix::copy(ALFloatMatrix* dst, const ALFloatMatrix* src)
{
    ALASSERT(dst!=NULL);
    ALASSERT(src!=NULL);
    ALASSERT(dst->width() == src->width());
    ALASSERT(dst->height() == src->height());
    auto w = dst->width();
    auto h = dst->height();
    for (auto i=0; i<h; ++i)
    {
        ALFLOAT* d = dst->vGetAddr(i);
        ALFLOAT* s = src->vGetAddr(i);
        ::memcpy(d, s, sizeof(ALFLOAT)*w);
    }
}

void ALFloatMatrix::transposeBasic(const ALFLOAT* src, size_t src_stride, ALFLOAT* dst, size_t dst_stride, size_t w, size_t h)
{
    ALASSERT(dst!=NULL);
    ALASSERT(src!=NULL);
    for (size_t i=0; i<h; ++i)
    {
        for (size_t j=0; j<w; ++j)
        {
            dst[j*dst_stride+i] = src[i*src_stride+j];
        }
    }
}


void ALFloatMatrix::quickSave(const ALFloatMatrix* m, ALWStream* f)
{
    ALASSERT(NULL!=m);
    ALASSERT(NULL!=f);
    uint32_t w = (uint32_t)m->width();
    uint32_t h = (uint32_t)m->height();
    f->write(w);
    f->write(h);
    for (size_t i=0; i<m->height(); ++i)
    {
        auto _m = m->vGetAddr(i);
        f->vWrite(_m, m->width()*sizeof(ALFLOAT));
    }
}
ALFloatMatrix* ALFloatMatrix::quickLoad(ALStream* f)
{
    ALASSERT(NULL!=f);
    auto w = f->read<uint32_t>();
    auto h = f->read<uint32_t>();
    if (f->vIsEnd())
    {
        return NULL;
    }
    ALASSERT(w>0 && h>0);
    auto M = ALFloatMatrix::create(w, h);
    f->vRead(M->vGetAddr(), w*h*sizeof(ALFLOAT));
    return M;
}
ALFloatMatrix* ALFloatMatrix::quickLoadLarge(ALStream* f)
{
    ALSp<ALFloatMatrix> first = quickLoad(f);
    ALASSERT(NULL!=first.get());
    ALLargeMatrix* total = new ALLargeMatrix(first);
    while (!f->vIsEnd())
    {
        ALSp<ALFloatMatrix> temp = quickLoad(f);
        if (NULL == temp.get())
        {
            break;
        }
        total->addMatrix(temp);
    }
    return total;
}

void ALFloatMatrix::save(const ALFloatMatrix* m, ALWStream* f)
{
    ALASSERT(NULL!=m);
    auto w = m->width();
    auto h = m->height();
    for (int i=0; i<h; ++i)
    {
        std::ostringstream line;
        auto _line = m->vGetAddr(i);
        for (int j=0; j<w; ++j)
        {
            line << _line[j] << " ";
        }
        line << "\n";
        f->vWrite(line.str().c_str(), line.str().size());
    }
}

ALFloatMatrix* ALFloatMatrix::load(ALStream* f)
{
    const size_t matrix_hunit = 100;
    ALSp<ALStreamReader> reader = new ALStreamReader(f);
    ALASSERT(!reader->end());
    /*First line, Compute Units*/
    ALDynamicBuffer dyBuffer(4096);//TODO
    size_t len = reader->readline(&dyBuffer);
    size_t num = ALStandardLoader::measureNumbers(dyBuffer.content(), len);
    
    /*Load first matrix begin*/
    ALFloatMatrix* first = new ALBaseFloatMatrix(num, matrix_hunit);
    //Copy First Line
    ALStandardLoader::loadNumbers(first->vGetAddr(0), num, dyBuffer.content(), len);
    size_t cur = 1;
    while (!reader->end() && cur < matrix_hunit)
    {
        len = reader->readline(&dyBuffer);
        ALStandardLoader::loadNumbers(first->vGetAddr(cur++), num, dyBuffer.content(), len);
    }
    /*Load first matrix end*/
    
    if (reader->end())
    {
        /*One Matrix is OK, just return it*/
        first->mHeight = cur;
        return first;
    }
    /*Not enough, use largematrix*/
    ALLargeMatrix* result = new ALLargeMatrix(first);
    while (!reader->end())
    {
        cur = 0;
        ALSp<ALFloatMatrix> unit = new ALBaseFloatMatrix(num, matrix_hunit);
        while (!reader->end() && cur < matrix_hunit)
        {
            len = reader->readline(&dyBuffer);
            ALStandardLoader::loadNumbers(unit->vGetAddr(cur++), num, dyBuffer.content(), len);
        }
        unit->mHeight = cur;

        result->addMatrix(unit);
    }
    return result;
}


ALFloatMatrix* ALFloatMatrix::createCropVirtualMatrix(const ALFloatMatrix* base, size_t l, size_t t, size_t r, size_t b)
{
    ALASSERT(NULL!=base);
    ALASSERT(0<=l && 0<=t);
    ALASSERT(l<=r && t<=b);
    ALASSERT(r<base->width() && b<base->height());
    return new ALCropVirtualMatrix(base, l, t, r, b);
}

ALFloatMatrix* ALFloatMatrix::createIndexVirtualMatrix(ALFLOAT** indexes, size_t w, size_t h)
{
    ALASSERT(NULL!=indexes);
    ALASSERT(w>0);
    ALASSERT(h>0);
    return new ALIndexVirtualMatrix(indexes, w, h);
}


ALFloatMatrix* ALFloatMatrix::genTypes(const ALFloatMatrix* Y)
{
    ALASSERT(NULL!=Y);
    ALASSERT(1 == Y->height());
    std::vector<ALFLOAT> types;
    auto w = Y->width();
    auto yv = Y->vGetAddr();
    for (size_t i=0; i<w; ++i)
    {
        bool find = false;
        for (auto t : types)
        {
            if (ZERO(t-yv[i]))
            {
                find = true;
                break;
            }
        }
        if (!find)
        {
            types.push_back(yv[i]);
        }
    }
    ALASSERT(types.size()>0);
    ALFloatMatrix* result = ALFloatMatrix::create(types.size(), 1);
    auto _r = result->vGetAddr(0);
    for (size_t i=0; i<types.size(); ++i)
    {
        _r[i] = types[i];
    }
    return result;
}

ALFloatMatrix* ALFloatMatrix::randomSelectMatrix(const ALFloatMatrix* base, size_t height, bool copy)
{
    ALASSERT(NULL!=base);
    ALASSERT(height < base->height());
    ALASSERT(height>0);
    auto totalHeight = base->height();
    auto width = base->width();
    if (copy)
    {
        ALFloatMatrix* copyedM = create(width, height);
        for (size_t i=0; i<height; ++i)
        {
            int j = ALRandom::mid(0, (int)totalHeight);
            auto src = base->vGetAddr(j);
            auto dst = copyedM->vGetAddr(i);
            ::memcpy(dst, src, sizeof(ALFLOAT)*width);
        }
        return copyedM;
    }
    ALAUTOSTORAGE(indexes, ALFLOAT*, height);
    for (int i=0; i<height; ++i)
    {
        int j = ALRandom::mid(0, (int)totalHeight);
        indexes[i] = base->vGetAddr(j);
    }
    ALFloatMatrix* result = new ALIndexVirtualMatrix(indexes, base->width(), height, true);
    return result;
}

ALFloatMatrix* ALFloatMatrix::unionHorizontal(const ALFloatMatrix* Y, const ALFloatMatrix* X)
{
    ALASSERT(NULL!=Y);
    ALASSERT(NULL!=X);
    ALASSERT(Y->height() == X->height());
    auto totalWidth = Y->width() + X->width();
    auto height = Y->height();
    ALFloatMatrix* unionMatrix = ALFloatMatrix::create(totalWidth, height);
    ALSp<ALFloatMatrix> virtualY = createCropVirtualMatrix(unionMatrix, 0, 0, Y->width()-1, height-1);
    copy(virtualY.get(), Y);
    ALSp<ALFloatMatrix> virtualX = createCropVirtualMatrix(unionMatrix, Y->width(), 0, totalWidth-1, height-1);
    copy(virtualX.get(), X);
    return unionMatrix;
}

ALFloatMatrix* ALFloatMatrix::createRefMatrix(ALFLOAT* base, size_t w, size_t h)
{
    return new ALRefMatrix(base, w, h);
}

void ALFloatMatrix::linear(ALFloatMatrix* C, const ALFloatMatrix* A, ALFLOAT pa, const ALFloatMatrix* B, ALFLOAT pb)
{
    ALASSERT(NULL!=A);
    ALASSERT(NULL!=B);
    ALASSERT(NULL!=C);
    ALASSERT(A->width() == B->width());
    ALASSERT(A->height() == B->height());
    ALASSERT(C->width() == A->width());
    ALASSERT(C->height() == A->height());
    auto w = A->width();
    auto h = B->height();
    for (size_t i=0; i<h; ++i)
    {
        ALFLOAT* a = A->vGetAddr(i);
        ALFLOAT* b = B->vGetAddr(i);
        ALFLOAT* c = C->vGetAddr(i);
        for (size_t j=0; j<w; ++j)
        {
            auto _a = a[j];
            auto _b = b[j];
            *(c+j) = _a*pa + _b*pb;
        }
    }

}

void ALFloatMatrix::linearDirect(ALFloatMatrix* X, ALFLOAT a, ALFLOAT b)
{
    ALASSERT(NULL!=X);
    auto w = X->width();
    auto h = X->height();
    for (int i=0; i<h; ++i)
    {
        auto x = X->vGetAddr(i);
        for (int j=0; j<w; ++j)
        {
            x[j] = a*x[j]+b;
        }
    }
}
ALFloatMatrix* ALFloatMatrix::getTypes(const ALFloatMatrix* YP, const ALFloatMatrix* prop)
{
    ALASSERT(NULL!=YP);
    ALASSERT(NULL!=prop);
    ALASSERT(YP->width() == prop->width());
    ALASSERT(1 == prop->height());
    ALFloatMatrix* result = ALFloatMatrix::create(YP->height(), 1);
    auto dst = result->vGetAddr();
    auto w = YP->width();
    auto h = YP->height();
    
    auto p = prop->vGetAddr();
    for (int i=0; i<h; ++i)
    {
        auto yp = YP->vGetAddr(i);
        int maxIndex = 0;
        ALFLOAT maxNumber = yp[0];
        for (int j=1; j<w; ++j)
        {
            if (yp[j] > maxNumber)
            {
                maxNumber = yp[j];
                maxIndex = j;
            }
        }
        dst[i] = p[maxIndex];
    }
    return result;
}

void ALFloatMatrix::checkAndSet(ALFloatMatrix* X, ALFLOAT c)
{
    ALASSERT(NULL!=X);
    auto w = X->width();
    auto h = X->height();
    for (int i=0; i<h; ++i)
    {
        auto x = X->vGetAddr(i);
        for (int j=0; j<w; ++j)
        {
            auto xj = x[j];
            if (!(xj>=0)&&(!(xj<0)))
            {
                x[j] = c;
            }
        }
    }
}

bool ALFloatMatrix::theSame(const ALFloatMatrix* X, const ALFloatMatrix* Y, ALFLOAT error)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(X->width() == Y->width());
    ALASSERT(X->height() == Y->height());
    ALASSERT(error>=0);
    auto w = X->width();
    auto h = Y->height();
    for (int i=0; i<h; ++i)
    {
        auto x = X->vGetAddr(i);
        auto y = Y->vGetAddr(i);
        for (int j=0; j<w; ++j)
        {
            auto diff = x[j]-y[j];
            if (diff > error || diff <-error)
            {
                return false;
            }
        }
    }
    return true;
}

void ALFloatMatrix::typeExpand(ALFloatMatrix* Y_Expand/*Output*/, const ALFloatMatrix* YT/*Input*/)
{
    ALASSERT(NULL!=YT);
    ALASSERT(NULL!=Y_Expand);
    ALASSERT(1 == YT->height());
    ALASSERT(Y_Expand->height() == YT->width());
    auto h = Y_Expand->height();
    auto y = YT->vGetAddr();
    ALFloatMatrix::zero(Y_Expand);
    for (size_t i=0; i<h; ++i)
    {
        auto y_e = Y_Expand->vGetAddr(i);
        size_t pos = y[i];
        y_e[pos] = 1.0;
    }
}

void ALFloatMatrix::linearVector(ALFloatMatrix* C, const ALFloatMatrix* A, ALFLOAT a, const ALFloatMatrix* B, ALFLOAT b)
{
    ALASSERT(C->width() == A->width());
    ALASSERT(C->width() == B->width());
    ALASSERT(B->height() == 1);
    ALASSERT(C->height() == A->height());
    auto bv = B->vGetAddr();
    auto w = C->width();
    auto h = A->height();
    for (size_t i=0; i<h; ++i)
    {
        auto dst = C->vGetAddr(i);
        auto av = A->vGetAddr(i);
        for (size_t j=0; j<w; ++j)
        {
            dst[j] = av[j]*a + bv[j]*b;
        }
    }
}

void ALFloatMatrix::runLineFunction(ALFloatMatrix* dst, const ALFloatMatrix* src, std::function<void(ALFLOAT*, ALFLOAT*, size_t)> function)
{
    ALASSERT(NULL!=src);
    ALASSERT(NULL!=dst);
    ALASSERT(dst->height() == src->height());
    ALASSERT(dst->width() == src->width());//TODO
    auto w = src->width();
    auto h = src->height();
    for (size_t y=0; y<h; ++y)
    {
        auto _dst = dst->vGetAddr(y);
        auto _src = src->vGetAddr(y);
        function(_dst, _src, w);
    }
}
void ALFloatMatrix::productDot(ALFloatMatrix* C, const ALFloatMatrix* A, const ALFloatMatrix* B)
{
    ALASSERT(NULL!=C);
    ALASSERT(NULL!=B);
    ALASSERT(NULL!=A);
    ALASSERT(C->width() == A->width());
    ALASSERT(C->width() == B->width());
    ALASSERT(C->height() == A->height());
    ALASSERT(C->height() == B->height());
    auto w = A->width();
    auto h = A->height();
    for (size_t y=0; y<h; ++y)
    {
        auto c = C->vGetAddr(y);
        auto b = B->vGetAddr(y);
        auto a = A->vGetAddr(y);
        for (size_t x=0; x<w; ++x)
        {
            c[x] = b[x]*a[x];
        }
    }

}
void ALFloatMatrix::productDivide(ALFloatMatrix* C, const ALFloatMatrix* A, const ALFloatMatrix* B)
{
    ALASSERT(NULL!=C);
    ALASSERT(NULL!=B);
    ALASSERT(NULL!=A);
    ALASSERT(C->width() == A->width());
    ALASSERT(C->width() == B->width());
    ALASSERT(C->height() == A->height());
    ALASSERT(C->height() == B->height());
    auto w = A->width();
    auto h = A->height();
    for (size_t y=0; y<h; ++y)
    {
        auto c = C->vGetAddr(y);
        auto b = B->vGetAddr(y);
        auto a = A->vGetAddr(y);
        for (size_t x=0; x<w; ++x)
        {
            c[x] = a[x]/b[x];
        }
    }
}

void ALFloatMatrix::set(ALFloatMatrix* X, ALFLOAT c)
{
    ALASSERT(NULL!=X);
    auto w = X->width();
    auto h = X->height();
    for (size_t i=0; i<h; ++i)
    {
        auto a = X->vGetAddr(i);
        for (size_t j=0; j<w; ++j)
        {
            a[j] = c;
        }
    }
}
void ALFloatMatrix::runReduceFunction(ALFloatMatrix* dst, const ALFloatMatrix* src, std::function<void(ALFLOAT*, ALFLOAT*, size_t)> function)
{
    ALASSERT(NULL!=src);
    ALASSERT(NULL!=dst);
    ALASSERT(dst->height()==1);
    ALASSERT(dst->width() == src->width());//TODO
    auto w = src->width();
    auto h = src->height();
    auto _dst = dst->vGetAddr();
    for (size_t y=0; y<h; ++y)
    {
        auto _src = src->vGetAddr(y);
        function(_dst, _src, w);
    }
}
