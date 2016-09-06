#include "learn/ALSVMKernelFactory.h"
#include <sstream>
#include <math.h>
using namespace std;
typedef map<string, string>::const_iterator ITER;
#ifdef ALOPENCL_MAC
#include "opencl/ALOpenCL.h"
#endif

#include <fstream>
static inline void KernelValid(ALFloatMatrix* Y, const ALFloatMatrix* X1, const ALFloatMatrix* X2)
{
    ALASSERT(NULL!=Y);
    ALASSERT(NULL!=X1);
    ALASSERT(NULL!=X2);
    auto w = Y->width();
    auto h = Y->height();
    ALASSERT(X1->width() == X2->width());
    ALASSERT(h == X1->height());
    ALASSERT(w == X2->height());
}
class LinearKernel:public ALSVM::Kernel
{
public:
    virtual void vCompute(ALFloatMatrix* Y, const ALFloatMatrix* X1, const ALFloatMatrix* X2) const override
    {
        KernelValid(Y, X1, X2);
        auto w = Y->width();
        auto h = Y->height();
        auto l = X1->width();
        for (int i=0; i<h; ++i)
        {
            ALFLOAT* x1 = (ALFLOAT*)(X1->vGetAddr(i));
            ALFLOAT* y = (ALFLOAT*)(Y->vGetAddr(i));
            for (int j=0; j<w; ++j)
            {
                ALFLOAT sum = 0.0f;
                ALFLOAT* x2 = (ALFLOAT*)(X2->vGetAddr(j));
                for (int k=0; k<l; ++k)
                {
                    sum+= x2[k]*x1[k];
                }
                y[j] = sum;
            }
        }
    }
    virtual void vComputeSST(ALFloatMatrix* Y, const ALFloatMatrix* X) const override
    {
        KernelValid(Y, X, X);
        /*TODO Optimize this*/
        this->vCompute(Y, X, X);
    }
    virtual std::map<std::string, std::string> vPrint() const override
    {
        std::map<std::string, std::string> res;
        res.insert(make_pair("kernel_type", "linear"));
        return res;
    }
};
class RBFKernel:public ALSVM::Kernel
{
public:
    RBFKernel(ALFLOAT gamma):mGama(gamma){}
    virtual ~RBFKernel(){}
    virtual void vCompute(ALFloatMatrix* Y, const ALFloatMatrix* X1, const ALFloatMatrix* X2) const override
    {
        KernelValid(Y, X1, X2);
        auto w = Y->width();
        auto h = Y->height();
        auto l = X1->width();
        for (int i=0; i<h; ++i)
        {
            ALFLOAT* x1 = (ALFLOAT*)(X1->vGetAddr(i));
            ALFLOAT* y = (ALFLOAT*)(Y->vGetAddr(i));
            for (int j=0; j<w; ++j)
            {
                ALFLOAT sum = 0.0f;
                ALFLOAT* x2 = (ALFLOAT*)(X2->vGetAddr(j));
                for (int k=0; k<l; ++k)
                {
                    ALFLOAT d = x2[k]-x1[k];
                    sum+= (d*d);
                }
                y[j] = exp(-mGama*sum);
            }
        }
    }
    virtual void vComputeSST(ALFloatMatrix* Y, const ALFloatMatrix* X) const override
    {
        ALAUTOTIME;
        KernelValid(Y, X, X);
        auto w = Y->width();
        auto l = X->width();
#ifdef ALOPENCL_MAC
        static const char* gRBFSource = KERNEL(
                                               __kernel void rbfsst(__global float *a, __global float* c, size_t w, size_t l, float gamma)
                                               {
                                                   int x = get_global_id(0);
                                                   int y = get_global_id(1);
                                                   if (x > y) return;
                                                   float sum = 0.0;
                                                   int i;
                                                   float df;
                                                   int r = l/4;
                                                   __global float* _a = a + y*l;
                                                   __global float* _b = a + x*l;
                                                   for (i=0; i<r; ++i)
                                                   {
                                                       float4 aa = vload4(i, _a);
                                                       float4 bb = vload4(i, _b);
                                                       float4 cc = (aa - bb)*(aa-bb);
                                                       sum += cc.x + cc.y + cc.z + cc.w;
                                                   }
                                                   i = r*4;
                                                   for (; i< l; ++i)
                                                   {
                                                       sum += (_a[i]-_b[i])*(_a[i]-_b[i]);
                                                   }
                                                   *(c + y*w + x) = exp(-gamma*sum);
                                                   *(c + x*w + y) = exp(-gamma*sum);
                                               }
                                               );
        static cl_kernel gKernel = NULL;
        static ALOpenCL::PrepareWork gPrepare = {
            [&](cl_context context, cl_device_id device) {
                gKernel = ALOpenCL::compileAndBuild(gRBFSource, "rbfsst", context, device);
                return true;
            },
            [&](cl_context c){
                clReleaseKernel(gKernel);
                return true;
            }
        };
        auto run = [=](cl_context context, cl_command_queue queue)
        {
            cl_int errorcode;
            auto sizea = sizeof(ALFLOAT)*w*l;
            auto sizec = sizeof(ALFLOAT)*w*w;
            cl_mem ma = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizea, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem mc = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizec, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            ALDefer([&](){clReleaseMemObject(ma);clReleaseMemObject(mc);});
            {
                errorcode = clEnqueueWriteBuffer(queue, ma, false, 0, sizea, X->vGetAddr()/*FIXME*/, 0, NULL, NULL);
                ALASSERT(errorcode == CL_SUCCESS);
                errorcode = clSetKernelArg(gKernel, 0, sizeof(cl_mem), &ma);
                ALASSERT(errorcode == CL_SUCCESS);
                errorcode = clSetKernelArg(gKernel, 1, sizeof(cl_mem), &mc);
                ALASSERT(errorcode == CL_SUCCESS);
                errorcode = clSetKernelArg(gKernel, 2, sizeof(size_t), &w);
                ALASSERT(errorcode == CL_SUCCESS);
                errorcode = clSetKernelArg(gKernel, 3, sizeof(size_t), &l);
                ALASSERT(errorcode == CL_SUCCESS);
                errorcode = clSetKernelArg(gKernel, 4, sizeof(ALFLOAT), &mGama);
                ALASSERT(errorcode == CL_SUCCESS);
                size_t size[] = {
                    w, w
                };
                errorcode = clEnqueueNDRangeKernel(queue, gKernel, 2, NULL, size, NULL, 0, NULL, NULL);
                ALASSERT(errorcode == CL_SUCCESS);
            }
            clEnqueueReadBuffer(queue, mc, true, 0, sizec, Y->vGetAddr()/*FIXME*/, 0, NULL, NULL);
            return true;
        };
        ALOpenCL& cl = ALOpenCL::getInstance();
        cl.prepare(&gPrepare);
        cl.queueWork(run);
#else
        for (int i=0; i<w; ++i)
        {
            for (int j=i; j<w; ++j)
            {
                ALFLOAT* x1 = X->vGetAddr(i);
                ALFLOAT* x2 = X->vGetAddr(j);
                ALFLOAT sum = 0.0;
                for (int k=0; k<l; ++k)
                {
                    ALFLOAT d = x2[k]-x1[k];
                    sum+= (d*d);
                }
                sum = exp(-mGama*sum);
                Y->vGetAddr(i)[j] = sum;
                Y->vGetAddr(j)[i] = sum;
            }
        }
#endif
    }
    virtual std::map<std::string, std::string> vPrint() const override
    {
        std::map<std::string, std::string> res;
        std::ostringstream os;
        os << mGama;
        res.insert(make_pair("kernel_type", "rbf"));
        res.insert(make_pair("gamma", os.str()));
        return res;
    }
private:
    ALFLOAT mGama;
};



ALSp<ALSVM::Kernel> ALSVMKernelFactory::create(const std::map<std::string, std::string>& heads)
{
    ITER iter;
    iter = heads.find("kernel_type");
    ALASSERT(iter!=heads.end());
    istringstream is(iter->second);
    string name;
    is>>name;
    if (name == "rbf")
    {
        iter = heads.find("gamma");
        ALASSERT(iter!=heads.end());
        istringstream __is(iter->second);
        ALFLOAT gamma;
        __is >> gamma;
        return new RBFKernel(gamma);
    }
    if (name == "linear")
    {
        return new LinearKernel;
    }
    /*TODO*/
    ALASSERT(false);
    return NULL;
}

ALSp<ALSVM::Kernel> ALSVMKernelFactory::createRBF(ALFLOAT gamma)
{
    return new RBFKernel(gamma);
}
