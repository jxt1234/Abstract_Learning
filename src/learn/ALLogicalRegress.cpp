#include "learn/ALLogicalRegress.h"
#include <math.h>

#ifdef ALOPENCL_MAC
#include "opencl/ALOpenCL.h"
static const char* gComoputeThetha = KERNEL(
                                            __kernel void thetha(__global float *thetha, __global float* x, __global float* y, __global float* weight, size_t w, size_t stride, float alpha)
                                            {
                                                int i = get_global_id(0);
                                                __global float* _x = x + stride*i;
                                                __global float* the = thetha + w*i;
                                                int k;
                                                float sum = 0.0;
                                                for (k=0; k<w; ++k)
                                                {
                                                    sum += weight[k]*_x[k];
                                                }
                                                sum = 1.0/(1.0+exp(-sum));
                                                sum = sum - y[i];
                                                for (k=0; k<w; ++k)
                                                {
                                                    the[k] = sum*_x[k]*alpha;
                                                }
                                            }
                                            );
static const char* gSumThetha = KERNEL(
                                       __kernel void sum_theta(__global float* sum_result, __global float *thetha, __global float* x, size_t w, size_t stride)
                                       {
                                           int i = get_global_id(0);
                                           int j = get_global_id(1);
                                           __global float* _x = x + i*w*stride + j;
                                           __global float* _t = thetha + i*w;
                                           int k;
                                           float sum = 0.0;
                                           for (k=0; k<w; ++k)
                                           {
                                               sum += _x[k*stride]*_x[k];
                                           }
                                           sum_result[i*w+j] = sum;
                                       }
                                       );

static cl_kernel gThethaKernel = NULL;
static cl_kernel gSumKernel = NULL;
static ALOpenCL::PrepareWork gPrepare = {
    [](cl_context context, cl_device_id device) {
        gThethaKernel = ALOpenCL::compileAndBuild(gComoputeThetha, "thetha", context, device);
        gSumKernel = ALOpenCL::compileAndBuild(gSumThetha, "sum_theta", context, device);
        return true;
    },
    [](cl_context c){
        clReleaseKernel(gThethaKernel);
        clReleaseKernel(gSumKernel);
        return true;
    }
};
static void setup(cl_mem X, cl_mem Y, cl_mem thetha, cl_mem weight, size_t w, size_t h, size_t sumunit, ALFLOAT alpha, cl_command_queue queue)
{
    cl_int error;
    error = clSetKernelArg(gThethaKernel, 0, sizeof(cl_mem), &thetha);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gThethaKernel, 1, sizeof(cl_mem), &X);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gThethaKernel, 2, sizeof(cl_mem), &Y);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gThethaKernel, 3, sizeof(cl_mem), &weight);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gThethaKernel, 4, sizeof(size_t), &w);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gThethaKernel, 5, sizeof(size_t), &w);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gThethaKernel, 6, sizeof(ALFLOAT), &alpha);
    ALASSERT(error == CL_SUCCESS);
    error = clEnqueueNDRangeKernel(queue, gThethaKernel, 1, NULL, &h, NULL, 0, NULL, NULL);
#if 0
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gSumKernel, 0, sizeof(cl_mem), &thetha);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gSumKernel, 1, sizeof(cl_mem), &X);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gSumKernel, 2, sizeof(size_t), &sumunit);
    ALASSERT(error == CL_SUCCESS);
    error = clSetKernelArg(gSumKernel, 3, sizeof(size_t), &w);
    ALASSERT(error == CL_SUCCESS);
    size_t gsizeofsum[2] = {h/sumunit, w};
    error = clEnqueueNDRangeKernel(queue, gSumKernel, 2, NULL, gsizeofsum, NULL, 0, NULL, NULL);
    ALASSERT(error == CL_SUCCESS);
#endif
}
#endif
static ALFLOAT computeThetha(const ALFLOAT* x, const ALFLOAT* _w, size_t w)
{
    ALFLOAT thetha = 0.0;
    //_w[0];
    for (int j=0; j<w; ++j)
    {
        thetha += _w[j]*x[j];
    }
    thetha = 1.0/(1.0 + exp(-thetha));
    return thetha;
}
ALLogicalRegress::ALLogicalRegress(int iter, ALFLOAT alpha)
{
    mMaxIter = iter;
    mAlpha = alpha;
}
ALLogicalRegress::~ALLogicalRegress()
{
}
ALIMatrixPredictor* ALLogicalRegress::vLearn(const ALFloatMatrix* X, const ALFloatMatrix* Y) const
{
    ALSp<ALFloatMatrix> w = learn(X, Y, mMaxIter, mAlpha);
    class LogicalJudger:public ALIMatrixPredictor
    {
    public:
        LogicalJudger(ALSp<ALFloatMatrix> W):mW(W)
        {
        }
        virtual ~LogicalJudger(){}
        virtual void vPredict(const ALFloatMatrix* X, ALFloatMatrix* Y/*Output*/) const override
        {
            ALASSERT(X->height() == Y->height());
            ALASSERT(Y->width()>=1);
            for (auto i =0; i<X->height(); ++i)
            {
                auto x = X->vGetAddr(i);
                auto y = Y->vGetAddr(i);
                y[0] = computeThetha(x, mW->vGetAddr(), X->width());
            }
        }
        virtual void vPrint(std::ostream& output) const override
        {
            output << "<LogicModel>\n";
            ALFloatMatrix::print(mW.get(), output);
            output << "</LogicModel>\n";
        }
    private:
        ALSp<ALFloatMatrix> mW;
    };
    return new LogicalJudger(w);
}

ALSp<ALFloatMatrix> ALLogicalRegress::learn(const ALFloatMatrix* X, const ALFloatMatrix* Y, size_t maxiter, ALFLOAT alpha)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=Y);
    ALASSERT(maxiter>=1);
    ALASSERT(X->height() == Y->height());
    auto w = X->width();
    auto h = Y->height();
    /*TODO let user adust the alpha*/
    ALSp<ALFloatMatrix> W = ALFloatMatrix::create(w, 1);
    ALSp<ALFloatMatrix> WOLD = ALFloatMatrix::create(w, 1);
    ALFLOAT* _w = W->vGetAddr();
    ::memset(_w, 0, sizeof(ALFLOAT)*(w));
    ALFLOAT* wold = WOLD->vGetAddr();
    ALAUTOSTORAGE(thetha, ALFLOAT, h);
#ifdef ALOPENCL_MAC
    if (0)
    //if (h>500)
    {
        auto run = [=](cl_context context, cl_command_queue queue)
        {
            cl_int errorcode;
            auto size_weight = sizeof(ALFLOAT)*w;
            auto size_thetha = sizeof(ALFLOAT)*h*w;
            auto size_X = sizeof(ALFLOAT)*h*w;
            auto size_Y = sizeof(ALFLOAT)*h;
            cl_mem m_weight = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, size_weight, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem m_thetha = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, size_thetha, NULL, &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem m_X = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size_X, X->vGetAddr(), &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            cl_mem m_Y = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR, size_Y, Y->vGetAddr(), &errorcode);
            ALASSERT(CL_SUCCESS == errorcode);
            clEnqueueWriteBuffer(queue, m_weight, false, 0, size_weight, _w, 0, NULL, NULL);
            ALDefer([&]()
                    {
                        clReleaseMemObject(m_weight);
                        clReleaseMemObject(m_thetha);
                        clReleaseMemObject(m_X);
                        clReleaseMemObject(m_Y);
                    });
            for (size_t k=0; k<maxiter; ++k)
            {
                const int sumunit = 16;
                setup(m_X, m_Y, m_thetha, m_weight, w, h, sumunit, alpha, queue);
                ALFLOAT* c_w = (ALFLOAT*)clEnqueueMapBuffer(queue, m_weight, true, CL_MAP_READ | CL_MAP_WRITE, 0, size_weight, 0, NULL, NULL, NULL);
                ALFLOAT* c_t = (ALFLOAT*)clEnqueueMapBuffer(queue, m_thetha, true, CL_MAP_READ, 0, size_thetha, 0, NULL, NULL, NULL);
                for (size_t j=0; j<w; ++j)
                {
                    ALFLOAT sum = 0.0;
                    for (size_t i=0; i<h; ++i)
                    {
                        auto thetha_l = c_t + w*i;
                        sum += thetha_l[j]/(ALFLOAT)h;
                    }
                    c_w[j]-=sum;
                }
                clEnqueueUnmapMemObject(queue, m_weight, c_w, 0, NULL, NULL);
                clEnqueueUnmapMemObject(queue, m_thetha, c_t, 0, NULL, NULL);
            }
            clEnqueueReadBuffer(queue, m_weight, true, 0, size_weight, _w, 0, NULL, NULL);
            return true;
        };
        ALOpenCL& cl = ALOpenCL::getInstance();
        cl.prepare(&gPrepare);
        cl.queueWork(run);
        return W;
    }
#endif
    /*Start for iterator*/
    for (size_t k=0; k<maxiter; ++k)
    {
        ::memcpy(wold, _w, (w)*sizeof(ALFLOAT));
        /*Wnew = Wold - a*sum((thetha-y)*x)*/
        for (size_t i=0; i<h; ++i)
        {
            /*thetha = 1/(1+exp(W*(1,XT)))*/
            ALFLOAT y = *(Y->vGetAddr(i));
            ALFLOAT* x = X->vGetAddr(i);
            thetha[i] = computeThetha(x, wold, w)-y;
        }
        for (size_t i=0; i<h; ++i)
        {
            ALFLOAT* x = X->vGetAddr(i);
            for (size_t j=0; j<w; ++j)
            {
                _w[j] -= (thetha[i])*x[j]*alpha;
            }
        }
        /*If W and WOLD is the same: norm(W-WOLD)<0.001, break*/
        ALFLOAT error = 0.0;
        for (size_t i=0; i<w; ++i)
        {
            error += (_w[i]-wold[i])*(_w[i]-wold[i]);
        }
        error/=(ALFLOAT)(w+1);
        if (ZERO(error))
        {
            break;
        }
    }
    return W;
}
ALSp<ALFloatMatrix> ALLogicalRegress::predict(const ALFloatMatrix* X, const ALFloatMatrix* W)
{
    ALASSERT(NULL!=X);
    ALASSERT(NULL!=W);
    ALASSERT(X->width() == W->width()-1);
    auto w = X->width();
    auto h = X->height();
    ALSp<ALFloatMatrix> Y = ALFloatMatrix::create(1, h);
    ALFLOAT* _w = W->vGetAddr();
    for (int i=0; i<h; ++i)
    {
        ALFLOAT* x = X->vGetAddr(i);
        *(Y->vGetAddr(i)) = computeThetha(x, _w, w);
    }
    return Y;
}
