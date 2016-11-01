#include "test/GPTest.h"
#include "math/ALIMatrix4DOp.h"
#include <fstream>
class ALMatrixOpTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALIMatrix4DOp> op = ALIMatrix4DOp::create(ALIMatrix4DOp::OPENCL);
            ALSp<ALIMatrix4DOp> op2 = ALIMatrix4DOp::create(ALIMatrix4DOp::BASIC);
            ALIMatrix4DOp::Matrix4D src;
            src.iWidth = 12;
            src.iHeight = 12;
            src.iDepth = 3;
            src.iExpand = 0;
            
            int batchSize = 50;
            src.pOrigin = ALFloatMatrix::create(src.getTotalWidth(), batchSize);
            ALAutoUnRef __src((ALFloatMatrix*)src.pOrigin);
            
            for (int i=0; i<batchSize; ++i)
            {
                auto addr = src.pOrigin->vGetAddr(i);
                for (int j=0; j<src.pOrigin->width(); ++j)
                {
                    addr[j] = ALRandom::rate();
                }
            }
            
            ALIMatrix4DOp::Matrix4D kernel;
            kernel.iWidth = 5;
            kernel.iHeight = 5;
            kernel.iDepth = 3;
            kernel.iExpand = 1;
            int filterSize = 9;
            kernel.pOrigin = ALFloatMatrix::create(kernel.getTotalWidth(), filterSize);
            ALAutoUnRef __kernel((ALFloatMatrix*)kernel.pOrigin);

            
            for (int i=0; i<filterSize; ++i)
            {
                auto addr = kernel.pOrigin->vGetAddr(i);
                for (int j=0; j<kernel.iWidth*kernel.iHeight*kernel.iDepth; ++j)
                {
                    addr[j] = ALRandom::rate()/25.0;
                }
                addr[kernel.iWidth*kernel.iHeight*kernel.iDepth] = 0.5f;
            }
            
            
            ALIMatrix4DOp::Matrix4D dst;
            dst.iWidth = src.iWidth-kernel.iWidth+1;
            dst.iHeight = src.iHeight-kernel.iHeight+1;
            dst.iDepth = filterSize;
            dst.iExpand = 0;
            dst.pOrigin = ALFloatMatrix::create(dst.getTotalWidth(), batchSize);
            ALAutoUnRef __dst((ALFloatMatrix*)dst.pOrigin);

            ALIMatrix4DOp::Matrix4D dst2;
            dst2.iWidth = src.iWidth-kernel.iWidth+1;
            dst2.iHeight = src.iHeight-kernel.iHeight+1;
            dst2.iDepth = filterSize;
            dst2.iExpand = 0;
            dst2.pOrigin = ALFloatMatrix::create(dst2.getTotalWidth(), batchSize);
            ALAutoUnRef __dst2((ALFloatMatrix*)dst2.pOrigin);
            {
                std::ofstream output("output/ALMatrixOpTest_src.txt");
                ALFloatMatrix::print(src.pOrigin, output);
            }
            {
                std::ofstream output("output/ALMatrixOpTest_kernel.txt");
                ALFloatMatrix::print(kernel.pOrigin, output);
            }

            op->vFilter(dst, src, kernel, 1);
            {
                std::ofstream output("output/ALMatrixOpTest_dst.txt");
                ALFloatMatrix::print(dst.pOrigin, output);
            }

            op2->vFilter(dst2, src, kernel, 1);
            {
                std::ofstream output("output/ALMatrixOpTest_dst2.txt");
                ALFloatMatrix::print(dst2.pOrigin, output);
            }
            
            ALASSERT(ALFloatMatrix::theSame(dst.pOrigin, dst2.pOrigin));
            
            
            
            ALIMatrix4DOp::Matrix4D dstDiff;
            dstDiff.iWidth = src.iWidth-kernel.iWidth+1;
            dstDiff.iHeight = src.iHeight-kernel.iHeight+1;
            dstDiff.iDepth = filterSize;
            dstDiff.iExpand = 0;
            dstDiff.pOrigin = ALFloatMatrix::create(dst.getTotalWidth(), batchSize);
            ALAutoUnRef __dstDiff((ALFloatMatrix*)dstDiff.pOrigin);
            
            for (int i=0; i<batchSize; ++i)
            {
                auto addr = dstDiff.pOrigin->vGetAddr(i);
                for (int j=0; j<dstDiff.pOrigin->width(); ++j)
                {
                    addr[j] = ALRandom::rate()/3.0f;
                }
            }
            {
                std::ofstream output("output/ALMatrixOpTest_dstDiff.txt");
                ALFloatMatrix::print(dstDiff.pOrigin, output);
            }
            
            
            ALIMatrix4DOp::Matrix4D kernelDiff = kernel;
            kernelDiff.pOrigin = ALFloatMatrix::create(kernelDiff.getTotalWidth(), kernel.pOrigin->height());
            ALAutoUnRef __kernelDiff((ALFloatMatrix*)kernelDiff.pOrigin);
            ALFloatMatrix::zero((ALFloatMatrix*)kernelDiff.pOrigin);

            
            ALIMatrix4DOp::Matrix4D srcDiff = src;
            srcDiff.pOrigin = ALFloatMatrix::create(srcDiff.getTotalWidth(), srcDiff.pOrigin->height());
            ALAutoUnRef __srcDiff((ALFloatMatrix*)srcDiff.pOrigin);
            ALFloatMatrix::zero((ALFloatMatrix*)srcDiff.pOrigin);
            
            ALIMatrix4DOp::Matrix4D kernelDiff2 = kernel;
            kernelDiff2.pOrigin = ALFloatMatrix::create(kernelDiff2.getTotalWidth(), kernel.pOrigin->height());
            ALAutoUnRef __kernelDiff2((ALFloatMatrix*)kernelDiff2.pOrigin);
            ALFloatMatrix::zero((ALFloatMatrix*)kernelDiff2.pOrigin);
            
            
            ALIMatrix4DOp::Matrix4D srcDiff2 = src;
            srcDiff2.pOrigin = ALFloatMatrix::create(srcDiff.getTotalWidth(), srcDiff.pOrigin->height());
            ALAutoUnRef __srcDiff2((ALFloatMatrix*)srcDiff2.pOrigin);
            ALFloatMatrix::zero((ALFloatMatrix*)srcDiff2.pOrigin);
            
            
            op->vDeterFilter(dstDiff, dst, src, srcDiff, kernel, kernelDiff, 1);
            op2->vDeterFilter(dstDiff, dst, src, srcDiff2, kernel, kernelDiff2, 1);
            
            
            {
                std::ofstream output("output/ALMatrixOpTest_kernelDiff.txt");
                ALFloatMatrix::print(kernelDiff.pOrigin, output);
                std::ofstream output2("output/ALMatrixOpTest_kernelDiff2.txt");
                ALFloatMatrix::print(kernelDiff2.pOrigin, output2);
            }
            {
                std::ofstream output("output/ALMatrixOpTest_srcDiff.txt");
                ALFloatMatrix::print(srcDiff.pOrigin, output);
                std::ofstream output2("output/ALMatrixOpTest_srcDiff2.txt");
                ALFloatMatrix::print(srcDiff2.pOrigin, output2);
            }
            ALASSERT(ALFloatMatrix::theSame(srcDiff2.pOrigin, srcDiff.pOrigin));
            ALASSERT(ALFloatMatrix::theSame(kernelDiff2.pOrigin, kernelDiff.pOrigin));


        }
        ALMatrixOpTest(){}
        virtual ~ALMatrixOpTest(){}
};

static GPTestRegister<ALMatrixOpTest> a("ALMatrixOpTest");
