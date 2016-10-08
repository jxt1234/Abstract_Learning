#include "test/GPTest.h"
#include "math/ALIMatrix4DOp.h"
#include <fstream>
class ALMatrixOpTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALIMatrix4DOp> op = ALIMatrix4DOp::create();
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


            op->vFilter(dst, src, kernel, 1);
            
            {
                std::ofstream output("output/ALMatrixOpTest_src.txt");
                ALFloatMatrix::print(src.pOrigin, output);
            }
            {
                std::ofstream output("output/ALMatrixOpTest_dst.txt");
                ALFloatMatrix::print(dst.pOrigin, output);
            }
            {
                std::ofstream output("output/ALMatrixOpTest_kernel.txt");
                ALFloatMatrix::print(kernel.pOrigin, output);
            }
            
            
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
            
            op->vDeterFilter(dstDiff, src, srcDiff, kernel, kernelDiff, 1);
            
            {
                std::ofstream output("output/ALMatrixOpTest_kernelDiff.txt");
                ALFloatMatrix::print(kernelDiff.pOrigin, output);
            }
            {
                std::ofstream output("output/ALMatrixOpTest_srcDiff.txt");
                ALFloatMatrix::print(srcDiff.pOrigin, output);
            }

        }
        ALMatrixOpTest(){}
        virtual ~ALMatrixOpTest(){}
};

static GPTestRegister<ALMatrixOpTest> a("ALMatrixOpTest");
