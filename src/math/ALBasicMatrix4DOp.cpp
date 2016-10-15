#include "ALBasicMatrix4DOp.h"
#include <fstream>

static ALSp<ALFloatMatrix> _expand(const ALBasicMatrix4DOp::Matrix4D& src, const ALBasicMatrix4DOp::Matrix4D& kernelData, int stride, int z)
{
    ALASSERT(kernelData.iExpand == 1);
    /*GEMM method, like caffe*/
    auto kernelWidth = kernelData.iWidth;
    auto kernelHeight = kernelData.iHeight;
    auto kernelDepth = kernelData.iDepth;
    auto dstWidth = (src.iWidth-kernelWidth)/stride+1;
    auto dstHeight = (src.iHeight-kernelHeight)/stride+1;

    auto srcBatch = src.pOrigin->vGetAddr(z);
    //Expand Begin
    auto srcExpandH = kernelData.pOrigin->width();
    auto srcExpandW = dstWidth*dstHeight;
    ALSp<ALFloatMatrix> srcExpand = ALFloatMatrix::create(srcExpandW, srcExpandH);
    for (int i=0; i<kernelDepth; ++i)
    {
        ALFLOAT* srcInDepth = srcBatch + src.iHeight*src.iWidth*i;
        for (int j=0; j<kernelHeight; ++j)
        {
            for (int k=0; k<kernelWidth; ++k)
            {
                auto line = i*kernelWidth*kernelHeight + j*kernelWidth + k;
                auto dstImage = srcExpand->vGetAddr(line);
                for (int di=0; di<dstHeight; ++di)
                {
                    auto dstLine = dstImage + di*dstWidth;
                    auto srcLine = srcInDepth + (di*stride+j)*src.iWidth;
                    for (int dj=0; dj<dstWidth; ++dj)
                    {
                        dstLine[dj] = srcLine[dj*stride+k];
                    }
                }
            }
        }
    }
    /*The last line is all 1*/
    auto lastLine = srcExpand->vGetAddr(kernelData.pOrigin->width()-1);
    for (int i=0; i<srcExpandW; ++i)
    {
        lastLine[i] = 1.0f;
    }
    //Expand End

    return srcExpand;
}

static void _reduceAdd(const ALFloatMatrix* srcExpand, const ALBasicMatrix4DOp::Matrix4D& src, const ALBasicMatrix4DOp::Matrix4D& kernelData, int stride, int z)
{
    auto kernelWidth = kernelData.iWidth;
    auto kernelHeight = kernelData.iHeight;
    auto kernelDepth = kernelData.iDepth;
    auto dstWidth = (src.iWidth-kernelWidth)/stride+1;
    auto dstHeight = (src.iHeight-kernelHeight)/stride+1;
    auto srcBatch = src.pOrigin->vGetAddr(z);
    for (int i=0; i<kernelDepth; ++i)
    {
        ALFLOAT* srcInDepth = srcBatch + src.iHeight*src.iWidth*i;
        for (int j=0; j<kernelHeight; ++j)
        {
            for (int k=0; k<kernelWidth; ++k)
            {
                auto line = i*kernelWidth*kernelHeight + j*kernelWidth + k;
                auto dstImage = srcExpand->vGetAddr(line);
                for (int di=0; di<dstHeight; ++di)
                {
                    auto dstLine = dstImage + di*dstWidth;
                    auto srcLine = srcInDepth + (di*stride+j)*src.iWidth;
                    for (int dj=0; dj<dstWidth; ++dj)
                    {
                        srcLine[dj*stride+k] += dstLine[dj];
                    }
                }
            }
        }
    }
}


void ALBasicMatrix4DOp::vFilter(Matrix4D& output, const Matrix4D& src, const Matrix4D& kernelData, int stride) const
{
    ALASSERT(src.valid());
    ALASSERT(kernelData.valid());
    ALASSERT(stride>=1);
    ALASSERT(kernelData.iExpand == 1);
    ALASSERT(src.iExpand == 0);
    auto kernelWidth = kernelData.iWidth;
    auto kernelHeight = kernelData.iHeight;
    auto dstWidth = (src.iWidth-kernelWidth)/stride+1;
    auto dstHeight = (src.iHeight-kernelHeight)/stride+1;
    
    ALASSERT(output.iDepth == kernelData.pOrigin->height());
    ALASSERT(output.iHeight == dstHeight);
    ALASSERT(output.iWidth == dstWidth);
    ALASSERT(output.iExpand == 0);
    ALASSERT(output.valid());
    
    auto batchSize = src.pOrigin->height();
    for (int z=0; z<batchSize; ++z)
    {
        /*GEMM method, like caffe*/
        ALSp<ALFloatMatrix> srcExpand = _expand(src, kernelData, stride, z);
        
        //Product
        ALFLOAT* dstBatch = output.pOrigin->vGetAddr(z);
        ALFloatMatrix::productBasic(dstBatch, output.iWidth*output.iHeight, kernelData.pOrigin->vGetAddr(), kernelData.pOrigin->width(), srcExpand->vGetAddr(), srcExpand->width(), output.iWidth*output.iHeight, output.iDepth, kernelData.pOrigin->width());
    }
}


void ALBasicMatrix4DOp::vDeterFilter(const Matrix4D& dstDiff, const Matrix4D& dst, const Matrix4D& src, Matrix4D& srcDiff, const Matrix4D& kernelData, Matrix4D& kernelDataDiff, int stride) const
{
    ALASSERT(dstDiff.valid());
    ALASSERT(dst.valid());
    ALASSERT(kernelData.valid());
    ALASSERT(kernelDataDiff.valid());
    ALASSERT(kernelDataDiff.getTotalWidth() == kernelData.getTotalWidth());
    ALASSERT(stride>=1);
    ALASSERT(kernelData.iExpand == 1);
    ALASSERT(dst.iExpand == 0);
    auto batchSize = dst.pOrigin->height();
    
    ALFloatMatrix::zero(kernelDataDiff.getMutable());
    if (NULL!=srcDiff.pOrigin)
    {
        ALFloatMatrix::zero(srcDiff.getMutable());
    }
    ALSp<ALFloatMatrix> kernelDataT = ALFloatMatrix::transpose(kernelData.pOrigin);
    ALFLOAT rate = 1.0/src.getTotalWidth();
    for (int z=0; z<batchSize; ++z)
    {
        auto outputDiffBatch = dstDiff.pOrigin->vGetAddr(z);
        ALSp<ALFloatMatrix> outputDiffM = ALFloatMatrix::createRefMatrix(outputDiffBatch, dstDiff.iWidth*dstDiff.iHeight, dstDiff.iDepth);
        ALSp<ALFloatMatrix> inputExpandTemp = _expand(src, kernelData, stride, z);
        ALSp<ALFloatMatrix> kernelDiffTemp = ALFloatMatrix::productT(outputDiffM.get(), inputExpandTemp.get());
        if (false)
        {
            std::ofstream output_inputT("/Users/jiangxiaotang/Documents/Abstract_Learning/inputTotal.txt");
            ALFloatMatrix::print(src.pOrigin, output_inputT);
            ALSp<ALFloatMatrix> ett = ALFloatMatrix::transpose(inputExpandTemp.get());
            std::ofstream output_ii("/Users/jiangxiaotang/Documents/Abstract_Learning/inputExpandTemp.txt");
            ALFloatMatrix::print(ett.get(), output_ii);
            std::ofstream output_ie("/Users/jiangxiaotang/Documents/Abstract_Learning/outputDiffM.txt");
            ALFloatMatrix::print(outputDiffM.get(), output_ie);
            std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/kernelDiffTemp.txt");
            ALFloatMatrix::print(kernelDiffTemp.get(), output);
            std::ofstream outputvvv("/Users/jiangxiaotang/Documents/Abstract_Learning/kernelDiff.txt");
            ALFloatMatrix::print(kernelDataDiff.pOrigin, outputvvv);
        }
        ALFloatMatrix::linear(kernelDataDiff.getMutable(), kernelDiffTemp.get(), rate, kernelDataDiff.pOrigin, 1.0);

        if (NULL!=srcDiff.pOrigin)
        {
            inputExpandTemp = ALFloatMatrix::product(kernelDataT.get(), outputDiffM.get());
            _reduceAdd(inputExpandTemp.get(), srcDiff, kernelData, stride, z);
        }
    }
}
