#include "CNNDerivativeFunction.h"
#include <fstream>
namespace ALCNN {
    CNNDerivativeFunction::CNNDerivativeFunction(ALSp<LayerWrap> first, ALSp<LayerWrap> last, int outputSize)
    {
        mFirst = first;
        mLast = last;
        mOutputSize = outputSize;
    }
    CNNDerivativeFunction::~CNNDerivativeFunction()
    {
    }

    ALFloatMatrix* CNNDerivativeFunction::vCompute(ALFloatMatrix* coefficient, const ALFloatMatrix* Merge) const
    {
        mFirst->resetBatchSize((int)Merge->height());
        ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(Merge, mOutputSize, 0, Merge->width()-1, Merge->height()-1);
        ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(Merge, 0, 0, mOutputSize-1, Merge->height()-1);
        mFirst->setParameters(coefficient, 0);

        auto YP = mFirst->forward(X);
        ALASSERT(YP->width() == Y->width());
        ALASSERT(YP->height() == Y->height());

        ALSp<ALFloatMatrix> YDiff = ALFloatMatrix::create(YP->width(), YP->height());
        auto yh = YP->height();
        auto yw = YP->width();
        for (int i=0; i<yh; ++i)
        {
            auto dst = YDiff->vGetAddr(i);
            auto srcP = YP->vGetAddr(i);
            auto srcO = Y->vGetAddr(i);
            for (int j=0; j<yw; ++j)
            {
                dst[j] = (srcP[j]-srcO[j]);
            }
        }
        mLast->backward(YDiff);
        ALFloatMatrix* resultDiff = ALFloatMatrix::create(coefficient->width(), coefficient->height());
        mFirst->readParametersDiff(resultDiff, 0);
        if (false)
        {
            std::ofstream outputX("/Users/jiangxiaotang/Documents/Abstract_Learning/.XX.txt");
            ALFloatMatrix::print(X.get(), outputX);
            std::ofstream output("/Users/jiangxiaotang/Documents/Abstract_Learning/temp.txt");
            ALFloatMatrix::print(YP.get(), output);
            std::ofstream outputY("/Users/jiangxiaotang/Documents/Abstract_Learning/temp_Y.txt");
            ALFloatMatrix::print(Y.get(), outputY);
            std::ofstream outputYDiff("/Users/jiangxiaotang/Documents/Abstract_Learning/temp_Y_diff.txt");
            ALFloatMatrix::print(YDiff.get(), outputYDiff);
            
            std::ofstream outputc("/Users/jiangxiaotang/Documents/Abstract_Learning/temp_c.txt");
            ALFloatMatrix::print(coefficient, outputc);
            std::ofstream outputcc("/Users/jiangxiaotang/Documents/Abstract_Learning/temp_cd.txt");
            ALFloatMatrix::print(resultDiff, outputcc);
        }
        return resultDiff;
    }
}
