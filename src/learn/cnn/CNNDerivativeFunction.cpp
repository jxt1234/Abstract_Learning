#include "CNNDerivativeFunction.h"
#include <fstream>
namespace ALCNN {
    CNNDerivativeFunction::CNNDerivativeFunction(ALSp<LayerWrap> first, ALSp<LayerWrap> last, int outputSize)
    {
        mFirst = first;
        mLast = last;
        mOutputSize = outputSize;
        mDecay = 0.001;//TODO
    }
    CNNDerivativeFunction::~CNNDerivativeFunction()
    {
    }

    ALFloatMatrix* CNNDerivativeFunction::vCompute(ALFloatMatrix* coefficient, const ALFloatMatrix* Merge) const
    {
        ALFORCEAUTOTIME;
        mFirst->resetBatchSize((int)Merge->height());
        ALSp<ALFloatMatrix> XV = ALFloatMatrix::createCropVirtualMatrix(Merge, mOutputSize, 0, Merge->width()-1, Merge->height()-1);
        ALSp<ALFloatMatrix> YV = ALFloatMatrix::createCropVirtualMatrix(Merge, 0, 0, mOutputSize-1, Merge->height()-1);
        
        ALSp<ALFloatMatrix> X = ALFloatMatrix::create(XV->width(), XV->height());
        ALFloatMatrix::copy(X.get(), XV.get());
        ALSp<ALFloatMatrix> Y = ALFloatMatrix::create(YV->width(), YV->height());
        ALFloatMatrix::copy(Y.get(), YV.get());
        mFirst->setParameters(coefficient, 0);

        ALSp<ALFloatMatrix> YP;
        {
            //ALFORCEAUTOTIME;
            YP = mFirst->forward(X);
        }

        ALASSERT(YP->width() == Y->width());
        ALASSERT(YP->height() == Y->height());

        ALSp<ALFloatMatrix> YDiff = ALFloatMatrix::create(YP->width(), YP->height());
        auto yh = YP->height();
        auto yw = YP->width();
        ALFLOAT loss = 0.0;
        for (int i=0; i<yh; ++i)
        {
            auto dst = YDiff->vGetAddr(i);
            auto srcP = YP->vGetAddr(i);
            auto srcO = Y->vGetAddr(i);
            for (int j=0; j<yw; ++j)
            {
                dst[j] = (srcP[j]-srcO[j]);
                loss += dst[j]*dst[j];
            }
        }
        loss/=YP->height();
        mCurrentLoss = loss;
        if (true)
        {
            static int gNumber = 0;
            gNumber++;
            if (gNumber % 50==0)
            {
                FUNC_PRINT_ALL(mCurrentLoss, f);
            }
        }
        {
            //ALFORCEAUTOTIME;
            mLast->backward(YDiff);
        }
        ALFloatMatrix* resultDiff = ALFloatMatrix::create(coefficient->width(), coefficient->height());
        {
            //ALFORCEAUTOTIME;
            mFirst->readParametersDiff(resultDiff, 0);
            ALFloatMatrix::linear(resultDiff, resultDiff, 1.0f/Merge->height(), coefficient, mDecay);
        }
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
