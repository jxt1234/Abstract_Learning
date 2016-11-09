#include "CNNDerivativeFunction.h"
#include <fstream>
namespace ALCNN {
    size_t CNNDerivativeFunction::vInitParameters(ALFloatMatrix* coefficient) const
    {
        auto size = mFirst->getParameterSize();
        if (NULL == coefficient)
        {
            return size;
        }
        auto c = coefficient->vGetAddr();
        ALASSERT(size == coefficient->width());
        ALASSERT(1 == coefficient->height());
        for (size_t i=0; i<size; ++i)
        {
            c[i] = 0.1*ALRandom::rate()-0.05;
        }
        return size;
    }
    CNNDerivativeFunction::CNNDerivativeFunction(ALSp<LayerWrap> first, int outputSize)
    {
        mFirst = first;
        mLast = first->getLastLayer();
        ALASSERT(mFirst.get()!=mLast);
        mOutputSize = outputSize;
        mDecay = 0.001;//TODO
    }
    CNNDerivativeFunction::~CNNDerivativeFunction()
    {
    }

    ALFloatMatrix* CNNDerivativeFunction::vCompute(ALFloatMatrix* coefficient, const ALFloatMatrix* Merge) const
    {
        //ALFORCEAUTOTIME;
        ALSp<ALFloatMatrix> X = ALFloatMatrix::createCropVirtualMatrix(Merge, mOutputSize, 0, Merge->width()-1, Merge->height()-1);
        ALSp<ALFloatMatrix> Y = ALFloatMatrix::createCropVirtualMatrix(Merge, 0, 0, mOutputSize-1, Merge->height()-1);
        ALFloatMatrix* resultDiff = ALFloatMatrix::create(coefficient->width(), coefficient->height());
        
        mFirst->mapParameters(coefficient, 0);
        mFirst->mapParametersDiff(resultDiff, 0);

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
            if (gNumber % 100==0)
            {
                FUNC_PRINT_ALL(mCurrentLoss, f);
            }
        }
        {
            //ALFORCEAUTOTIME;
            mLast->backward(YDiff);
        }
        {
            //ALFORCEAUTOTIME;
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
