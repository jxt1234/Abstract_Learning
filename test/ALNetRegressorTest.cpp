#include "test/GPTest.h"
#include <iostream>
#include <fstream>
#include "package/ALPackage.h"
#include "core/ALExpanderFactory.h"
#include "core/ALLabeldMethodFactory.h"
#include "loader/ALStandardLoader.h"
#include "learn/ALLogicalRegress.h"
#include "learn/ALIChainLearner.h"
#include "learn/ALNetRegressor.h"
#include "math/ALFloatMatrix.h"
using namespace std;

class ALNetRegressorTest:public GPTest
{
    public:
        virtual void run()
        {
            ALSp<ALFloatDataChain> c = ALStandardLoader::load("bao.txt");
            ALSp<ALLabeldData> labeldData = ALPackageLabled(c.get(), 1.0);
            ALARStructure ar;
            ar.l = 3;
            ar.w = 1;
            ar.d = 1;
            ar.c = 0;
            ALSp<ALIExpander> xe = ALExpanderFactory::createAR(ar);
            ALSp<ALIChainLearner> l = ALIChainLearner::createFromBasic(new ALNetRegressor(2), xe);
            ALSp<ALFloatPredictor> p = l->vLearn(labeldData.get());
            ofstream out("output/ALNetRegressorTest.txt");
            for (auto labelpoint : labeldData->get())
            {
                ALFLOAT y_real = labelpoint.first;
                auto pred = p->vPredict(labelpoint.second);
                out << "Predict: "<<pred << ", Real is "<<y_real<<endl;
            }
        }
        ALNetRegressorTest(){}
        virtual ~ALNetRegressorTest(){}
};

static GPTestRegister<ALNetRegressorTest> a("ALNetRegressorTest");
