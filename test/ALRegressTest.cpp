#include "test/GPTest.h"
#include "package/ALPackage.h"
#include "core/ALLabeldMethodFactory.h"
#include "core/ALExpanderFactory.h"
#include "learn/ALRegressor.h"
#include "learn/ALIChainLearner.h"
#include "loader/ALStandardLoader.h"
#include <fstream>
#include <iostream>
#include <math.h>
using namespace std;

class ALRegressTest:public GPTest
{
    public:
        virtual void run()
        {
            ALStandardLoader s;
            ALSp<ALFloatDataChain> c = s.load("bao.txt");
            ALSp<ALLabeldData> ldata = ALPackageLabled(c.get(), 1.0);
            ALARStructure ar;
            ar.l = 1;
            ar.w = 1;
            ar.c = 1;
            ar.d = 0;
            ALSp<ALIChainLearner> reg = ALIChainLearner::createFromBasic(new ALRegressor, ALExpanderFactory::createAR(ar));
            ALSp<ALFloatPredictor> p = reg->vLearn(ldata.get());
            p->vPrint(cout);
            ALIChainLearner::Error res = ALIChainLearner::computeError(ldata.get(), p.get());
            ALFLOAT error = res.sum;
            int num = res.num;
            cout << num/error << endl;
            ofstream out("output/ALRegressTest.txt");
            for (auto dp : ldata->get())
            {
                out << dp.first << " : " << p->vPredict(dp.second) <<endl;
            }
            out.close();
        }
        ALRegressTest(){}
        virtual ~ALRegressTest(){}
};

static GPTestRegister<ALRegressTest> a("ALRegressTest");
