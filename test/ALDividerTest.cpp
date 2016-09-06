#include "test/GPTest.h"
#include "core/ALExpanderFactory.h"
#include "learn/ALDivider.h"
#include "package/ALPackage.h"
#include "loader/ALStandardLoader.h"
#include "learn/ALRegressor.h"
#include <fstream>
#include <iostream>
#include <math.h>
using namespace std;

class ALDividerTest:public GPTest
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
            ALSp<ALIExpander> xe = ALExpanderFactory::createAR(ar);
            ALSp<ALIChainLearner> left = ALIChainLearner::createFromBasic(new ALRegressor, xe);
            ALSp<ALIChainLearner> right = ALIChainLearner::createFromBasic(new ALRegressor, xe);
            ALDivider com(0.7, 1, left.get(), right.get());
            ALSp<ALFloatPredictor> predict = com.vLearn(ldata.get());
            int num = 0;
            ALFLOAT error = 0.0;
            for (auto p : ldata->get())
            {
                auto real = p.first;
                auto pred = predict->vPredict(p.second);
                error += (real - pred)*(real-pred);
                num++;
            }
            cout << num/error << endl;
            ofstream out("output/ALDividerTest.txt");
            for (auto par : ldata->get())
            {
                out << par.first << " : "<<predict->vPredict(par.second) << "\n";
            }
            out.close();
        }
        ALDividerTest(){}
        virtual ~ALDividerTest(){}
};

static GPTestRegister<ALDividerTest> a("ALDividerTest");
