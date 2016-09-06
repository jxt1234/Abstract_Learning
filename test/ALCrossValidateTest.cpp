#include "test/GPTest.h"
#include "package/ALPackage.h"
#include "core/ALExpanderFactory.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALRegressor.h"
#include "loader/ALStandardLoader.h"
#include <iostream>
using namespace std;

class ALCrossValidateTest:public GPTest
{
    public:
        virtual void run()
        {
            ALStandardLoader s;
            ALSp<ALFloatDataChain> c = s.load("bao.txt");
            ALSp<ALLabeldData> data = ALPackageLabled(c.get(), 1.0);
            ALARStructure ar;
            ar.l = 2;
            ar.w = 1;
            ar.c = 0;
            ar.d = 0;
            ALSp<ALIChainLearner> reg = ALIChainLearner::createFromBasic(new ALRegressor, ALExpanderFactory::createAR(ar));
            cout <<ALLearnFactory::crossValidate(reg.get(), data.get()) <<endl;
        }
        ALCrossValidateTest(){}
        virtual ~ALCrossValidateTest(){}
};

static GPTestRegister<ALCrossValidateTest> a("ALCrossValidateTest");
