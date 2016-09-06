#include "test/GPTest.h"
#include "learn/ALCGPRegressor.h"
#include "core/ALARStructure.h"
#include "core/ALBasicExpander.h"
#include "package/ALPackage.h"
#include "core/ALExpanderFactory.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALRegressor.h"
#include "loader/ALStandardLoader.h"
#include <iostream>
using namespace std;


class ALCGPRegressorTest:public GPTest
{
public:
    virtual void run()
    {
        ALStandardLoader s;
        ALSp<ALFloatDataChain> c = s.load("bao.txt");
        ALSp<ALLabeldData> data = ALPackageLabled(c.get(), 1.0);
        
        int w = 20;
        int h = 20;
        ALAUTOSTORAGE(parameter, double, w*h*2);
        ALSp<ALCGPRegressor> learner = new ALCGPRegressor(w, h);
        for (int k=0; k<10; ++k)
        {
            for (int i=0; i<w*h*2; ++i)
            {
                parameter[i] = ALRandom::rate();
            }
            learner->map(parameter, w*h*2);
            cout <<k<<":"<<ALLearnFactory::crossValidate(learner.get(), data.get()) <<endl;
        }
    }
    ALCGPRegressorTest(){}
    virtual ~ALCGPRegressorTest(){}
};

static GPTestRegister<ALCGPRegressorTest> a("ALCGPRegressorTest");
