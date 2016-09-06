#include "test/GPTest.h"
#include "learn/ALStepRegressor.h"
#include "core/ALBasicExpander.h"
#include "package/ALPackage.h"
#include "core/ALExpanderFactory.h"
#include "learn/ALLearnFactory.h"
#include "learn/ALRegressor.h"
#include "loader/ALStandardLoader.h"
#include <iostream>


class ALStepRegressorTest:public GPTest
{
    public:
        virtual void run()
        {
            ALStandardLoader s;
            ALSp<ALFloatDataChain> c = s.load("bao.txt");
            ALSp<ALLabeldData> data = ALPackageLabled(c.get(), 1.0);
            ALStepRegressor step(10);
            ALSp<ALIExpander> xe = step.train(data.get());
            xe->vPrint(std::cout);
        }
        ALStepRegressorTest(){}
        virtual ~ALStepRegressorTest(){}
};

static GPTestRegister<ALStepRegressorTest> a("ALStepRegressorTest");
