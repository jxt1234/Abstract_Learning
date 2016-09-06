#include "test/GPTest.h"
#include "learn/ALSVMLearner.h"
#include "learn/ALIChainLearner.h"
#include "core/ALExpanderFactory.h"
#include "core/ALILabeldMethod.h"
#include "core/ALLabeldMethodFactory.h"
#include "loader/ALStandardLoader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "UpthreadLabeld.h"
using namespace std;

static int test_main()
{
    ALStandardLoader s;
    ALSp<ALFloatDataChain> c = s.load("bao.txt");
    ALARStructure ar;
    ar.l = 3;
    ar.w = 3;
    ar.d = 1;
    ar.c = 0;
    ALSp<ALIExpander> xe = ALExpanderFactory::createAR(ar);
    ALSp<ALILabeldMethod> lm = new TrendLabeld(1, -1);
    ALSp<ALLabeldData> labeldData = ALLabeldMethodFactory::delayLabel(c->get(), lm.get(), 1);
    ALSp<ALIChainLearner> l = ALIChainLearner::createFromBasic(new ALSVMLearner, xe);

    ALSp<ALFloatPredictor> p = l->vLearn(labeldData.get());

    ofstream out("output/ALSVMTest.txt");
    size_t sum = 0;
    size_t correct = 0;
    for (auto labelpoint : labeldData->get())
    {
        ALFLOAT y_real = labelpoint.first;
        auto pred = p->vPredict(labelpoint.second);
        out << "Predict: "<<pred << ", Real is "<<y_real<<endl;
        sum++;
        if (ZERO(pred-y_real))
        {
            correct ++;
        }
    }
    cout << correct << "/"<<sum<<endl;
    return 1;
}

class ALSVMTest:public GPTest
{
    public:
        virtual void run()
        {
            test_main();
        }
        ALSVMTest(){}
        virtual ~ALSVMTest(){}
};

static GPTestRegister<ALSVMTest> a("ALSVMTest");
